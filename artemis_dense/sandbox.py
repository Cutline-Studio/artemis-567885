"""Loader for artemis_dense/sandbox.env: the isolated test environment for anything Artemis
generates or modifies before promotion.

  sb = SandboxEnv.load()              # parse sandbox.env (no shell; ${VAR} expanded from the file itself)
  with sb.session():                   # mkdirs under /tmp/artemis-sandbox, audit start/end, wipe on exit
      ex = CodeExecEngine(sb.work_dir("exec"), sandbox=sb)   # limits + env + denied paths applied

What is enforced by the engines that load it (code_exec, self_author, self_train, watch practice):
  * all writes/run dirs must be under ARTEMIS_SANDBOX_ROOT (else SandboxViolation)
  * ARTEMIS_PROD_PATHS_DENY (/workspace, /home/box, /cursor/stores) is never bound into a bwrap
    child and check_path() refuses them for engine-level reads/writes
  * ARTEMIS_NETWORK=off -> code_exec REQUIRES bubblewrap (--unshare-all: no network namespace);
    the best-effort non-bwrap fallback is refused under ARTEMIS_UNTRUSTED=1
  * resource limits -> SandboxLimits (timeout, address space, nofile, output cap, nproc)
  * ARTEMIS_ALLOW_PUSH/PAYMENTS/EMAIL/DEPLOY/GPU_RENTAL=0 -> deny(<capability>) raises
  * every load / session / violation is appended to ARTEMIS_AUDIT_LOG (JSONL; wiped with the root
    on exit unless keep_audit_copy is given)
The child process env is ONLY the sandbox vars below + a minimal PATH/LANG; no parent secrets.
"""
import contextlib, json, os, re, shutil, time
from typing import Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox.env")
CHILD_KEYS = ("ARTEMIS_ENV", "ARTEMIS_UNTRUSTED", "ARTEMIS_SANDBOX_LABEL", "ARTEMIS_NETWORK", "ARTEMIS_MODEL_SIZE",
              "ARTEMIS_DEVICE", "ARTEMIS_SEED", "ARTEMIS_DETERMINISTIC", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")


class SandboxViolation(PermissionError):
    pass


def parse_env(text: str, overrides: Optional[dict] = None) -> dict:
    """Overrides replace a key where it is defined, so later ${KEY} references see the override."""
    out, overrides = {}, dict(overrides or {})
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        v = re.sub(r"\$\{(\w+)\}", lambda m: out.get(m.group(1), ""), v)
        k = k.strip()
        out[k] = str(overrides.pop(k)) if k in overrides else v
    out.update({k: str(v) for k, v in overrides.items()})
    return out


class SandboxEnv:
    def __init__(self, vars: dict, source: str = "<dict>"):
        self.vars, self.source = vars, source
        self.root = os.path.realpath(vars["ARTEMIS_SANDBOX_ROOT"])
        if self.root in ("/", "") or not self.root.startswith("/tmp/"):
            raise SandboxViolation(f"ARTEMIS_SANDBOX_ROOT must be under /tmp, got {self.root!r}")
        self.deny_paths = [os.path.realpath(p) for p in vars.get("ARTEMIS_PROD_PATHS_DENY", "").split(",") if p]

    @classmethod
    def load(cls, path: str = DEFAULT_PATH, **overrides) -> "SandboxEnv":
        v = parse_env(open(path, encoding="utf-8").read(), overrides)
        sb = cls(v, path)
        return sb

    # -------------------------------------------------------------- flags / limits
    def flag(self, key: str) -> bool:
        return self.vars.get(key, "0") in ("1", "true", "on", "yes")

    @property
    def untrusted(self):
        return self.flag("ARTEMIS_UNTRUSTED")

    @property
    def network_off(self):
        return self.vars.get("ARTEMIS_NETWORK", "off") == "off" or not self.flag("ARTEMIS_ALLOW_OUTBOUND")

    def deny(self, capability: str):
        """capability in push/payments/email/deploy/gpu_rental -> raise unless the env allows it."""
        key = {"gpu_rental": "ARTEMIS_GPU_RENTAL"}.get(capability, f"ARTEMIS_ALLOW_{capability.upper()}")
        if not self.flag(key):
            self.audit("denied", capability=capability)
            raise SandboxViolation(f"{capability} is disabled in the sandbox ({key}=0)")

    def limits(self, base=None):
        from .engines.code_exec import SandboxLimits
        v = self.vars
        mb = int(v.get("ARTEMIS_MAX_MEMORY_MB", 2048)) * 1024 * 1024
        t = float(v.get("ARTEMIS_TIMEOUT_SECONDS", 300))
        b = base or SandboxLimits()
        return SandboxLimits(timeout_s=min(b.timeout_s, t), cpu_s=min(b.cpu_s, int(t)), mem_bytes=min(b.mem_bytes, mb),
                             fsize_bytes=min(b.fsize_bytes, int(v.get("ARTEMIS_MAX_DISK_MB", 1024)) * 1024 * 1024),
                             nofile=min(b.nofile, int(v.get("ARTEMIS_MAX_OPEN_FILES", 256))),
                             max_output=min(b.max_output, int(v.get("ARTEMIS_MAX_OUTPUT_BYTES", 1 << 20))),
                             use_bwrap=True if self.network_off or self.untrusted else b.use_bwrap,
                             ro_paths=tuple(p for p in b.ro_paths if self.allowed_ro(p)),
                             nproc=int(v.get("ARTEMIS_MAX_PROCESSES", 32)))

    # -------------------------------------------------------------- paths
    def is_denied(self, path: str) -> bool:
        p = os.path.realpath(path)
        return any(p == d or p.startswith(d + os.sep) for d in self.deny_paths)

    def allowed_ro(self, path: str) -> bool:
        return not self.is_denied(path)

    def check_path(self, path: str, write: bool = False) -> str:
        p = os.path.realpath(path)
        if self.is_denied(p) or (write and not (p == self.root or p.startswith(self.root + os.sep))):
            self.audit("violation", path=p, write=write)
            raise SandboxViolation(f"{'write to' if write else 'access to'} {p} is not allowed in the sandbox")
        return p

    def require_inside(self, path: str) -> str:
        return self.check_path(path, write=True)

    def work_dir(self, *parts) -> str:
        p = os.path.join(self.vars.get("ARTEMIS_WORKDIR", os.path.join(self.root, "work")), *parts)
        os.makedirs(self.require_inside(p), exist_ok=True)
        return p

    def stage_libs(self, *modules) -> str:
        """Copy pure-python libs (e.g. sympy, mpmath) from user site-packages, which is denied, into
        <root>/libs so a bwrap child can get them read-only without seeing /home/box."""
        import importlib
        dest = os.path.join(self.root, "libs")
        os.makedirs(dest, exist_ok=True)
        for m in modules:
            src = os.path.dirname(importlib.import_module(m).__file__)
            if not os.path.exists(os.path.join(dest, m)):
                shutil.copytree(src, os.path.join(dest, m), ignore=shutil.ignore_patterns("__pycache__", "tests"))
        return dest

    # -------------------------------------------------------------- env / audit / lifecycle
    def child_env(self, workdir: str) -> dict:
        e = {k: self.vars[k] for k in CHILD_KEYS if k in self.vars}
        e.update({"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "LANG": "C.UTF-8",
                  "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"})
        return e

    def audit(self, event: str, **kw):
        p = self.vars.get("ARTEMIS_AUDIT_LOG", os.path.join(self.root, "logs", "audit.jsonl"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **kw}) + "\n")

    def setup(self):
        for k in ("ARTEMIS_WORKDIR", "ARTEMIS_CHECKPOINT_DIR", "ARTEMIS_DATA_DIR", "ARTEMIS_LOG_DIR", "ARTEMIS_TMPDIR", "HOME"):
            if k in self.vars:
                os.makedirs(self.require_inside(self.vars[k]), exist_ok=True)
        self.audit("session_start", source=self.source, label=self.vars.get("ARTEMIS_SANDBOX_LABEL"))

    def wipe(self, keep_audit_copy: Optional[str] = None):
        if keep_audit_copy:
            a = self.vars.get("ARTEMIS_AUDIT_LOG")
            if a and os.path.exists(a):
                shutil.copy(a, keep_audit_copy)
        shutil.rmtree(self.root, ignore_errors=True)

    @contextlib.contextmanager
    def session(self, keep_audit_copy: Optional[str] = None):
        self.setup()
        try:
            yield self
        finally:
            self.audit("session_end")
            if self.flag("ARTEMIS_WIPE_ON_EXIT"):
                self.wipe(keep_audit_copy)


def load(path: str = DEFAULT_PATH, **overrides) -> SandboxEnv:
    return SandboxEnv.load(path, **overrides)
