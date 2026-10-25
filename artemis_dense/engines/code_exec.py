"""Code-execution engine: write -> run -> observe -> revise, inside a sandbox.

Sandbox layers (see artemis_dense/README.md "What is enforced vs best-effort"):
  ENFORCED (always)
    * separate child process, own session; hard wall-clock timeout -> SIGKILL of the process group
    * resource.setrlimit in the child: CPU seconds, address space, max file size, open files
    * minimal env built from scratch: only PATH/HOME/LANG/TMPDIR + PYTHON flags. Nothing from
      the parent env is inherited (no VAST_API_KEY, tokens, keys, proxies)
    * python -I -S (isolated mode: ignores PYTHON* env vars and user site-packages)
    * cwd is a fresh temp dir under <run_dir>/exec/; stdout/stderr captured to files, capped
  ENFORCED only when bubblewrap (bwrap) is available (it is on this box, /usr/bin/bwrap):
    * --unshare-all: new net namespace (no interfaces but loopback-down -> no network),
      pid/ipc/uts/user/cgroup namespaces
    * filesystem view = read-only /usr (+ lib symlinks), private /tmp, /proc, /dev, and the
      task's temp dir bound read-write at /work. The rest of the box (home, /workspace,
      ~/.ssh, run dirs, checkpoints) is not visible.
  BEST-EFFORT (fallback when bwrap is missing or disabled)
    * socket.socket / socket.create_connection / getaddrinfo are replaced with functions that raise,
      before user code runs. Determined code could bypass this via _socket / ctypes.
    * filesystem confinement is only "cwd is the temp dir"; absolute paths remain readable
      subject to normal Unix permissions.
"""
import json, os, shutil, signal, subprocess, sys, tempfile, time, uuid
from dataclasses import dataclass, asdict, field
from typing import Callable, List, Optional

try:
    import resource
except ImportError:          # non-POSIX
    resource = None

_GUARD = r'''
import socket as _s, sys as _sys, runpy as _r
def _blocked(*a, **k):
    raise PermissionError("network disabled in Artemis code_exec sandbox")
class _NoSocket(_s.socket):
    def __init__(self, *a, **k):
        _blocked()
_s.socket = _NoSocket
_s.create_connection = _blocked
_s.getaddrinfo = _blocked
_s.socketpair = _blocked
_sys.argv = ["main.py"]
_r.run_path("main.py", run_name="__main__")
'''


@dataclass
class SandboxLimits:
    timeout_s: float = 5.0
    cpu_s: int = 5
    mem_bytes: int = 512 * 1024 * 1024
    fsize_bytes: int = 4 * 1024 * 1024
    nofile: int = 64
    max_output: int = 16 * 1024
    use_bwrap: Optional[bool] = None      # None = auto (use if installed)
    ro_paths: tuple = ()                  # extra host dirs bound READ-ONLY (e.g. a site-packages dir for sympy)
    nproc: Optional[int] = None           # RLIMIT_NPROC (set from sandbox.env ARTEMIS_MAX_PROCESSES)


@dataclass
class ExecResult:
    ok: bool
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float
    sandbox: str
    workdir: str
    truncated: bool = False

    def to_dict(self):
        return asdict(self)


def _owned(pid: str) -> bool:
    try:
        return os.stat(f"/proc/{pid}").st_uid == os.getuid()
    except OSError:
        return False


def _threads(pid: str) -> int:
    try:
        return len(os.listdir(f"/proc/{pid}/task"))
    except OSError:
        return 0


def _minimal_env(home: str) -> dict:
    return {"PATH": "/usr/bin:/bin", "HOME": home, "TMPDIR": home, "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}


class CodeExecEngine:
    def __init__(self, run_dir: str, limits: Optional[SandboxLimits] = None, sandbox=None):
        """sandbox: an artemis_dense.sandbox.SandboxEnv (sandbox.env). When given, run_dir must be inside
        the sandbox root, limits are clamped to the env's limits, denied prod paths are never bound,
        bwrap is required (no best-effort fallback), and the child env carries the sandbox markers."""
        self.sandbox = sandbox
        if sandbox is not None:
            sandbox.require_inside(run_dir)
            limits = sandbox.limits(limits)
            sandbox.audit("code_exec_init", run_dir=os.path.abspath(run_dir))
        self.run_dir = os.path.abspath(run_dir)
        self.exec_root = os.path.join(self.run_dir, "exec")
        os.makedirs(self.exec_root, exist_ok=True)
        self.limits = limits or SandboxLimits()
        self.bwrap = shutil.which("bwrap")
        if self.limits.use_bwrap is False:
            self.bwrap = None
        elif self.bwrap and not self._bwrap_works():
            self.bwrap = None
        if self.limits.use_bwrap is True and not self.bwrap:
            raise RuntimeError("use_bwrap=True but bubblewrap is not usable on this host")

    # ---------------------------------------------------------------- sandbox
    def _bwrap_works(self) -> bool:
        try:
            r = subprocess.run([self.bwrap, "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr",
                                "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
                                "--symlink", "usr/lib64", "/lib64", "/usr/bin/true"],
                               capture_output=True, timeout=5, env={"PATH": "/usr/bin"})
            return r.returncode == 0
        except Exception:
            return False

    def _env(self, home):
        return self.sandbox.child_env(home) if self.sandbox is not None else _minimal_env(home)

    def _preexec(self):
        os.setsid()
        if resource is None:
            return
        L = self.limits
        for lim, val in ((resource.RLIMIT_CPU, L.cpu_s), (resource.RLIMIT_AS, L.mem_bytes),
                         (resource.RLIMIT_FSIZE, L.fsize_bytes), (resource.RLIMIT_NOFILE, L.nofile),
                         (resource.RLIMIT_CORE, 0)) + (((resource.RLIMIT_NPROC, L.nproc + self._uid_procs),) if L.nproc else ()):
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):
                pass

    def _argv(self, workdir: str) -> List[str]:
        py = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
        inner = [py, "-I", "-S", "_guard.py"]
        if not self.bwrap:
            return inner
        return [self.bwrap, "--unshare-all", "--die-with-parent", "--new-session",
                "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
                "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                *sum((["--ro-bind", p, p] for p in self.limits.ro_paths
                      if os.path.isdir(p) and (self.sandbox is None or not self.sandbox.is_denied(p))), []),
                "--bind", workdir, "/work", "--chdir", "/work", "--clearenv",
                *sum((["--setenv", k, v] for k, v in self._env("/work").items()), []),
                *inner]

    def run(self, code: str, files: Optional[dict] = None) -> ExecResult:
        """Run `code` as main.py in a fresh temp dir under <run_dir>/exec/."""
        if self.sandbox is not None:
            self.sandbox.audit("code_exec_run", bytes=len(code))
        # RLIMIT_NPROC is per real uid, so the cap is "processes this user already has + nproc"
        # (the kernel counts threads, so count /proc/<pid>/task entries)
        self._uid_procs = sum(_threads(d) for d in os.listdir("/proc") if d.isdigit() and _owned(d)) if self.limits.nproc else 0
        workdir = tempfile.mkdtemp(prefix="task-", dir=self.exec_root)
        with open(os.path.join(workdir, "main.py"), "w") as f:
            f.write(code)
        with open(os.path.join(workdir, "_guard.py"), "w") as f:
            f.write(_GUARD)
        for name, content in (files or {}).items():
            dest = os.path.realpath(os.path.join(workdir, name))
            if not dest.startswith(os.path.realpath(workdir) + os.sep):
                raise ValueError(f"file {name!r} escapes the task dir")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            open(dest, "w").write(content)
        outdir = tempfile.mkdtemp(prefix="out-", dir=self.exec_root)   # outside the child's view
        out_p, err_p = os.path.join(outdir, "stdout"), os.path.join(outdir, "stderr")
        t0, timed_out, rc = time.time(), False, None
        with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
            proc = subprocess.Popen(self._argv(workdir), cwd=workdir, stdin=subprocess.DEVNULL,
                                    stdout=fo, stderr=fe, env=self._env(workdir),
                                    preexec_fn=self._preexec, close_fds=True)
            try:
                rc = proc.wait(timeout=self.limits.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                rc = proc.returncode
        cap = self.limits.max_output
        out_b, err_b = open(out_p, "rb").read(cap + 1), open(err_p, "rb").read(cap + 1)
        truncated = len(out_b) > cap or len(err_b) > cap
        shutil.rmtree(outdir, ignore_errors=True)        # our own capture dir under run_dir only
        return ExecResult(ok=(rc == 0 and not timed_out), returncode=rc,
                          stdout=out_b[:cap].decode("utf-8", "replace"), stderr=err_b[:cap].decode("utf-8", "replace"),
                          timed_out=timed_out, seconds=round(time.time() - t0, 3),
                          sandbox="bwrap+rlimits" if self.bwrap else "subprocess+rlimits",
                          workdir=workdir, truncated=truncated)

    # ---------------------------------------------------------------- iterate loop
    def iterate(self, task: str, propose: Callable[[str, list], str], tests: str = "",
                max_iters: int = 4, skills=None, trainer=None, skill_name: Optional[str] = None,
                gate=None) -> dict:
        """write -> run -> observe -> revise. `propose(task, history) -> code` is the model
        (or a scripted stand-in). `tests` is appended to the code; pass == exit code 0.
        On success: optionally save as a skill (kind="code"); every outcome can be recorded
        as an execution signal for the self-training engine (stub, no training)."""
        history = []
        for i in range(1, max_iters + 1):
            code = propose(task, history)
            res = self.run(code + ("\n\n" + tests if tests else ""))
            history.append({"iteration": i, "code": code, **res.to_dict()})
            if res.ok:
                break
        last = history[-1]
        out = {"task": task, "passed": last["ok"], "iterations": len(history), "code": last["code"],
               "stdout": last["stdout"], "stderr": last["stderr"], "history": history}
        if out["passed"] and skills is not None and skill_name:
            from ..promotion import PromotionGate
            g = gate or PromotionGate(sandbox=self.sandbox)
            v = g.check(f"code skill {skill_name}", text=last["code"], action="code_exec.promote_skill",
                        artifact_path=last["workdir"])
            out["promotion"] = {"allowed": g.promotable(v), "outcome": v.outcome,
                                "violations": [x.principle for x in v.violations]}
        if out["passed"] and skills is not None and skill_name and out["promotion"]["allowed"]:
            skills.acquire(skill_name, f"Solved code task: {task}", task, "code", last["code"],
                           source="code_exec", overwrite=True)
            out["skill_saved"] = skill_name
        if trainer is not None:
            out["signal_path"] = trainer.add_execution_signal(out)
        with open(os.path.join(self.run_dir, "code_exec_log.jsonl"), "a") as f:
            f.write(json.dumps({k: v for k, v in out.items() if k != "history"}) + "\n")
        return out
