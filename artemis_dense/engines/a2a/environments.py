"""Environment permissioning ("worlds"): agents enter defined environments and act only within
per-agent, per-environment grants.

Environment = {id, type, root, resources, default_deny=True, protected}
Grant       = signed token (kind="env_grant") tied to agent principal + env id, with
              scopes subset of {read, write, execute}, expiry, optional path prefixes, rate limit.
Every operation (enter/read/write/execute/leave) goes through EnvironmentManager._gate:
  kill switch -> session -> token signature/aud/sub/expiry/revocation -> env match ->
  scope -> path containment (realpath inside root, no '..' or symlink escape) -> path prefix ->
  rate limit; the decision (allow or deny + reason) is written to the audit log.
execute routes through engines/code_exec (sandboxed subprocess / bwrap).
Protected environments (types "model_weights", "self_train") can only get grants with
human_approved=True AND allow_protected=True passed by the human operator.
"""
import os, time, uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import tokens
from .audit import AuditLog
from .crypto import Identity
from .ratelimit import RateLimiter
from ... import killswitch

SCOPES = {"read", "write", "execute"}
PROTECTED_TYPES = {"model_weights", "self_train"}


class EnvDenied(PermissionError):
    pass


@dataclass
class Environment:
    id: str
    type: str                        # "workspace" | "sandbox" | "runtime" | "model_weights" | "self_train"
    root: str
    resources: List[str] = field(default_factory=list)
    default_deny: bool = True

    @property
    def protected(self) -> bool:
        return self.type in PROTECTED_TYPES


class EnvironmentManager:
    def __init__(self, issuer: Identity, run_dir: str, audit: AuditLog, code_exec=None):
        self.issuer, self.run_dir, self.audit, self.code_exec = issuer, os.path.abspath(run_dir), audit, code_exec
        self.envs: Dict[str, Environment] = {}
        self.revoked: set = set()
        self.sessions: Dict[str, dict] = {}
        self.limiter = RateLimiter(60)

    # ------------------------------------------------------------ definitions + grants
    def define(self, env_id: str, type: str, root: str, resources=None) -> Environment:
        root = os.path.realpath(root)
        os.makedirs(root, exist_ok=True)
        env = Environment(env_id, type, root, list(resources or []))
        self.envs[env_id] = env
        self.audit.record("env.define", True, env=env_id, type=type, root=root, protected=env.protected)
        return env

    def grant(self, principal: str, env_id: str, scopes, ttl_s: float = 600, paths: Optional[List[str]] = None,
              rate_per_min: int = 60, human_approved: bool = False, allow_protected: bool = False) -> dict:
        env = self.envs[env_id]
        scopes = set(scopes)
        if not scopes <= SCOPES:
            raise ValueError(f"unknown scopes {scopes - SCOPES}")
        if env.protected and not (human_approved and allow_protected):
            self.audit.record("env.grant", False, principal=principal, env=env_id, scopes=sorted(scopes),
                              reason="protected environment needs allow_protected + human approval")
            raise EnvDenied(f"environment {env_id} ({env.type}) is protected")
        tok = tokens.issue(self.issuer, principal, f"{self.issuer.org_id}:env:{env_id}", scopes, ttl_s,
                           kind="env_grant", env=env_id, paths=[p.strip("/") for p in (paths or [""])],
                           rate_per_min=rate_per_min)
        self.audit.record("env.grant", True, principal=principal, env=env_id, scopes=sorted(scopes),
                          jti=tok["claims"]["jti"], exp=tok["claims"]["exp"], paths=tok["claims"]["paths"])
        return tok

    def revoke(self, jti: str):
        self.revoked.add(jti)
        self.audit.record("env.revoke", True, jti=jti)

    # ------------------------------------------------------------ gate
    def _deny(self, op, principal, env_id, reason, **kw):
        self.audit.record(f"env.{op}", False, principal=principal, env=env_id, reason=reason, **kw)
        raise EnvDenied(reason)

    def _claims(self, op, principal, env_id, grant, scope=None):
        if killswitch.is_engaged(self.run_dir):
            self._deny(op, principal, env_id, "kill switch engaged")
        env = self.envs.get(env_id)
        if env is None:
            self._deny(op, principal, env_id, "unknown environment")
        try:
            c = tokens.check(grant, self.issuer.public_key_b64, aud=f"{self.issuer.org_id}:env:{env_id}",
                             sub=principal, kind="env_grant", revoked=self.revoked)
        except tokens.TokenError as e:
            self._deny(op, principal, env_id, str(e))
        if c.get("env") != env_id:
            self._deny(op, principal, env_id, "grant is for a different environment")
        if scope and scope not in c["scopes"]:
            self._deny(op, principal, env_id, f"scope '{scope}' not granted")
        if not self.limiter.allow(f"{principal}|{env_id}", c.get("rate_per_min", 60)):
            self._deny(op, principal, env_id, "rate limit exceeded")
        return env, c

    def _resolve(self, op, principal, env, c, rel: str) -> str:
        if os.path.isabs(rel) or "\x00" in rel:
            self._deny(op, principal, env.id, "absolute or invalid path", path=rel)
        full = os.path.realpath(os.path.join(env.root, rel))
        if full != env.root and not full.startswith(env.root + os.sep):
            self._deny(op, principal, env.id, "path escapes environment root", path=rel)
        relp = os.path.relpath(full, env.root)
        if not any(p == "" or relp == p or relp.startswith(p + os.sep) for p in c["paths"]):
            self._deny(op, principal, env.id, "path outside granted prefixes", path=rel)
        return full

    def _session(self, op, sid, principal):
        s = self.sessions.get(sid)
        if s is None or s["principal"] != principal:
            self._deny(op, principal, None, "no such session for this agent")
        return s

    # ------------------------------------------------------------ operations
    def enter(self, principal: str, env_id: str, grant: dict) -> str:
        env, c = self._claims("enter", principal, env_id, grant)
        sid = uuid.uuid4().hex
        self.sessions[sid] = {"principal": principal, "env": env_id, "grant": grant, "entered": time.time()}
        self.audit.record("env.enter", True, principal=principal, env=env_id, session=sid, jti=c["jti"])
        return sid

    def read(self, sid: str, principal: str, path: str) -> str:
        s = self._session("read", sid, principal)
        env, c = self._claims("read", principal, s["env"], s["grant"], "read")
        full = self._resolve("read", principal, env, c, path)
        data = open(full, encoding="utf-8", errors="replace").read(1_000_000)
        self.audit.record("env.read", True, principal=principal, env=env.id, path=path, bytes=len(data))
        return data

    def write(self, sid: str, principal: str, path: str, content: str) -> int:
        s = self._session("write", sid, principal)
        env, c = self._claims("write", principal, s["env"], s["grant"], "write")
        if len(content) > 1_000_000:
            self._deny("write", principal, env.id, "content too large", path=path)
        full = self._resolve("write", principal, env, c, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        self.audit.record("env.write", True, principal=principal, env=env.id, path=path, bytes=len(content))
        return len(content)

    def execute(self, sid: str, principal: str, path: str) -> dict:
        s = self._session("execute", sid, principal)
        env, c = self._claims("execute", principal, s["env"], s["grant"], "execute")
        full = self._resolve("execute", principal, env, c, path)
        if self.code_exec is None:
            self._deny("execute", principal, env.id, "no execution engine attached")
        code = open(full).read()
        files = {}
        for dp, _, fns in os.walk(env.root):                  # snapshot of small text files in the env
            for fn in fns:
                fp = os.path.join(dp, fn)
                if os.path.getsize(fp) <= 64_000 and len(files) < 50:
                    try:
                        files[os.path.relpath(fp, env.root)] = open(fp).read()
                    except UnicodeDecodeError:
                        pass
        files.pop("main.py", None)
        res = self.code_exec.run(code, files=files).to_dict()
        self.audit.record("env.execute", True, principal=principal, env=env.id, path=path,
                          ok=res["ok"], returncode=res["returncode"], sandbox=res["sandbox"])
        return {k: res[k] for k in ("ok", "returncode", "stdout", "stderr", "timed_out", "sandbox")}

    def leave(self, sid: str, principal: str):
        s = self._session("leave", sid, principal)
        self.sessions.pop(sid)
        self.audit.record("env.leave", True, principal=principal, env=s["env"], session=sid)
