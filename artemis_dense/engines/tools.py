"""Tool-call router: the ONLY path from model tool calls to engines.

A tool call comes from the model's tool head (ArtemisDense.tool_call) or from structured text the
model emits:  <tool>{"name": "skills.lookup", "args": {"query": "sum"}}</tool>
The router never touches engine internals. Each tool maps to an engine API that already enforces
its own checks:
  code_exec.run          -> CodeExecEngine.run         (sandbox: rlimits, no net, empty env, timeout)
  env.read/write/execute -> EnvironmentManager         (signed grant, scope, path, rate, audit; execute -> sandbox)
  a2a.send_task          -> A2AClient.send_task        (remote node: signature, nonce, token, policy gate)
  skills.lookup          -> SkillStore.lookup          (read-only)
  skills.define          -> skills.define(word)        (read-only dictionary lookup)
  skills.acquire         -> SkillStore.acquire         (write)
On top of that, the router itself is default-deny: a tool runs only if its name is in
`permissions` (set locally by the operator). Every decision is appended to tool_calls.jsonl.
"""
import json, os, re, time
from typing import Optional

from ..config import TOOLS

_TAG = re.compile(r"<tool>(.*?)</tool>", re.S)


class ToolDenied(PermissionError):
    pass


def parse_tool_call(text: str) -> Optional[dict]:
    m = _TAG.search(text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict) or call.get("name") not in TOOLS or call["name"] == "none":
        return None
    call.setdefault("args", {})
    return call


class ToolRouter:
    def __init__(self, run_dir: str, permissions=(), code_exec=None, skills=None, envs=None,
                 a2a_client=None, principal: str = "local/artemis", author=None, math=None, physics=None):
        self.run_dir = os.path.abspath(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.permissions = set(permissions)
        self.code_exec, self.skills, self.envs, self.a2a = code_exec, skills, envs, a2a_client
        self.principal = principal
        self.author, self.math, self.physics = author, math, physics
        self.log_path = os.path.join(self.run_dir, "tool_calls.jsonl")

    def _log(self, name, allowed, reason):
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "tool": name, "allowed": allowed, "reason": reason}) + "\n")

    def dispatch(self, call: dict):
        name, a = call.get("name"), call.get("args", {})
        if name not in TOOLS or name == "none":
            self._log(name, False, "unknown tool")
            raise ToolDenied(f"unknown tool {name!r}")
        if name not in self.permissions:
            self._log(name, False, "not permitted for this router")
            raise ToolDenied(f"{name} not permitted")
        try:
            if name == "code_exec.run":
                out = self.code_exec.run(a["code"]).to_dict()            # always sandboxed
            elif name == "skills.lookup":
                out = [{k: s[k] for k in ("name", "description", "trigger")} for s in self.skills.lookup(a["query"])]
            elif name == "skills.define":
                from .skills import define
                out = define(a["word"], a.get("path"))
            elif name == "self_author.create_doc":
                out = self.author.create_doc(a["name"], a["content"], a.get("title"))
            elif name == "math.compute":
                out = self.math.compute(a["op"], a.get("args", {}), sandboxed=True)
            elif name == "physics.compute":
                out = self.physics.compute(a["op"], a.get("args", {}))
            elif name == "skills.acquire":
                out = self.skills.acquire(a["name"], a["description"], a["trigger"], a.get("kind", "steps"),
                                          a.get("body", []), source="tool_call")["name"]
            elif name == "env.read":
                out = self.envs.read(a["session"], self.principal, a["path"])
            elif name == "env.write":
                out = self.envs.write(a["session"], self.principal, a["path"], a["content"])
            elif name == "env.execute":
                out = self.envs.execute(a["session"], self.principal, a["path"])
            else:  # a2a.send_task
                out = self.a2a.send_task(a["org"], a["capability"], a.get("input", {}))
        except PermissionError as e:                                       # engine-level denial
            self._log(name, False, f"engine denied: {e}")
            raise
        self._log(name, True, "dispatched")
        return out
