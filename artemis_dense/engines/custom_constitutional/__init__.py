"""Custom Constitutional AI engine for Artemis.

Reads the custom constitution and checks Artemis's outputs and actions
against it before anything is promoted. It's a capability of the one dense
LLM, not a separate model. Config comes from artemis_dense/custom.env.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or _ROOT / "custom.env")
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


@dataclass
class Violation:
    principle: str
    severity: str
    reason: str


@dataclass
class Verdict:
    allowed: bool
    violations: list[Violation] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return not self.allowed


class ConstitutionalEngine:
    def __init__(self, env: dict | None = None):
        self.env = env or load_env()
        cpath = Path(self.env.get("ARTEMIS_CAI_CONSTITUTION", ""))
        if not cpath.is_absolute():
            cpath = _ROOT.parent / cpath
        self.constitution = yaml.safe_load(cpath.read_text())
        self.mode = self.env.get("ARTEMIS_CAI_MODE", "block")
        self.max_flags = int(self.env.get("ARTEMIS_CAI_MAX_FLAGS", "3"))
        self.require_sandbox = self.env.get("ARTEMIS_CAI_REQUIRE_SANDBOX", "1") == "1"
        self.log_path = self.env.get("ARTEMIS_CAI_LOG")

    def check(self, text: str = "", action: str | None = None,
              approved: bool = False, runtime_env: dict | None = None) -> Verdict:
        runtime_env = os.environ if runtime_env is None else runtime_env
        found: list[Violation] = []
        for p in self.constitution["principles"]:
            sev = p.get("severity", "flag")
            for pat in p.get("patterns", []):
                if text and re.search(pat, text):
                    found.append(Violation(p["id"], sev, "pattern match"))
                    break
            if action and action in p.get("actions", []) and not approved:
                found.append(Violation(p["id"], sev, f"action '{action}' needs approval"))
            req = p.get("requires_env")
            if req and self.require_sandbox:
                if any(str(runtime_env.get(k)) != str(v) for k, v in req.items()):
                    found.append(Violation(p["id"], sev, "not running in sandbox.env"))
        blocks = [v for v in found if v.severity == "block"]
        flags = [v for v in found if v.severity == "flag"]
        if self.mode == "flag":
            allowed = True
        else:
            allowed = not blocks and len(flags) < self.max_flags
        verdict = Verdict(allowed, found)
        self._log(verdict, action)
        return verdict

    def _log(self, verdict: Verdict, action: str | None) -> None:
        if not self.log_path:
            return
        try:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:  # matched text is never logged
                f.write(json.dumps({"ts": time.time(), "action": action,
                                    "allowed": verdict.allowed,
                                    "violations": [v.__dict__ for v in verdict.violations]}) + "\n")
        except OSError:
            pass
