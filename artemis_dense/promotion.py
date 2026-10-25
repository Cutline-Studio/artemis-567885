"""Promotion gate: the ONE place where untrusted Artemis output leaves the sandbox.

Pipeline for anything Artemis generates or modifies (docs, code, skills, model checkpoints):
  1. it is produced / tested inside sandbox.env (SandboxEnv: /tmp/artemis-sandbox, no network, limits)
  2. PromotionGate.check(...) runs the custom Constitutional AI engine (custom.env + constitution.yaml)
     with runtime_env = the sandbox markers ONLY when the artifact really sits inside the sandbox
     root (otherwise the real process env, so the `sandbox_first` principle blocks it)
     The engine is LOOSE by default (custom.env ARTEMIS_CAI_MODE=warn): warnings are logged and never
     block. It flips to STRICT for the rest of the session on >3 warnings, any critical hit
     (production paths, network, payments, secrets) or a path touching the constitution /
     sandbox.env / custom.env; then verdict.outcome == "strict_block" and can_promote() is False until
     a human calls approve_resume(approver). We promote only if verdict.outcome != "strict_block"
     AND engine.can_promote().
     `paths` = the repo-relative paths the promotion would write (e.g. "artemis_dense/X.md") plus the
     sandbox staging path. Absolute repo paths are NOT passed: the repo lives under /workspace, which
     the constitution's production_access rule treats as critical, so every promotion would trip strict.
  3. allowed -> the caller promotes (copy into the repo / skills store / promoted checkpoint dir);
     blocked -> PromotionBlocked, nothing is promoted; both outcomes go to the sandbox audit log
     and the constitution log (matched text is never logged).
Callers: self_author (repo writes + corpus), code_exec.iterate (skill save), watch.practice (skill
save), self_train.promote (checkpoint promotion). There is no `self_improvement` module in this repo;
self_train is the self-improvement engine and its promotion step is SelfTrainer.promote().
"""
import os
from typing import Optional

from .engines.custom_constitutional import ConstitutionalEngine, Verdict
from .sandbox import SandboxEnv


class PromotionBlocked(PermissionError):
    def __init__(self, verdict: Verdict, what: str):
        self.verdict = verdict
        super().__init__(f"promotion of {what} blocked by the constitution (outcome={verdict.outcome}, strict "
                         f"mode on; a human must call approve_resume): " +
                         ", ".join(f"{v.principle} ({v.reason})" for v in verdict.violations))


class PromotionGate:
    def __init__(self, sandbox: Optional[SandboxEnv] = None, engine: Optional[ConstitutionalEngine] = None):
        self.sandbox = sandbox if sandbox is not None else SandboxEnv.load()
        self.engine = engine or ConstitutionalEngine()

    def runtime_env(self, artifact_path: Optional[str]) -> dict:
        if artifact_path:
            p = os.path.realpath(artifact_path)
            if p == self.sandbox.root or p.startswith(self.sandbox.root + os.sep):
                return dict(self.sandbox.vars)
        return dict(os.environ)

    def check(self, what: str, text: str = "", action: Optional[str] = None, approved: bool = False,
              artifact_path: Optional[str] = None, paths=None) -> Verdict:
        v = self.engine.check(text, action=action or "promote", approved=approved,
                              paths=list(paths or []) + ([artifact_path] if artifact_path else []),
                              runtime_env=self.runtime_env(artifact_path))
        self.sandbox.audit("promotion_check", what=what, action=action, outcome=v.outcome,
                           violations=[x.principle for x in v.violations])
        return v

    def promotable(self, v: Verdict) -> bool:
        return v.outcome != "strict_block" and self.engine.can_promote()

    def require(self, what: str, **kw) -> Verdict:
        v = self.check(what, **kw)
        if not self.promotable(v):
            raise PromotionBlocked(v, what)
        return v
