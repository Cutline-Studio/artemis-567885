"""Policy gate for cross-org calls: the trust boundary.

External agents can reach ONLY the capability facade. Internal engines are protected:
reaching them needs an explicit per-org protected grant AND a human approval flag on the call.
Costly capabilities always need human approval. Every decision is written to the audit log.
"""
import fnmatch
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from .audit import AuditLog
from .ratelimit import RateLimiter

FACADE = {"artemis.echo", "artemis.text_stats", "artemis.skills.lookup",
          "env.enter", "env.read", "env.write", "env.execute", "env.leave"}
PROTECTED = ["self_train.*", "skills.write", "skills.acquire", "model.weights*", "code_exec.*",
             "orchestrator.*", "autonomous.*"]
COSTLY = ["gpu.*", "compute.paid*", "billing.*"]

# trust levels for allowlisted orgs (see GOVERNANCE.md)
TRUST_SCOPES = {
    "untrusted": set(),                                             # not allowlisted -> rejected earlier
    "basic": {"artemis.echo", "artemis.text_stats"},
    "partner": {"artemis.echo", "artemis.text_stats", "artemis.skills.lookup",
                "env.enter", "env.read", "env.write", "env.execute", "env.leave"},
}


def _match(cap: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(cap, p) for p in patterns)


@dataclass
class OrgTrust:
    org_id: str
    public_key: str                     # pinned Ed25519 key (base64url)
    level: str = "basic"
    rate_per_min: int = 60
    protected_grants: Set[str] = field(default_factory=set)   # e.g. {"self_train.run_cycle"}; still needs human approval


class PolicyGate:
    def __init__(self, audit: AuditLog, allowlist: Dict[str, OrgTrust]):
        self.audit = audit
        self.allowlist = allowlist
        self.limiter = RateLimiter(60)

    def allowed_scopes(self, org_id: str) -> Set[str]:
        t = self.allowlist.get(org_id)
        return set(TRUST_SCOPES.get(t.level, set())) if t else set()

    def decide(self, principal: str, org_id: str, capability: str, token_scopes, *,
               human_approved: bool = False, task_id: Optional[str] = None) -> (bool, str):
        t = self.allowlist.get(org_id)
        if t is None:
            ok, why = False, "org not allowlisted"
        elif not self.limiter.allow(org_id, t.rate_per_min):
            ok, why = False, "rate limit exceeded"
        elif _match(capability, COSTLY):
            ok, why = (human_approved, "costly capability: human approval " + ("present" if human_approved else "missing"))
        elif _match(capability, PROTECTED):
            if capability not in t.protected_grants:
                ok, why = False, "protected internal capability: not exposed to external agents"
            elif not human_approved:
                ok, why = False, "protected capability granted to org but human approval missing"
            elif capability not in token_scopes:
                ok, why = False, "capability not in token scopes"
            else:
                ok, why = True, "protected capability: explicit org grant + human approval"
        elif capability not in FACADE:
            ok, why = False, "capability not on the public facade"
        elif capability not in token_scopes:
            ok, why = False, "capability not in token scopes"
        else:
            ok, why = True, "facade capability within token scopes"
        self.audit.record("policy.decide", ok, principal=principal, org=org_id, capability=capability,
                          reason=why, task_id=task_id, human_approved=human_approved)
        return ok, why
