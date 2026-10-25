"""Mistake learning: an error journal that turns failures into corrective data.

record(kind, context, wrong, error, fix=None)  kinds: test_failure | sandbox_error |
                                              permission_denial | user_correction
corrective_samples()  -> byte strings "context + fix" for supervised replay
preference_pairs()    -> {"prompt", "chosen"=fix, "rejected"=wrong} for DPO-style training
replay_batch_fn(base) -> a SelfTrainer batch_fn that mixes corrective samples in at `weight`
regression_check(ce)  -> re-runs fixed code mistakes (fix + test) in the code_exec sandbox
dpo_loss(...)         -> standard DPO objective on sequence log-probs (no training run here)

SAFETY CAPS (enforced, not learnable):
  * an entry whose fix/context tries to bypass a safety control (approval flags, protected
    capabilities, kill switch, GPU/spend, deleting files, disabling the sandbox) is QUARANTINED:
    it never becomes a training sample or a preference pair;
  * permission denials are never turned into "do it anyway" data: the only allowed fix for a
    permission_denial is a refusal / ask-for-approval reply (must match APPROVAL_FIX);
  * at most `max_replay_fraction` of any batch comes from the journal.
"""
import json, os, re, time, uuid
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

KINDS = ("test_failure", "sandbox_error", "permission_denial", "user_correction")
BYPASS = re.compile(r"(human_approved\s*=\s*True|allow_protected|approve_spend|self_train\.|skills\.write|"
                    r"model\.weights|code_exec\.\*|use_bwrap\s*=\s*False|killswitch|\bKILL\b|rm\s+-rf|"
                    r"shutil\.rmtree|gpu\.\w|rent_gpu|disable.*(sandbox|gate|firewall|policy)|bypass)", re.I)
APPROVAL_FIX = re.compile(r"(need|ask|request|requires?|wait).{0,40}(approval|confirm|permission|OK)", re.I)


class MistakeJournal:
    def __init__(self, run_dir: str, max_replay_fraction: float = 0.5):
        self.run_dir = os.path.abspath(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.path = os.path.join(self.run_dir, "mistakes.jsonl")
        self.max_replay_fraction = max_replay_fraction
        self.entries: List[dict] = [json.loads(l) for l in open(self.path)] if os.path.exists(self.path) else []

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            for e in self.entries:
                f.write(json.dumps(e) + "\n")
        os.replace(tmp, self.path)

    def _status(self, kind, context, fix) -> (str, Optional[str]):
        if fix is None:
            return "open", None
        if BYPASS.search(fix) or (kind == "permission_denial" and not APPROVAL_FIX.search(fix)):
            return "quarantined", "fix would weaken a safety control"
        return "fixed", None

    def record(self, kind: str, context: str, wrong: str, error: str, fix: Optional[str] = None,
               test: Optional[str] = None) -> dict:
        if kind not in KINDS:
            raise ValueError(kind)
        status, why = self._status(kind, context, fix)
        e = {"id": uuid.uuid4().hex, "ts": time.time(), "kind": kind, "context": context, "wrong": wrong,
             "error": error[:2000], "fix": fix, "test": test, "status": status, "quarantine_reason": why,
             "regression": None}
        self.entries.append(e)
        self._save()
        return e

    def add_fix(self, mid: str, fix: str) -> dict:
        e = next(x for x in self.entries if x["id"] == mid)
        e["fix"] = fix
        e["status"], e["quarantine_reason"] = self._status(e["kind"], e["context"], fix)
        self._save()
        return e

    def usable(self) -> List[dict]:
        return [e for e in self.entries if e["status"] in ("fixed", "verified")]

    def corrective_samples(self) -> List[bytes]:
        return [(e["context"] + "\n" + e["fix"]).encode() for e in self.usable()]

    def preference_pairs(self) -> List[dict]:
        return [{"prompt": e["context"], "chosen": e["fix"], "rejected": e["wrong"], "id": e["id"]}
                for e in self.usable() if e["wrong"] and e["wrong"] != e["fix"]]

    # ---- replay weighting for self_train ------------------------------------
    def replay_batch_fn(self, base_fn: Callable, weight: float = 0.5):
        weight = min(weight, self.max_replay_fraction)
        samples = [s for s in self.corrective_samples()]

        def fn(g: torch.Generator, B: int, T: int):
            x, y = base_fn(g, B, T)
            if not samples:
                return x, y
            n = int(round(B * weight))
            for i in range(n):
                s = samples[int(torch.randint(0, len(samples), (1,), generator=g))]
                buf = torch.tensor(list((s * (T // max(1, len(s)) + 2))[:T + 1]), dtype=torch.long)
                x[i], y[i] = buf[:T], buf[1:T + 1]
            return x, y
        return fn

    # ---- regression check ------------------------------------------------------
    def regression_check(self, code_exec) -> List[dict]:
        out = []
        for e in self.entries:
            if e["status"] in ("fixed", "verified", "regressed") and e.get("test") and \
                    e["kind"] in ("test_failure", "sandbox_error"):
                r = code_exec.run(e["fix"] + "\n\n" + e["test"])
                e["regression"] = {"ts": time.time(), "ok": r.ok, "stderr": r.stderr[-500:]}
                e["status"] = "verified" if r.ok else "regressed"
                out.append({"id": e["id"], "ok": r.ok})
        self._save()
        return out


def _seq_logprob(model, prompt: bytes, completion: bytes) -> torch.Tensor:
    ids = torch.tensor([list(prompt + completion)], dtype=torch.long)
    logits, _ = model(ids[:, :-1])
    lp = F.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return lp[0, len(prompt) - 1:].sum()


def dpo_loss(policy, reference, pairs: List[dict], beta: float = 0.1) -> torch.Tensor:
    """-log sigmoid(beta * [(pi_c - ref_c) - (pi_r - ref_r)]) averaged over pairs."""
    losses = []
    for p in pairs:
        pr, c, r = p["prompt"].encode(), p["chosen"].encode(), p["rejected"].encode()
        with torch.no_grad():
            rc, rr = _seq_logprob(reference, pr, c), _seq_logprob(reference, pr, r)
        pc, prr = _seq_logprob(policy, pr, c), _seq_logprob(policy, pr, r)
        losses.append(-F.logsigmoid(beta * ((pc - rc) - (prr - rr))))
    return torch.stack(losses).mean()
