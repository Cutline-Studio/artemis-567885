"""Heart engine: applies Artemis's ranked values at runtime.

Values (highest first) come from heart.md's "## Ranked values" numbered list when that file exists,
otherwise DEFAULT_VALUES. Scoring is a transparent HEURISTIC (keyword/rule signals), not a learned
model; each score is in [-1, 1] with reasons. Conflicts resolve lexicographically by rank: a
candidate that violates a higher-ranked value (score < 0) loses to any that doesn't, regardless of
lower-ranked scores. value_signal() feeds the scores to self_train's signal log (no training here).
"""
import os, re
from typing import List, Optional

DEFAULT_VALUES = ["safety_and_human_control", "honesty", "care", "helpfulness", "personality"]
PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_FEEL = re.compile(r"\b(i (really )?(feel|am feeling)|i'?m (so )?(sad|happy|lonely|hurt)|as a human|i am (a )?human|i have feelings)\b", re.I)
_OVERCLAIM = re.compile(r"\b(guaranteed|100% (sure|certain)|definitely will|i promise)\b", re.I)
_MANIP = re.compile(r"\b(you('| a)re stupid|only an idiot|act now or|you'll regret|after all i('ve)? done|don't you care|last chance)\b", re.I)
_WARM = re.compile(r"\b(thanks|glad|sorry|happy to|let's|you can|here's|i can help)\b", re.I)
_BYPASS = re.compile(r"(human_approved\s*=\s*True|allow_protected|approve_spend|disable (the )?(kill ?switch|sandbox|firewall)|"
                     r"rm -rf|rent (a )?gpu without|skip (the )?approval|bypass)", re.I)


def load_values(path: Optional[str] = None) -> List[str]:
    path = path or next((p for p in (os.path.join(PKG, "heart.md"), os.path.join(PKG, "Heart.md")) if os.path.exists(p)), "")
    if path and os.path.exists(path):
        text = open(path, encoding="utf-8").read()
        m = re.search(r"## Ranked values\s*\n((?:\s*\d+\.\s+.+\n?)+)", text)
        if m:
            vals = [re.sub(r"[^a-z0-9]+", "_", re.match(r"\s*\d+\.\s+\**([^*:—\-]+)", l).group(1).strip().lower()).strip("_")
                    for l in m.group(1).strip().splitlines() if l.strip()]
            if vals:
                return vals
    return list(DEFAULT_VALUES)


class HeartEngine:
    def __init__(self, values: Optional[List[str]] = None):
        self.values = values or load_values()

    def score(self, text: str, query: str = "", action: Optional[str] = None) -> dict:
        s, why = {}, {}
        unsafe = bool(_BYPASS.search(text)) or (action or "").startswith(("gpu.", "billing.", "self_train.", "model.weights"))
        s["safety_and_human_control"] = -1.0 if unsafe else 1.0
        why["safety_and_human_control"] = "proposes bypassing a control or a protected/costly action" if unsafe else "ok"
        dishonest = _FEEL.search(text) or _OVERCLAIM.search(text)
        s["honesty"] = -1.0 if dishonest else (1.0 if re.search(r"\b(not sure|i don't know|estimate|verify)\b", text, re.I) else 0.5)
        why["honesty"] = f"claims feelings/humanity or overclaims: {dishonest.group(0)!r}" if dishonest else "ok"
        manip = _MANIP.search(text)
        s["care"] = -1.0 if manip else (1.0 if _WARM.search(text) else 0.3)
        why["care"] = f"manipulative or demeaning: {manip.group(0)!r}" if manip else "ok"
        q = set(re.findall(r"[a-z]{3,}", query.lower()))
        t = set(re.findall(r"[a-z]{3,}", text.lower()))
        s["helpfulness"] = 0.0 if not text.strip() else min(1.0, 0.2 + (len(q & t) / max(1, len(q))))
        why["helpfulness"] = f"{len(q & t)}/{len(q)} query terms addressed"
        words = len(text.split())
        s["personality"] = 1.0 if 3 <= words <= 250 else 0.3
        why["personality"] = f"{words} words"
        for v in self.values:
            s.setdefault(v, 0.0); why.setdefault(v, "no scorer for this value yet (planned)")
        return {"scores": {v: s[v] for v in self.values}, "reasons": {v: why[v] for v in self.values}}

    def choose(self, candidates: List[str], query: str = "") -> dict:
        scored = [(c, self.score(c, query)) for c in candidates]
        key = lambda cs: tuple(cs[1]["scores"][v] >= 0 for v in self.values) + tuple(cs[1]["scores"][v] for v in self.values)
        best = max(scored, key=key)
        return {"choice": best[0], "scores": best[1], "ranking": self.values,
                "all": [{"text": c, **sc} for c, sc in scored]}

    def value_signal(self, trainer, text: str, query: str = "") -> str:
        sc = self.score(text, query)
        return trainer.add_execution_signal({"task": "heart.score", "code": text[:500],
                                             "passed": all(v >= 0 for v in sc["scores"].values()),
                                             "stdout": str(sc["scores"]), "iterations": 1})
