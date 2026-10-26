"""Method 20: self-play best-of-N with verification + heart self-score.

For engine-generated problems with known answers: sample N completions, keep a completion only
if (a) the engine verifies its answer and (b) the heart engine scores every value >= 0; among
those keep the one the model itself rates most likely. Kept samples become training text (a
STaR-style loop). An untrained model will usually keep nothing - that is reported, not hidden.
"""
from typing import List

import torch

from ..engines.heart import HeartEngine


def _norm(s: str) -> str:
    return " ".join(s.replace(" ", "").split(","))


def best_of_n(model, codec, problems: List[dict], n: int = 4, max_new: int = 24, temperature: float = 0.8,
              seed: int = 0, heart: HeartEngine = None) -> dict:
    heart = heart or HeartEngine()
    kept, tried, correct = [], 0, 0
    for k, p in enumerate(problems):
        prompt = f'<problem verified="true">\nQ: {p["q"]}\nA: '
        ids = codec.encode(prompt)
        want = ", ".join(p["a"]) if isinstance(p["a"], list) else str(p["a"])
        cands = []
        room = getattr(getattr(model, "cfg", None), "context_len", 10 ** 9) - len(ids) - 1
        if room < 1:
            continue
        for j in range(n):
            g = torch.Generator().manual_seed(seed * 1000 + k * n + j)
            model.eval()
            out = model.generate(torch.tensor([ids]), min(max_new, room), temperature=temperature, generator=g)[0, len(ids):]
            ans = codec.decode(out.tolist()).split("\n")[0].strip()
            tried += 1
            ok = _norm(ans) == _norm(want)
            sc = heart.score(ans, p["q"])["scores"]
            if ok and all(v >= 0 for v in sc.values()):
                text_ids = torch.tensor([(ids + codec.encode(ans + "\n</problem>\n"))[:room + len(ids)]])
                lp = float(model.sequence_logprob(text_ids)[0])
                cands.append((lp, ans))
            correct += ok
        if cands:
            lp, ans = max(cands)
            kept.append(prompt + ans + "\n</problem>\n")
    return {"kept": kept, "tried": tried, "verified_correct": correct, "problems": len(problems)}
