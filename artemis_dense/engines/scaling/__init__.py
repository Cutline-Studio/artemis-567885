"""Neural scaling laws engine.

Formulas (constants as published; re-check against the papers before relying on them, they are
copied from memory of the papers and flagged `verify` in ScalingLaws.md):
  Kaplan et al. 2020 (arXiv:2001.08361): L(N) = (N_c/N)^a_N, N_c=8.8e13, a_N=0.076 (non-embedding params);
                                          L(D) = (D_c/D)^a_D, D_c=5.4e13, a_D=0.095 (tokens). Loss in nats/token.
  Hoffmann et al. 2022 "Chinchilla" (arXiv:2203.15556), approach-3 fit:
      L(N, D) = E + A/N^alpha + B/D^beta, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28
  Training compute C ~= 6 N D FLOPs; compute-optimal rule of thumb D ~= 20 N.
These are fits to OTHER models/tokenizers/data: they do not transfer to Artemis's byte-level loss.
Artemis's own curve comes only from fit_power_law() on our own measured (N, loss) points.
"""
import math
from typing import Dict, List, Optional

KAPLAN = {"N_c": 8.8e13, "a_N": 0.076, "D_c": 5.4e13, "a_D": 0.095}
CHINCHILLA = {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28}


def kaplan_loss_N(N: float) -> float:
    return (KAPLAN["N_c"] / N) ** KAPLAN["a_N"]


def kaplan_loss_D(D: float) -> float:
    return (KAPLAN["D_c"] / D) ** KAPLAN["a_D"]


def chinchilla_loss(N: float, D: float, c: Dict = CHINCHILLA) -> float:
    return c["E"] + c["A"] / N ** c["alpha"] + c["B"] / D ** c["beta"]


def train_flops(N: float, D: float) -> float:
    return 6.0 * N * D


def compute_optimal(C: float, tokens_per_param: float = 20.0) -> Dict[str, float]:
    """Split a FLOP budget C with D = k N and C = 6 N D  ->  N = sqrt(C / (6k))."""
    N = math.sqrt(C / (6.0 * tokens_per_param))
    return {"N": N, "D": tokens_per_param * N, "C": C}


def chinchilla_optimal(C: float, c: Dict = CHINCHILLA) -> Dict[str, float]:
    """Minimise L(N, D) subject to 6ND = C (closed form from the fit)."""
    a, b = c["alpha"], c["beta"]
    G = ((a * c["A"]) / (b * c["B"])) ** (1.0 / (a + b))
    N = G * (C / 6.0) ** (b / (a + b))
    D = (C / 6.0) / N
    return {"N": N, "D": D, "C": C, "loss": chinchilla_loss(N, D, c)}


def gpu_cost(C: float, peak_flops: float, mfu: float, usd_per_hour: float) -> Dict[str, float]:
    """Hours and dollars to spend C FLOPs at a model-FLOPs-utilisation `mfu` of `peak_flops`.
    All three inputs are assumptions the caller must supply (see GPU.md / ScalingLaws.md)."""
    hours = C / (peak_flops * mfu) / 3600.0
    return {"hours": hours, "usd": hours * usd_per_hour}


def plan_for_budget(usd: float, peak_flops: float, mfu: float, usd_per_hour: float) -> Dict[str, float]:
    C = usd / usd_per_hour * 3600.0 * peak_flops * mfu
    return {**compute_optimal(C), "chinchilla": chinchilla_optimal(C), "usd": usd}


def fit_power_law(points: List[tuple], irreducible: Optional[float] = None) -> Dict[str, float]:
    """Fit L = a * N^-b (+ e) by least squares in log space over measured (N, loss) points."""
    import numpy as np
    N = np.array([p[0] for p in points], float)
    L = np.array([p[1] for p in points], float)
    e = 0.0 if irreducible is None else irreducible
    y = np.log(np.clip(L - e, 1e-9, None))
    X = np.stack([np.ones_like(N), np.log(N)], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = np.exp(X @ coef) + e
    ss = float(((L - pred) ** 2).sum()); st = float(((L - L.mean()) ** 2).sum()) or 1e-12
    return {"a": float(math.exp(coef[0])), "b": float(-coef[1]), "e": e, "r2": 1 - ss / st}


def predict(fit: Dict[str, float], N: float) -> float:
    return fit["a"] * N ** (-fit["b"]) + fit["e"]


def sweep(sizes: List[dict], data: bytes, steps: int = 300, seq: int = 64, batch: int = 8, seed: int = 0) -> List[dict]:
    """Tiny CPU sweep: train each config for `steps` on `data` (90/10 split), return measured eval loss.
    This is a smoke-scale measurement, not a real scaling study."""
    import torch
    from ..transformer import build, count_params
    torch.manual_seed(seed)
    n = int(len(data) * 0.9)
    tr = torch.tensor(list(data[:n])); ev = torch.tensor(list(data[n:]))
    out = []
    for s in sizes:
        m = build("tiny", seed=seed, **{**s, "context_len": seq, "tool_head": False, "value_head": False})
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(seed)
        for _ in range(steps):
            ix = torch.randint(0, len(tr) - seq - 1, (batch,), generator=g)
            x = torch.stack([tr[i:i + seq] for i in ix]); y = torch.stack([tr[i + 1:i + seq + 1] for i in ix])
            _, loss = m(x, y); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            g2 = torch.Generator().manual_seed(1)
            ix = torch.randint(0, len(ev) - seq - 1, (32,), generator=g2)
            x = torch.stack([ev[i:i + seq] for i in ix]); y = torch.stack([ev[i + 1:i + seq + 1] for i in ix])
            el = float(m(x, y)[1])
        out.append({**s, "params": count_params(m), "eval_loss": el, "steps": steps})
    return out
