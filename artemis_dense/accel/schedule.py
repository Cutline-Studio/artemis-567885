"""Methods 17 and 18: scaling-law sizing and LR warmup + cosine with a small LR range finder."""
import copy, math
from typing import Callable, Dict, List

import torch

from ..config import DenseConfig, SCALES
from ..engines import scaling


# ---- 17 scaling-law sizing ------------------------------------------------------------------------
def non_embedding_params(cfg: DenseConfig) -> int:
    d, f, hd = cfg.d_model, cfg.d_ff, cfg.head_dim
    return cfg.n_layers * (2 * d * d + 2 * d * cfg.n_kv_heads * hd + 3 * d * f + 2 * d)


def choose_size(tokens_per_s: Dict[str, float], seconds: float, overrides: dict = None) -> dict:
    """For each candidate size, D = measured tokens/s * budget seconds, N = non-embedding params;
    pick the one with the lowest Chinchilla-fit loss L(N, D) from engines/scaling. The fit's
    constants come from large-scale runs, so at this scale it is a heuristic (labelled as such)."""
    rows = []
    for name, tps in tokens_per_s.items():
        cfg = DenseConfig.preset(name, **(overrides or {}))
        N, D = non_embedding_params(cfg), tps * seconds
        rows.append({"size": name, "N": N, "D_est": D, "tokens_per_param": D / N,
                     "chinchilla_loss_est": scaling.chinchilla_loss(N, D),
                     "compute_optimal_N_for_C": scaling.compute_optimal(6 * N * D)["N"]})
    best = min(rows, key=lambda r: r["chinchilla_loss_est"])
    return {"choice": best["size"], "candidates": rows, "note": "estimate: Chinchilla fit extrapolated to tiny scale"}


# ---- 18 LR warmup + cosine ------------------------------------------------------------------------
def lr_at(frac: float, peak: float, warmup_frac: float, floor: float = 0.1) -> float:
    """frac = progress in [0, 1] (time- or step-based). Linear warmup, cosine to floor * peak."""
    if warmup_frac > 0 and frac < warmup_frac:
        return peak * max(frac, 1e-3) / warmup_frac
    p = (frac - warmup_frac) / max(1e-9, 1 - warmup_frac)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, p)))))


def lr_range_test(model, batch_fn: Callable, lr_min: float = 1e-5, lr_max: float = 3e-2, steps: int = 40,
                  loss_fn: Callable = None, beta: float = 0.8) -> dict:
    """Smith's LR range test on a throwaway copy: exponential LR ramp, smoothed loss; suggest
    lr at the steepest descent (clamped to <= min-loss lr / 3)."""
    m = copy.deepcopy(model); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr_min, betas=(0.9, 0.95), weight_decay=0.1)
    mult = (lr_max / lr_min) ** (1 / max(1, steps - 1))
    lrs, losses, avg, best = [], [], 0.0, float("inf")
    lr = lr_min
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch_fn()
        loss = loss_fn(m, x, y) if loss_fn else m(x, y)[1]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        avg = beta * avg + (1 - beta) * loss.item()
        sm = avg / (1 - beta ** (i + 1))
        lrs.append(lr); losses.append(sm)
        best = min(best, sm)
        if i > 5 and (not math.isfinite(sm) or sm > 4 * best):
            break
        lr *= mult
    i_min = min(range(len(losses)), key=losses.__getitem__)
    grads = [(losses[i + 1] - losses[i]) for i in range(len(losses) - 1)]
    i_steep = min(range(len(grads)), key=grads.__getitem__) if grads else 0
    sug = min(lrs[i_steep], lrs[i_min] / 3)
    return {"suggested_lr": float(min(max(sug, 1e-4), 1e-2)), "lr_at_min_loss": lrs[i_min],
            "steps": len(lrs), "lrs": [round(l, 6) for l in lrs], "losses": [round(l, 4) for l in losses]}
