"""Artemis 567885 physics sandbox: coordinate-routed SIREN mixture-of-experts PINN.

EXPLORATORY (repo-checks, check 1): this file does NOT use the historical
Artemis pilot setup restored in checks/historical.py. Deviations: target is a
Gaussian bump advected on [-1,1]^2 x [0,1] (historical: sin(k(x - 0.5 t)) on
[0,2pi]^2 x [0,2]); inputs are not the historical x/pi-1, y/pi-1, t-1
normalisation; SIREN biases keep nn.Linear's default init (historical: bias
~ U(-b, b) with the same bound as the weights).

Separate from the sin(X) expand engine. Solves 2-D linear advection

    dh/dt + c * (dh/dx + dh/dy) = 0,   (x, y) in [-1, 1]^2, t in [0, 1]

with initial condition h(x, y, 0) = h0(x, y) (Gaussian bump) and inflow boundary
values on x = -1 and y = -1. The exact solution is the travelling wave
h(x, y, t) = h0(x - c t, y - c t), used for evaluation only.

Why the IC/BC terms: the PDE residual alone is minimised exactly by any
constant field (h = const => all derivatives 0), so a residual-only loss has a
trivial solution. tests/test_physics.py demonstrates that.

Fixes relative to the user's sketch (source reconstructed from the spec):
  * `uniform__` -> `uniform_`
  * `pos indices, k_pos` (syntax error) -> `pos_indices, k_pos`
  * SIREN init: first layer U(-1/in, 1/in); hidden layers U(-sqrt(6/in)/omega0, +sqrt(6/in)/omega0)
  * expert outputs are scattered with out-of-place `index_add`; the in-place
    `output_field[pos] += ...` variant is also verified (it works with
    create_graph second-order gradients in torch 2.14, see test), but index_add
    avoids any version-counter / aliasing surprises.

Run:  python physics_simulation.py --seconds 120

Stage 8b options (all OFF by default; the defaults reproduce the iteration-1 run):
  * lr / lr_schedule="cosine" (+ min_lr_frac): cosine decay over max_steps
  * alpha (aux load-balance weight) and noise_std: Gaussian noise added to the
    router logits before top-k, in train mode only (model.eval() routes cleanly)
  * causal_eps > 0: causal residual weighting over n_t_bins time bins
    (Wang, Sankaran & Perdikaris 2022, "Respecting causality is all you need"):
    w_i = exp(-eps * sum_{j<i} L_j) (weights detached), L_pde = mean_i w_i L_i
  * curriculum_frac > 0: residual/BC samples use t in [0, t_max(step)], with
    t_max growing linearly from curriculum_t0 to 1 over curriculum_frac*max_steps
  * error_vs_t(model, ts): rel. L2 vs the analytic solution on a 41x41 grid per t
"""
import argparse, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F


class SineLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, is_first: bool = False, omega0: float = 30.0):
        super().__init__()
        self.in_f, self.is_first, self.omega0 = in_f, is_first, omega0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            bound = 1.0 / in_f if is_first else math.sqrt(6.0 / in_f) / omega0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


class PhysicsExpert(nn.Module):
    """SIREN: (x, y, t) -> h."""

    def __init__(self, in_dim: int = 3, hidden: int = 64, depth: int = 3, omega0: float = 30.0):
        super().__init__()
        layers = [SineLayer(in_dim, hidden, is_first=True, omega0=omega0)]
        layers += [SineLayer(hidden, hidden, omega0=omega0) for _ in range(depth - 1)]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 1)
        with torch.no_grad():
            b = math.sqrt(6.0 / hidden) / omega0
            self.out.weight.uniform_(-b, b)

    def forward(self, x):
        return self.out(self.net(x))


class CoordinateMoERouter(nn.Module):
    """Routes each coordinate to top-k experts from a small MLP on (x, y, t)."""

    def __init__(self, in_dim: int, num_experts: int, top_k: int = 2, hidden: int = 32, noise_std: float = 0.0):
        super().__init__()
        self.top_k, self.num_experts, self.noise_std = top_k, num_experts, noise_std
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, num_experts))

    def forward(self, coords):
        logits = self.net(coords)
        probs = F.softmax(logits, dim=-1)  # clean probabilities (aux loss)
        if self.training and self.noise_std > 0:  # noisy top-k routing, train mode only
            logits = logits + torch.randn_like(logits) * self.noise_std
        top_w, top_i = torch.topk(logits, self.top_k, dim=-1)
        return F.softmax(top_w, dim=-1), top_i, probs


class PhysicsMoELayer(nn.Module):
    def __init__(self, num_experts: int = 4, top_k: int = 2, hidden: int = 64, depth: int = 3,
                 omega0: float = 30.0, scatter: str = "index_add", noise_std: float = 0.0):
        super().__init__()
        self.router = CoordinateMoERouter(3, num_experts, top_k, noise_std=noise_std)
        self.experts = nn.ModuleList([PhysicsExpert(3, hidden, depth, omega0) for _ in range(num_experts)])
        self.num_experts, self.top_k, self.scatter = num_experts, top_k, scatter

    def forward(self, coords):
        weights, idx, probs = self.router(coords)
        n = coords.shape[0]
        output_field = coords.new_zeros(n, 1)
        for e in range(self.num_experts):
            pos_indices, k_pos = (idx == e).nonzero(as_tuple=True)
            if pos_indices.numel() == 0:
                continue
            contrib = self.experts[e](coords[pos_indices]) * weights[pos_indices, k_pos].unsqueeze(-1)
            if self.scatter == "index_add":
                output_field = output_field.index_add(0, pos_indices, contrib)
            else:  # the sketch's in-place form, kept for verification
                output_field[pos_indices] += contrib
        me = F.one_hot(idx, self.num_experts).float().sum(1).mean(0) / self.top_k
        aux = self.num_experts * (me * probs.mean(0)).sum()
        return output_field, aux, idx


# ------------------------------------------------------------------ problem
C = 0.5
X0, Y0, SIGMA = -0.4, -0.4, 0.25


def h0(x, y):
    return torch.exp(-((x - X0) ** 2 + (y - Y0) ** 2) / (2 * SIGMA ** 2))


def exact(x, y, t):
    return h0(x - C * t, y - C * t)


def pde_residual(model, coords):
    coords = coords.clone().requires_grad_(True)
    h, aux, idx = model(coords)
    g = torch.autograd.grad(h.sum(), coords, create_graph=True)[0]
    r = g[:, 2:3] + C * (g[:, 0:1] + g[:, 1:2])
    return r, aux


def sample(n, gen):
    xy = torch.rand(n, 2, generator=gen) * 2 - 1
    t = torch.rand(n, 1, generator=gen)
    return torch.cat([xy, t], 1)


def causal_weights(bin_losses, eps: float):
    """w_i = exp(-eps * sum_{j<i} L_j), detached (no gradient through the weights)."""
    L = bin_losses.detach()
    prev = torch.cumsum(L, 0) - L
    return torch.exp(-eps * prev)


def causal_pde_loss(r, t, eps: float, n_bins: int = 16, t_max: float = 1.0):
    """Causally weighted residual loss: bin points by t into n_bins equal windows of
    [0, t_max]; L_i = mean r^2 in bin i; return (mean_i w_i L_i, w, L)."""
    b = torch.clamp((t.flatten() / t_max * n_bins).long(), 0, n_bins - 1)
    r2 = r.flatten().pow(2)
    sums = torch.zeros(n_bins, dtype=r2.dtype).index_add(0, b, r2)
    cnt = torch.bincount(b, minlength=n_bins).clamp(min=1).to(r2.dtype)
    L = sums / cnt
    w = causal_weights(L, eps)
    return (w * L).mean(), w, L


def curriculum_tmax(step: int, max_steps: int, frac: float, t0: float = 0.1):
    if frac <= 0:
        return 1.0
    return min(1.0, t0 + (1.0 - t0) * step / max(1.0, frac * max_steps))


def lr_at(step: int, max_steps: int, lr: float, schedule: str = "constant", min_lr_frac: float = 0.0):
    if schedule == "constant":
        return lr
    if schedule == "cosine":
        p = min(1.0, step / max(1, max_steps))
        return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))
    raise ValueError(f"unknown lr_schedule {schedule!r}")


def losses(model, n, gen, w_ic=10.0, w_bc=10.0, t_max: float = 1.0, causal_eps: float = 0.0, n_t_bins: int = 16):
    pts = sample(n, gen)
    if t_max < 1.0:
        pts[:, 2] *= t_max
    r, aux = pde_residual(model, pts)
    if causal_eps > 0:
        l_pde, _, _ = causal_pde_loss(r, pts[:, 2], causal_eps, n_t_bins, t_max)
    else:
        l_pde = r.pow(2).mean()
    ic = sample(n // 2, gen); ic[:, 2] = 0
    h_ic, _, _ = model(ic)
    l_ic = (h_ic - h0(ic[:, :1], ic[:, 1:2])).pow(2).mean()
    bc = sample(n // 2, gen)
    if t_max < 1.0:
        bc[:, 2] *= t_max
    side = torch.rand(n // 2, generator=gen) < 0.5
    bc[side, 0] = -1.0; bc[~side, 1] = -1.0  # inflow boundaries (c > 0)
    h_bc, _, _ = model(bc)
    l_bc = (h_bc - exact(bc[:, :1], bc[:, 1:2], bc[:, 2:3])).pow(2).mean()
    return l_pde, l_ic, l_bc, aux, w_ic, w_bc


class _eval_mode:
    """Temporarily put a module in eval mode (no-op for the default, noise-free router)."""
    def __init__(self, m):
        self.m = m
    def __enter__(self):
        self.was = self.m.training; self.m.eval()
    def __exit__(self, *a):
        self.m.train(self.was)


def expert_usage_stats(counts):
    """Experts used (count > 0) and normalised entropy (1 = perfectly even) of a usage histogram."""
    c = torch.tensor(counts, dtype=torch.float64)
    p = c / c.sum()
    nz = p[p > 0]
    ent = float(-(nz * nz.log()).sum() / math.log(len(counts))) if len(counts) > 1 else 1.0
    return {"experts_used": int((c > 0).sum()), "usage_entropy": round(ent, 6) + 0.0}  # + 0.0: no -0.0


@torch.no_grad()
def error_vs_t(model, ts=None, res: int = 41):
    """Rel. L2 and max abs error vs the analytic solution on a res x res grid at each t."""
    ts = [i / 10 for i in range(11)] if ts is None else ts
    lin = torch.linspace(-1, 1, res)
    X, Y = torch.meshgrid(lin, lin, indexing="ij")
    rows = []
    with _eval_mode(model):
        for t in ts:
            coords = torch.stack([X.flatten(), Y.flatten(), torch.full_like(X.flatten(), float(t))], 1)
            h, _, _ = model(coords)
            ex = exact(coords[:, :1], coords[:, 1:2], coords[:, 2:3])
            rows.append({"t": round(float(t), 4), "rel_l2": round(float(torch.norm(h - ex) / torch.norm(ex)), 6),
                         "max_abs_err": round(float((h - ex).abs().max()), 6)})
    return rows


@torch.no_grad()
def evaluate(model, res: int = 41):
    with _eval_mode(model):
        return _evaluate(model, res)


def _evaluate(model, res: int = 41):
    lin = torch.linspace(-1, 1, res)
    ts = torch.tensor([0.0, 0.5, 1.0])
    out = {}
    errs_num, errs_den = 0.0, 0.0
    for t in ts:
        X, Y = torch.meshgrid(lin, lin, indexing="ij")
        coords = torch.stack([X.flatten(), Y.flatten(), torch.full_like(X.flatten(), float(t))], 1)
        h, _, _ = model(coords)
        ex = exact(coords[:, :1], coords[:, 1:2], coords[:, 2:3])
        num, den = (h - ex).pow(2).sum().item(), ex.pow(2).sum().item()
        out[f"rel_l2_t{float(t):.1f}"] = round(math.sqrt(num / den), 6)
        out[f"max_abs_err_t{float(t):.1f}"] = round((h - ex).abs().max().item(), 6)
        errs_num += num; errs_den += den
    out["rel_l2_all"] = round(math.sqrt(errs_num / errs_den), 6)
    # expert usage by region (top-1 choice) on a 21^3 grid
    g = torch.linspace(-1, 1, 21); gt = torch.linspace(0, 1, 21)
    X, Y, T = torch.meshgrid(g, g, gt, indexing="ij")
    coords = torch.stack([X.flatten(), Y.flatten(), T.flatten()], 1)
    _, idx, _ = model.router(coords)
    top1 = idx[:, 0]
    regions = {}
    for name, m in {"x<0,y<0": (coords[:, 0] < 0) & (coords[:, 1] < 0), "x>=0,y<0": (coords[:, 0] >= 0) & (coords[:, 1] < 0),
                    "x<0,y>=0": (coords[:, 0] < 0) & (coords[:, 1] >= 0), "x>=0,y>=0": (coords[:, 0] >= 0) & (coords[:, 1] >= 0)}.items():
        for tname, tm in {"t<0.5": coords[:, 2] < 0.5, "t>=0.5": coords[:, 2] >= 0.5}.items():
            regions[f"{name},{tname}"] = torch.bincount(top1[m & tm], minlength=model.num_experts).tolist()
    out["expert_usage_top1_total"] = torch.bincount(top1, minlength=model.num_experts).tolist()
    out["expert_usage_top1_by_region"] = regions
    out.update(expert_usage_stats(out["expert_usage_top1_total"]))
    return out


def heldout_reference(n: int = 161, times=(0.25, 0.5, 0.75, 1.0), stride: int = 4):
    """Held-out solution-error set (repo-checks, check 4).

    Reference values come from the validated Lax-Wendroff solver
    (physics_solver.py, order ~2 measured) on an n x n grid; every `stride`-th
    node is used at each time in `times`. Training samples are drawn from a
    continuous uniform generator, so these exact grid nodes are never trained on.
    The analytic values are returned too, so the solver's own error is visible.
    """
    import numpy as np
    import physics_solver as S
    x, X, Y, out, info = S.solve(n, max(times), "laxwendroff", snapshots=tuple(times))
    cs, fd, ex = [], [], []
    for t in times:
        Xs, Ys, U = X[::stride, ::stride], Y[::stride, ::stride], out[t][::stride, ::stride]
        cs.append(np.stack([Xs.ravel(), Ys.ravel(), np.full(Xs.size, t)], 1))
        fd.append(U.ravel()); ex.append(S.exact_np(Xs, Ys, t).ravel())
    coords = torch.tensor(np.concatenate(cs), dtype=torch.float32)
    fd = torch.tensor(np.concatenate(fd), dtype=torch.float32).unsqueeze(1)
    ex = torch.tensor(np.concatenate(ex), dtype=torch.float32).unsqueeze(1)
    return {"coords": coords, "fd": fd, "exact": ex, "solver": dict(n=n, stride=stride, times=list(times), **info),
            "fd_vs_exact_rel_l2": float(torch.norm(fd - ex) / torch.norm(ex))}


@torch.no_grad()
def heldout_error(model, ref):
    with _eval_mode(model):
        h, _, _ = model(ref["coords"])
    return {"heldout_rel_l2_vs_fd": float(torch.norm(h - ref["fd"]) / torch.norm(ref["fd"])),
            "heldout_rel_l2_vs_exact": float(torch.norm(h - ref["exact"]) / torch.norm(ref["exact"])),
            "heldout_max_abs_vs_fd": float((h - ref["fd"]).abs().max())}


def train(seconds: float = 120.0, max_steps: int = 200_000, n: int = 1024, lr: float = 1e-3, seed: int = 567885,
          num_experts: int = 4, top_k: int = 2, hidden: int = 64, omega0: float = 30.0, alpha: float = 0.01,
          log_every: int = 250, log_path: str = None, heldout: dict = None,
          lr_schedule: str = "constant", min_lr_frac: float = 0.0, noise_std: float = 0.0,
          causal_eps: float = 0.0, n_t_bins: int = 16, curriculum_frac: float = 0.0, curriculum_t0: float = 0.1,
          threads: int = None):
    """Stage 8b options default to the iteration-1 behaviour. lr_schedule="cosine" and
    curriculum_frac > 0 are defined over max_steps, so use a step budget (large `seconds`)."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads if threads else max(1, min(8, os.cpu_count() or 1)))
    gen = torch.Generator().manual_seed(seed)
    model = PhysicsMoELayer(num_experts, top_k, hidden, 3, omega0, noise_std=noise_std)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0, step, hist = time.time(), 0, []
    while step < max_steps and time.time() - t0 < seconds:
        if lr_schedule != "constant":
            for g in opt.param_groups:
                g["lr"] = lr_at(step, max_steps, lr, lr_schedule, min_lr_frac)
        t_max = curriculum_tmax(step, max_steps, curriculum_frac, curriculum_t0)
        l_pde, l_ic, l_bc, aux, w_ic, w_bc = losses(model, n, gen, t_max=t_max, causal_eps=causal_eps, n_t_bins=n_t_bins)
        loss = l_pde + w_ic * l_ic + w_bc * l_bc + alpha * aux
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaNLoss at step {step}")
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        step += 1
        if step % log_every == 0 or step == 1:
            rec = {"step": step, "sec": round(time.time() - t0, 1), "pde": l_pde.item(), "ic": l_ic.item(),
                   "bc": l_bc.item(), "aux": aux.item()}
            if lr_schedule != "constant":
                rec["lr"] = opt.param_groups[0]["lr"]
            if curriculum_frac > 0:
                rec["t_max"] = t_max
            if heldout is not None:
                rec.update(heldout_error(model, heldout))
            hist.append(rec)
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    # final losses on a fresh, larger sample
    eg = torch.Generator().manual_seed(seed - 1)
    with _eval_mode(model):
        l_pde, l_ic, l_bc, aux, _, _ = losses(model, 8192, eg)
    final = {"steps": step, "seconds": round(time.time() - t0, 1), "pde_residual_mse": l_pde.item(),
             "ic_mse": l_ic.item(), "bc_mse": l_bc.item(), "aux": aux.item(), **evaluate(model)}
    if heldout is not None:
        final.update(heldout_error(model, heldout))
    return model, final, hist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--points", type=int, default=1024)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=567885)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    ap.add_argument("--min-lr-frac", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--noise-std", type=float, default=0.0)
    ap.add_argument("--causal-eps", type=float, default=0.0)
    ap.add_argument("--t-bins", type=int, default=16)
    ap.add_argument("--curriculum-frac", type=float, default=0.0)
    ap.add_argument("--curriculum-t0", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "physics"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "train_log.jsonl")
    if os.path.exists(log):
        os.remove(log)
    model, final, hist = train(a.seconds, a.max_steps, a.points, lr=a.lr, seed=a.seed, num_experts=a.experts,
                               alpha=a.alpha, log_path=log, lr_schedule=a.lr_schedule, min_lr_frac=a.min_lr_frac,
                               noise_std=a.noise_std, causal_eps=a.causal_eps, n_t_bins=a.t_bins,
                               curriculum_frac=a.curriculum_frac, curriculum_t0=a.curriculum_t0, threads=a.threads)
    with open(os.path.join(a.out, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
