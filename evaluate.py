"""Fixed held-out evaluation + task checks for the Artemis 567885 expand engine.

The training objective is y = sin(x) element-wise on x ~ N(0, 1) vectors. The
model's built-in per-cycle eval uses one 32-sequence batch, which is too small
to tell real gains from noise, and it runs with the capacity limit on (some
tokens are dropped in eval). This script scores a checkpoint on a larger,
fixed held-out set that training never samples (its own seed), and reports:

  heldout_mse            MSE vs sin(x), capacity limit as configured
  heldout_mse_dropless   same, with dropless routing (every token reaches its experts)
  r2                     1 - heldout_mse / Var(sin x) on the held-out set
  baseline_zero_mse      predict 0            (reference)
  baseline_identity_mse  predict x            (reference)
  bucket_mse             MSE split by |x| in [0,1), [1,2), [2,inf)
  ood_mse                x ~ U(+-[2, 3.5])    (tails rarely seen in training)
  odd_symmetry_err       mean (f(-x) + f(x))^2   (sin is odd; 0 is perfect)
  token_order_err        mean (f(perm x) - perm f(x))^2 over token order (target is per-token,
                         so any value > 0 is cross-token interference through attention/capacity)
  legacy_eval_loss       the original 32-sequence eval batch (for continuity with config history)

EMA weights (checkpoint["ema"]) are scored too when present.

CLI: python evaluate.py [--root R] [--ckpt PATH] [--n-seqs 1024] [--device auto] [--out FILE]
"""
import argparse, json, os, time

import torch
import torch.nn.functional as F

import expand
from model import DynamicMultiBlockTransformer

HELDOUT_SEED_OFFSET = 777_777  # training uses seed + cycle (< 10_000); eval batch uses seed - 1


def load(root, ckpt_path, device):
    cfg = expand.load_config(root)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["arch"]
    def make(sd):
        m = DynamicMultiBlockTransformer(a["d_model"], a["n_heads"], a["d_ff"], a["num_experts"], a["num_layers"],
                                         a["top_k"], cfg["capacity_factor"])
        m.load_state_dict(sd)
        return m.to(device).eval()
    models = {"raw": make(ck["model"])}
    if ck.get("ema") is not None:
        models["ema"] = make(ck["ema"])
    return cfg, ck, models


@torch.no_grad()
def predict(model, x, dropless=False, chunk=128):
    cf = model._capacity_factor
    if dropless:
        model.set_capacity_factor(None)
    try:  # fixed chunk size so capacity behaviour matches the training batch scale, not the eval-set size
        return torch.cat([model(x[i:i + chunk])[0] for i in range(0, x.shape[0], chunk)])
    finally:
        model.set_capacity_factor(cf)


@torch.no_grad()
def score(model, cfg, n_seqs, device):
    S, D, seed = cfg["seq_len"], cfg["d_model"], cfg["seed"]
    g = torch.Generator().manual_seed(seed + HELDOUT_SEED_OFFSET)
    x = torch.randn(n_seqs, S, D, generator=g).to(device)
    t = torch.sin(x)
    chunk = cfg["eval_batch_size"]
    y = predict(model, x, chunk=chunk)
    yd = predict(model, x, dropless=True, chunk=chunk)
    se = (y - t) ** 2
    out = {"heldout_mse": se.mean().item(), "heldout_mse_dropless": F.mse_loss(yd, t).item(),
           "r2": 1 - se.mean().item() / t.var().item(),
           "baseline_zero_mse": (t ** 2).mean().item(), "baseline_identity_mse": F.mse_loss(x, t).item()}
    ax = x.abs()
    out["bucket_mse"] = {name: se[m].mean().item() for name, m in
                         (("|x|<1", ax < 1), ("1<=|x|<2", (ax >= 1) & (ax < 2)), ("|x|>=2", ax >= 2))}
    go = torch.Generator().manual_seed(seed + HELDOUT_SEED_OFFSET + 1)
    mag = 2 + 1.5 * torch.rand(n_seqs // 4, S, D, generator=go)
    sign = torch.where(torch.rand(n_seqs // 4, S, D, generator=go) < 0.5, -1.0, 1.0)
    xo = (mag * sign).to(device)
    out["ood_mse"] = F.mse_loss(predict(model, xo, chunk=chunk), torch.sin(xo)).item()
    out["odd_symmetry_err"] = ((predict(model, -x, chunk=chunk) + y) ** 2).mean().item()
    perm = torch.randperm(S, generator=torch.Generator().manual_seed(seed + HELDOUT_SEED_OFFSET + 2)).to(device)
    out["token_order_err"] = ((predict(model, x[:, perm], chunk=chunk) - y[:, perm]) ** 2).mean().item()
    ge = torch.Generator().manual_seed(seed - 1)
    xl = torch.randn(cfg["eval_batch_size"], S, D, generator=ge).to(device)
    out["legacy_eval_loss"] = F.mse_loss(model(xl)[0], torch.sin(xl)).item()
    return {k: (round(v, 6) if isinstance(v, float) else {kk: round(vv, 6) for kk, vv in v.items()})
            for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-seqs", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                       ("cpu" if a.device == "auto" else a.device))
    torch.manual_seed(0)
    cfg, ck, models = load(a.root, a.ckpt or expand.paths(a.root)["ckpt"], dev)
    res = {"cycle": ck.get("cycle"), "arch": ck["arch"], "n_seqs": a.n_seqs, "device": str(dev),
           "torch": torch.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           **{name: score(m, cfg, a.n_seqs, dev) for name, m in models.items()}}
    s = json.dumps(res, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s + "\n")
    print(s)
    return res


if __name__ == "__main__":
    main()
