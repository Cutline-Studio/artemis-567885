"""Artemis 567885 expand engine.

Task: regression X -> sin(X) on random continuous vectors (no tokenizer); the
model output must have the same shape as its input.

config.json is the single source of truth for architecture + hyper-parameters
(d_model, n_heads, d_ff, num_experts per layer, num_layers, top_k,
capacity_factor, alpha, ...) and the metrics history. Python source is never
rewritten. Weights + optimizer live in runs/checkpoint.pt.

CLI
  python expand.py init  [--root R] [--force]   write a fresh config.json
  python expand.py run-cycle [--root R]          train ONE cycle using config.json as-is
  python expand.py loop --cycles N [--root R]    plan + mutate + run N cycles in-process
                                                 (supervisor.py does the same with backups,
                                                 subprocess isolation and rollback)
"""
import argparse, copy, json, math, os, sys, time
from typing import Dict, List

import torch
import torch.nn.functional as F

from model import DynamicMultiBlockTransformer
from moe import expert_load_entropy, expert_min_load_ratio

DEFAULT_CONFIG = {
    "d_model": 16, "n_heads": 2, "d_ff": 64,
    "num_layers": 1, "num_experts": [4], "base_num_experts": 4,
    "top_k": 2, "capacity_factor": 0.8, "alpha": 0.1,
    "lr": 1e-3, "steps_per_cycle": 300, "batch_size": 8, "seq_len": 16,
    "eval_batch_size": 32, "seed": 567885,
    "rules": {"cf_step": 0.05, "aux_threshold": 1.2, "alpha_mult": 1.5, "stall_rel_improvement": 0.01,
              "expert_noise_std": 0.01},  # growth-health rules: see GROWTH_RULE_DEFAULTS (used when absent)
    "caps": {"capacity_factor_max": 2.5, "num_experts_max": 16, "alpha_max": 1.0, "num_layers_max": 12},
    "cycle": 0, "pending_mutation": None, "history": [],
}
DIM_KEYS = ("d_model", "n_heads", "d_ff", "top_k")

# Growth-health rules (2026-10-27). They apply whenever a config's "rules" lacks the key, so existing
# configs pick them up without edits. The "legacy" column reproduces the pre-2026-10-27 expand loop.
GROWTH_RULE_DEFAULTS = {
    # fix 1: an expert may only be added to a layer whose existing experts all carry load.
    # min_load_ratio = E * min_i(load_i / total) over the last eval batch's top-k assignments
    # (1.0 = uniform, 0 = an idle expert); entropy is moe.expert_load_entropy (normalised).
    "growth_min_load_ratio": 0.25,         # legacy: 0.0
    "growth_min_load_entropy": 0.8,        # legacy: 0.0
    "growth_min_healthy_layer_frac": 0.5,  # a new LAYER needs >= this fraction of load-healthy layers; legacy: 0.0
    # fix 3: growth must pay for itself. After a growth the loop waits `stall_patience` cycles, then compares
    # the mean eval loss of those cycles with the mean of the `stall_patience` cycles BEFORE the growth
    # (a pre-growth baseline, so the post-growth optimizer restart cannot feed the next decision).
    # gain < min_growth_gain -> an expert growth escalates to a layer, a layer growth means "converged":
    # no structural growth until growth_cooldown cycles have passed since that growth.
    "stall_patience": 3,                   # legacy: 0 (disables the whole growth-pays check)
    "min_growth_gain": 0.01,
    "growth_cooldown": 25,
    "loss_floor": 0.0,                     # eval loss <= floor -> never grow (0 disables)
    # fix 4: how a new expert is initialised: "mean" (average of the layer's experts + noise, gate row = mean
    # row + noise) or "most_loaded" (legacy: clone of the most-loaded expert + noise).
    "expert_init": "mean",
}
LEGACY_GROWTH_RULES = {"growth_min_load_ratio": 0.0, "growth_min_load_entropy": 0.0,
                       "growth_min_healthy_layer_frac": 0.0, "stall_patience": 0, "expert_init": "most_loaded"}


def rule(rules: dict, key: str):
    return rules[key] if key in rules else GROWTH_RULE_DEFAULTS[key]


class OutputShapeMismatch(RuntimeError):
    pass


class NaNLoss(RuntimeError):
    pass


# ---------------------------------------------------------------- io helpers
def paths(root: str) -> Dict[str, str]:
    return {"config": os.path.join(root, "config.json"),
            "ckpt": os.path.join(root, "runs", "checkpoint.pt"),
            "log": os.path.join(root, "runs", "expand_log.jsonl")}


def load_config(root: str) -> dict:
    with open(paths(root)["config"]) as f:
        return json.load(f)


def save_config(root: str, cfg: dict):
    p = paths(root)["config"]
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    os.replace(tmp, p)


def init_config(root: str, force: bool = False, overrides: dict = None) -> dict:
    p = paths(root)
    if os.path.exists(p["config"]) and not force:
        return load_config(root)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        cfg.update(overrides)
    os.makedirs(os.path.dirname(p["ckpt"]), exist_ok=True)
    if os.path.exists(p["ckpt"]):
        os.remove(p["ckpt"])
    save_config(root, cfg)
    return cfg


# ---------------------------------------------------------------- planning
def plan_mutations(cfg: dict) -> List[dict]:
    """Apply the expand rules to the last cycle's metrics.

    Returns candidate mutation paths in priority order; the supervisor tries the
    next one if a path fails. Each mutation is {"name", "changes": {...}, "reasons": [...]}.
    The last candidate is always the no-op.
    """
    hist = cfg.get("history") or []
    noop = {"name": "none", "changes": {}, "reasons": []}
    if not hist:
        return [noop]
    last = hist[-1]
    rules, caps = cfg["rules"], cfg["caps"]
    soft, reasons = {}, []
    if last["dropped"] > 0 and cfg["capacity_factor"] < caps["capacity_factor_max"]:
        soft["capacity_factor"] = round(min(caps["capacity_factor_max"], cfg["capacity_factor"] + rules["cf_step"]), 6)
        reasons.append(f"dropped={last['dropped']}>0 -> capacity_factor+={rules['cf_step']}")
    if last["aux"] > rules["aux_threshold"] and cfg["alpha"] < caps["alpha_max"]:
        soft["alpha"] = round(min(caps["alpha_max"], cfg["alpha"] * rules["alpha_mult"]), 6)
        reasons.append(f"aux={last['aux']:.4f}>{rules['aux_threshold']} -> alpha*={rules['alpha_mult']}")

    structural, s_reason, notes = {}, None, []
    stalled = last["rel_improvement"] < rules["stall_rel_improvement"]
    floor = float(rule(rules, "loss_floor"))
    if stalled and floor > 0 and _eval_metric(last) <= floor:
        stalled = False
        notes.append(f"eval loss {_eval_metric(last):.6f} <= loss_floor {floor} -> no growth")
    if stalled:
        # caps["num_experts_max"] is a PER-LAYER cap on growth: a stall may only add an expert to a
        # layer that is below it. Layers already above it (trained before the cap was lowered) keep
        # all their experts; they just stop growing.
        experts = list(cfg["num_experts"])
        prev_mut = (last.get("mutation") or {}).get("changes", {})
        expert_added_last = "num_experts" in prev_mut and "num_layers" not in prev_mut
        verdict, v_note = growth_verdict(hist, rules)
        if v_note:
            notes.append(v_note)
        if verdict in ("paid", "cooldown_over"):
            expert_added_last = False  # last growth paid off (or was long ago): keep growing experts first
        elif verdict == "unpaid_expert":
            expert_added_last = True   # an expert didn't pay -> escalate to a layer
        health = layer_health(last, experts, rules)
        below_cap = [i for i, e in enumerate(experts) if e < caps["num_experts_max"]]
        open_layers = [i for i in below_cap if health[i]["healthy"]]
        unhealthy_open = [i for i in below_cap if not health[i]["healthy"]]
        for i in unhealthy_open:
            notes.append(f"layer {i} not load-healthy (min_load_ratio={health[i]['min_load_ratio']:.3f}, "
                         f"entropy={health[i]['entropy']:.3f}) -> no expert added there")
        all_full = not below_cap
        healthy_frac = sum(h["healthy"] for h in health) / max(1, len(health))
        layer_ok = healthy_frac >= float(rule(rules, "growth_min_healthy_layer_frac"))
        layer_room = cfg["num_layers"] < caps["num_layers_max"]
        want_layer = expert_added_last or all_full
        if verdict in ("settling", "converged"):
            pass
        elif want_layer and layer_room and layer_ok:
            structural = {"num_layers": cfg["num_layers"] + 1, "num_experts": experts + [cfg["base_num_experts"]]}
            s_reason = (f"stall rel_improvement={last['rel_improvement']:.4f} after "
                        f"{'expert addition' if expert_added_last else 'per-layer expert cap on every layer'} -> add layer")
        elif verdict == "unpaid_expert":
            notes.append("expert growth didn't pay and no layer can be added -> no growth")
        elif open_layers:
            layer = open_layers[-1]  # deepest load-healthy layer still below the per-layer cap
            experts[layer] += 1
            structural = {"num_experts": experts}
            s_reason = f"stall rel_improvement={last['rel_improvement']:.4f} -> add expert to layer {layer}"
        elif want_layer and layer_room:  # (not layer_ok)
            notes.append(f"only {healthy_frac:.2f} of layers load-healthy -> no layer added")

    cands = []
    if structural:
        cands.append({"name": "structural+soft", "changes": {**soft, **structural}, "reasons": reasons + [s_reason]})
    if soft:
        cands.append({"name": "soft", "changes": dict(soft), "reasons": list(reasons) + notes})
        half = {}
        if "capacity_factor" in soft:
            half["capacity_factor"] = round(cfg["capacity_factor"] + rules["cf_step"] / 2, 6)
        if "alpha" in soft:
            half["alpha"] = round(min(caps["alpha_max"], cfg["alpha"] * (1 + (rules["alpha_mult"] - 1) / 2)), 6)
        cands.append({"name": "soft_half_step", "changes": half, "reasons": [r + " (half step)" for r in reasons]})
    cands.append({**noop, "reasons": list(notes)} if notes else noop)
    return cands


def _eval_metric(rec: dict) -> float:
    """Per-cycle eval loss used by the growth rules: dropless when recorded (it measures the model, not
    the capacity drops), else the capacity-limited eval loss."""
    v = rec.get("loss_dropless")
    return float(rec["loss"] if v is None else v)


def _is_growth(rec: dict):
    """'layer' / 'expert' if this cycle trained right after a structural growth, else None."""
    before = rec.get("num_experts_before")
    if before is not None and "num_experts" in rec:
        if len(rec["num_experts"]) > len(before):
            return "layer"
        return "expert" if rec["num_experts"] != before else None
    ch = (rec.get("mutation") or {}).get("changes", {})
    if "num_layers" in ch:
        return "layer"
    return "expert" if "num_experts" in ch else None


def growth_verdict(hist: List[dict], rules: dict):
    """Did the most recent structural growth pay for itself? Returns (verdict, note).

    verdict: "legacy"        stall_patience <= 0 (rule disabled) or no judgeable growth in history
             "settling"      fewer than stall_patience cycles since the growth -> wait, don't grow again
             "paid"          mean eval loss over the post-growth window beat the pre-growth window by >= min_growth_gain
             "unpaid_expert" an expert growth didn't pay -> the planner may escalate to a layer
             "converged"     a layer growth didn't pay -> no structural growth until growth_cooldown
             "cooldown_over" growth_cooldown cycles passed since an unpaid growth -> one more try
    Comparing eval losses across cycles (not the within-cycle rel_improvement of the post-growth cycle)
    means the optimizer restart / perturbation that growth causes cannot trigger the next growth.
    """
    pat = int(rule(rules, "stall_patience"))
    if pat <= 0:
        return "legacy", None
    g = next((i for i in range(len(hist) - 1, -1, -1) if _is_growth(hist[i])), None)
    if not g:  # no growth yet, or it happened in the very first recorded cycle (no pre-growth baseline)
        return "legacy", None
    kind, since = _is_growth(hist[g]), len(hist) - g
    if since < pat:
        return "settling", f"{kind} growth {since} cycle(s) ago < stall_patience {pat} -> waiting"
    pre_w = hist[max(0, g - pat):g]
    pre = sum(_eval_metric(r) for r in pre_w) / len(pre_w)
    post = sum(_eval_metric(r) for r in hist[g:g + pat]) / pat
    gain = (pre - post) / pre if pre > 0 else 0.0
    msg = f"last {kind} growth: eval {pre:.6f} -> {post:.6f} (gain {gain:+.4f}, need {rule(rules, 'min_growth_gain')})"
    if gain >= float(rule(rules, "min_growth_gain")):
        return "paid", msg + " -> paid"
    if since >= int(rule(rules, "growth_cooldown")):
        return "cooldown_over", msg + f" -> unpaid but {since} cycles ago >= growth_cooldown, retry"
    if kind == "expert":
        return "unpaid_expert", msg + " -> unpaid, escalate"
    return "converged", msg + " -> unpaid layer growth: converged, no growth"


def layer_health(last: dict, experts: List[int], rules: dict) -> List[dict]:
    """Per-layer load health from the last cycle's eval-batch expert_load (top-k assignment counts).
    A layer without a matching load record (no data yet / just grown) counts as healthy."""
    loads = last.get("expert_load") or []
    rmin, hmin = float(rule(rules, "growth_min_load_ratio")), float(rule(rules, "growth_min_load_entropy"))
    out = []
    for i, e in enumerate(experts):
        l = loads[i] if i < len(loads) else None
        if not l or len(l) != e:
            out.append({"healthy": True, "min_load_ratio": None, "entropy": None})
            continue
        r, h = expert_min_load_ratio(l), expert_load_entropy(l)
        out.append({"healthy": r >= rmin and h >= hmin, "min_load_ratio": r, "entropy": h})
    return out


def apply_mutation(cfg: dict, mutation: dict) -> dict:
    before_layers, before_experts = cfg["num_layers"], list(cfg["num_experts"])
    cfg = copy.deepcopy(cfg)
    caps = cfg["caps"]
    for k, v in mutation["changes"].items():
        cfg[k] = copy.deepcopy(v)
    # hard caps, enforced whatever the mutation says
    cfg["capacity_factor"] = min(cfg["capacity_factor"], caps["capacity_factor_max"])
    cfg["alpha"] = min(cfg["alpha"], caps["alpha_max"])
    # structural caps block GROWTH only: existing trained layers/experts above a cap are never pruned
    cfg["num_layers"] = min(cfg["num_layers"], max(caps["num_layers_max"], before_layers))
    cfg["num_experts"] = [min(e, max(caps["num_experts_max"], before_experts[i] if i < len(before_experts) else 0))
                          for i, e in enumerate(cfg["num_experts"])][:cfg["num_layers"]]
    cfg["pending_mutation"] = mutation
    return cfg


# ---------------------------------------------------------------- model build / growth
def arch_of(cfg: dict) -> dict:
    return {"d_model": cfg["d_model"], "n_heads": cfg["n_heads"], "d_ff": cfg["d_ff"], "top_k": cfg["top_k"],
            "num_layers": cfg["num_layers"], "num_experts": list(cfg["num_experts"])}


def resolve_device(cfg: dict) -> torch.device:
    """cfg["device"]: "cpu" (default, the original behaviour), "cuda" or "auto"."""
    want = cfg.get("device", "cpu")
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(want)


def _carry_optimizer_state(opt, model, old_names: List[str], old_opt_state: dict) -> int:
    """Copy Adam moments for every parameter whose name AND shape survived growth.

    Growth only appends experts/layers and widens the gate, so all other parameters
    keep their trained Adam statistics instead of restarting from zero. Returns the
    number of parameters whose state was carried over.
    """
    old_state = old_opt_state.get("state", {})
    old_ids = old_opt_state["param_groups"][0]["params"]
    by_name = {old_names[i]: old_state.get(pid) for i, pid in enumerate(old_ids) if i < len(old_names)}
    carried = 0
    for name, p in model.named_parameters():
        st = by_name.get(name)
        if not st or "exp_avg" not in st or st["exp_avg"].shape != p.shape:
            continue
        opt.state[p] = {k: (v.to(p.device) if torch.is_tensor(v) and v.dim() > 0 else
                            (v.clone() if torch.is_tensor(v) else v)) for k, v in st.items()}
        carried += 1
    return carried


def build_model(root: str, cfg: dict, ck: dict = None):
    """Build the model from config, loading + growing the checkpoint if one exists.

    Returns (model, optimizer, info). Dimension changes (d_model/n_heads/d_ff/top_k)
    against an existing checkpoint are NOT silently re-initialised: load_state_dict
    raises a size-mismatch error that the supervisor can auto-correct.
    `ck` may be passed in to avoid loading the checkpoint file twice.
    """
    p = paths(root)
    dev = resolve_device(cfg)
    info = {"optimizer_reset": False, "optimizer_carried": 0, "weights_action": "fresh_init", "grown": [],
            "device": str(dev)}
    gen = torch.Generator().manual_seed(cfg["seed"] + 1000 + cfg["cycle"])
    if ck is None and os.path.exists(p["ckpt"]):
        ck = torch.load(p["ckpt"], map_location="cpu", weights_only=False)
    if ck is None:
        torch.manual_seed(cfg["seed"])
        model = DynamicMultiBlockTransformer(cfg["d_model"], cfg["n_heads"], cfg["d_ff"], cfg["num_experts"],
                                             cfg["num_layers"], cfg["top_k"], cfg["capacity_factor"]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        return model, opt, info

    a = ck["arch"]
    if cfg["num_layers"] < a["num_layers"] or any(
            c < o for c, o in zip(cfg["num_experts"], a["num_experts"])):
        raise RuntimeError(f"ShrinkNotSupported: config num_layers={cfg['num_layers']} num_experts={cfg['num_experts']} "
                           f"< checkpoint num_layers={a['num_layers']} num_experts={a['num_experts']}")
    # same dims as config, structure as checkpoint -> load -> grow to config
    model = DynamicMultiBlockTransformer(cfg["d_model"], cfg["n_heads"], cfg["d_ff"], a["num_experts"],
                                         a["num_layers"], cfg["top_k"], cfg["capacity_factor"])
    model.load_state_dict(ck["model"])
    model.to(dev)
    old_names = [n for n, _ in model.named_parameters()]
    info["weights_action"] = "loaded_checkpoint"
    loads = ck.get("expert_load") or []
    for layer, target in enumerate(cfg["num_experts"]):
        if layer >= model.num_layers:
            torch.manual_seed(cfg["seed"] + 2000 + cfg["cycle"])
            model.add_layer(cfg["base_num_experts"] if target is None else target)
            info["grown"].append(f"layer {layer} added near-identity with {target} experts (only this layer initialised)")
            continue
        init = rule(cfg["rules"], "expert_init")
        while model.blocks[layer].moe.num_experts < target:
            if init == "mean":
                new = model.add_expert(layer, None, cfg["rules"]["expert_noise_std"], gen, init="mean")
                info["grown"].append(f"layer {layer} expert {new} = mean of the layer's experts + noise")
                continue
            if init != "most_loaded":
                raise ValueError(f"rules.expert_init must be 'mean' or 'most_loaded', got {init!r}")
            src = 0
            if layer < len(loads) and loads[layer]:
                src = int(torch.tensor(loads[layer]).argmax().item())
            new = model.add_expert(layer, src, cfg["rules"]["expert_noise_std"], gen)
            info["grown"].append(f"layer {layer} expert {new} = copy of most-loaded expert {src} + noise")
    model.set_capacity_factor(cfg["capacity_factor"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    if a == arch_of(cfg) and ck.get("optimizer") is not None:
        opt.load_state_dict(ck["optimizer"])
        for st in opt.state.values():  # optimizer state follows the model's device
            for k, v in st.items():
                if torch.is_tensor(v) and v.dim() > 0:
                    st[k] = v.to(dev)
    else:
        info["optimizer_reset"] = True
        if cfg.get("carry_optimizer_on_growth") and ck.get("optimizer") is not None:
            info["optimizer_carried"] = _carry_optimizer_state(opt, model, old_names, ck["optimizer"])
    if info["grown"]:
        info["weights_action"] = "loaded_checkpoint+grown"
    return model, opt, info


def lr_at(cfg: dict, step: int) -> float:
    """Per-step learning rate. lr_schedule "constant" (default, original) or "cosine":
    linear warmup for warmup_steps, then cosine decay to lr_min over the cycle."""
    base = cfg["lr"]
    if cfg.get("lr_schedule", "constant") != "cosine":
        return base
    warm, total, lo = int(cfg.get("warmup_steps", 0)), cfg["steps_per_cycle"], float(cfg.get("lr_min", 0.0))
    if warm and step < warm:
        return base * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return lo + 0.5 * (base - lo) * (1 + math.cos(math.pi * min(1.0, t)))


def gate_grad_norm(model) -> float:
    gs = [b.moe.gate.weight.grad.flatten() for b in model.blocks if b.moe.gate.weight.grad is not None]
    return float(torch.cat(gs).norm()) if gs else 0.0


@torch.no_grad()
def eval_mse(model, x: torch.Tensor, dropless: bool = False) -> float:
    """MSE vs sin(x) in eval mode; dropless=True temporarily disables the capacity limit."""
    was = model.training
    model.eval()
    cf = model._capacity_factor
    if dropless:
        model.set_capacity_factor(None)
    try:
        y, _, _ = model(x)
        return F.mse_loss(y, torch.sin(x)).item()
    finally:
        model.set_capacity_factor(cf)
        model.train(was)


def run_cycle(root: str) -> dict:
    t0 = time.time()
    p = paths(root)
    cfg = load_config(root)
    torch.set_num_threads(int(cfg.get("num_threads", max(1, min(4, os.cpu_count() or 1)))))
    mutation = cfg.get("pending_mutation") or {"name": "none", "changes": {}, "reasons": []}
    ck = torch.load(p["ckpt"], map_location="cpu", weights_only=False) if os.path.exists(p["ckpt"]) else None
    num_experts_before = ck["arch"]["num_experts"] if ck else None
    model, opt, info = build_model(root, cfg, ck)
    dev = resolve_device(cfg)

    # optional exponential moving average of the weights (checkpoint averaging); restarted on growth
    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema = None
    if ema_decay > 0:
        ema = copy.deepcopy(model)
        if ck and ck.get("ema") is not None and not info["grown"] and ck["arch"] == arch_of(cfg):
            ema.load_state_dict(ck["ema"])
        for q in ema.parameters():
            q.requires_grad_(False)

    cyc = cfg["cycle"]
    rep = 100_000 * int(cfg.get("data_seed", 0))  # replicate runs: new train data + router noise, same eval sets
    g = torch.Generator().manual_seed(cfg["seed"] + cyc + rep)
    torch.manual_seed(cfg["seed"] + 10_000 + cyc + rep)  # router noise
    B, S, D = cfg["batch_size"], cfg["seq_len"], cfg["d_model"]
    alpha = cfg["alpha"]
    losses, auxes, drops = [], [], []
    model.train()
    for step in range(cfg["steps_per_cycle"]):
        lr = lr_at(cfg, step)
        for grp in opt.param_groups:
            grp["lr"] = lr
        x = torch.randn(B, S, D, generator=g).to(dev)  # generated on CPU: same data on every device
        y, aux, dropped = model(x)
        if y.shape != x.shape:
            raise OutputShapeMismatch(f"OutputShapeMismatch: output {tuple(y.shape)} != input {tuple(x.shape)}")
        target = F.mse_loss(y, torch.sin(x))
        loss = target + alpha * aux
        if not torch.isfinite(loss):
            raise NaNLoss(f"NaNLoss: non-finite loss at cycle {cyc} step {step}: {loss.item()}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for qe, qm in zip(ema.parameters(), model.parameters()):
                    qe.lerp_(qm, 1.0 - ema_decay)
        losses.append(target.item()); auxes.append(aux.item() / model.num_layers); drops.append(dropped)

    w = max(1, len(losses) // 5)
    start, end = sum(losses[:w]) / w, sum(losses[-w:]) / w
    rel = (start - end) / start if start > 0 else 0.0

    # evaluation on a fixed held-out batch (eval mode: no router noise) + grad norms
    eg = torch.Generator().manual_seed(cfg["seed"] - 1)
    xe = torch.randn(cfg["eval_batch_size"], S, D, generator=eg).to(dev).requires_grad_(True)
    model.eval(); model.zero_grad(set_to_none=True)
    ye, aux_e, dropped_e = model(xe)
    if ye.shape != xe.shape:
        raise OutputShapeMismatch(f"OutputShapeMismatch: output {tuple(ye.shape)} != input {tuple(xe.shape)}")
    eval_target = F.mse_loss(ye, torch.sin(xe.detach()))
    if not torch.isfinite(eval_target):
        raise NaNLoss(f"NaNLoss: non-finite eval loss at cycle {cyc}")
    (eval_target + alpha * aux_e).backward()
    loads = []
    with torch.no_grad():
        h = xe.detach()
        for b in model.blocks:
            hn = b.ln2(h + b.attn(b.ln1(h)))
            idx = b.moe.gate(hn.reshape(-1, D)).topk(b.moe.top_k, -1).indices
            loads.append(torch.bincount(idx.flatten(), minlength=b.moe.num_experts).tolist())
            h, _, _ = b(h)
    dropped_tokens = sum(b.moe.last_stats.get("dropped_tokens", 0) for b in model.blocks)
    extra = {}
    if cfg.get("eval_extra", False):
        xd = xe.detach()
        extra["loss_dropless"] = round(eval_mse(model, xd, dropless=True), 6)
        if ema is not None:
            extra["ema_loss"] = round(eval_mse(ema, xd), 6)
            extra["ema_loss_dropless"] = round(eval_mse(ema, xd, dropless=True), 6)

    rec = {
        "cycle": cyc, "loss": round(eval_target.item(), 6), "train_loss_start": round(start, 6),
        "train_loss_end": round(end, 6), "rel_improvement": round(rel, 6),
        "aux": round(aux_e.item() / model.num_layers, 6), "aux_train_mean": round(sum(auxes) / len(auxes), 6),
        "dropped": int(dropped_e), "dropped_tokens": int(dropped_tokens),
        "dropped_train_mean": round(sum(drops) / len(drops), 3),
        "capacity_factor": cfg["capacity_factor"], "alpha": cfg["alpha"],
        "num_layers": model.num_layers, "num_experts": model.num_experts, "num_experts_before": num_experts_before,
        "grad_x_norm": round(xe.grad.norm().item(), 6), "grad_gate_norm": round(gate_grad_norm(model), 6),
        "output_shape_ok": True, "expert_load": loads,
        "expert_load_entropy": [round(expert_load_entropy(l), 6) for l in loads],
        "expert_min_load_ratio": [round(expert_min_load_ratio(l), 6) for l in loads], "mutation": mutation,
        "optimizer_reset": info["optimizer_reset"], "optimizer_carried": info["optimizer_carried"],
        "weights_action": info["weights_action"], "grown": info["grown"], "device": info["device"],
        "steps": cfg["steps_per_cycle"], "seconds": round(time.time() - t0, 2),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra,
    }
    # persist: checkpoint (always CPU tensors, so it loads anywhere), config (state write-back), jsonl log
    os.makedirs(os.path.dirname(p["ckpt"]), exist_ok=True)
    cpu = lambda sd: {k: v.detach().cpu() for k, v in sd.items()}
    opt_sd = opt.state_dict()  # its per-param dicts alias the live state, so build copies
    opt_sd = {"state": {i: {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in st.items()}
                        for i, st in opt_sd["state"].items()}, "param_groups": opt_sd["param_groups"]}
    torch.save({"arch": arch_of(cfg), "model": cpu(model.state_dict()), "optimizer": opt_sd,
                "ema": cpu(ema.state_dict()) if ema is not None else None,
                "expert_load": loads, "cycle": cyc}, p["ckpt"] + ".tmp")
    os.replace(p["ckpt"] + ".tmp", p["ckpt"])
    cfg["history"].append(rec)
    cfg["cycle"] = cyc + 1
    cfg["pending_mutation"] = None
    save_config(root, cfg)
    with open(p["log"], "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def loop(root: str, cycles: int) -> List[dict]:
    out = []
    for _ in range(cycles):
        cfg = load_config(root)
        save_config(root, apply_mutation(cfg, plan_mutations(cfg)[0]))
        out.append(run_cycle(root))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["init", "run-cycle", "loop"])
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        init_config(a.root, force=a.force)
        print(json.dumps({"ok": True, "config": paths(a.root)["config"]}))
    elif a.cmd == "run-cycle":
        rec = run_cycle(a.root)
        print("RESULT " + json.dumps({k: rec[k] for k in ("cycle", "loss", "aux", "dropped", "capacity_factor",
                                                         "alpha", "num_experts", "num_layers", "grad_x_norm",
                                                         "grad_gate_norm", "output_shape_ok")}))
    else:
        for rec in loop(a.root, a.cycles):
            print(json.dumps({k: rec[k] for k in ("cycle", "loss", "aux", "dropped", "capacity_factor", "alpha",
                                                  "num_experts", "num_layers")}))


if __name__ == "__main__":
    main()
