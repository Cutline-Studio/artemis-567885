"""Summarise the growth-health A/B arms (tools/ab_growth_health/run_ab.sh) into one JSON."""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from moe import expert_load_entropy, expert_min_load_ratio  # noqa: E402

W, O = sys.argv[1], sys.argv[2]
START = {"F": 0, "C": 365}
out = {}
for arm in ("F_old", "F_new", "F_new_ml", "C_old", "C_new"):
    try:
        cfg = json.load(open(f"{W}/arm_{arm}/config.json"))
        ev = json.load(open(f"{O}/eval_{arm}.json"))
    except Exception as e:  # noqa: BLE001
        out[arm] = {"error": repr(e)}
        continue
    h = [r for r in cfg["history"] if r["cycle"] >= START[arm[0]]]
    growth = [r for r in h if r.get("num_experts_before") is not None and r["num_experts"] != r["num_experts_before"]]
    last = h[-1]
    tail = h[-5:]
    out[arm] = {
        "cycles_run": len(h), "first_cycle": h[0]["cycle"], "last_cycle": last["cycle"],
        "growth_events": len(growth),
        "layer_adds": sum(len(r["num_experts"]) > len(r["num_experts_before"]) for r in growth),
        "num_layers": last["num_layers"], "num_experts": last["num_experts"], "total_experts": sum(last["num_experts"]),
        "load_entropy_per_layer": [round(expert_load_entropy(l), 4) for l in last["expert_load"]],
        "min_load_ratio_per_layer": [round(expert_min_load_ratio(l), 4) for l in last["expert_load"]],
        "mean_load_entropy": round(sum(expert_load_entropy(l) for l in last["expert_load"]) / len(last["expert_load"]), 4),
        "layers_with_idle_expert": sum(min(l) == 0 for l in last["expert_load"]),
        "eval_loss_last5_mean": round(sum(r["loss"] for r in tail) / len(tail), 6),
        "eval_loss_dropless_last5_mean": round(sum(r.get("loss_dropless", r["loss"]) for r in tail) / len(tail), 6),
        "sec_per_cycle": round(sum(r["seconds"] for r in h) / len(h), 2),
        "heldout": {k: {m: ev[k][m] for m in ("heldout_mse", "heldout_mse_dropless", "r2", "ood_mse")}
                    for k in ("raw", "ema") if k in ev},
    }
try:
    ev = json.load(open(f"{O}/eval_C_start365.json"))
    out["C_start365"] = {k: {m: ev[k][m] for m in ("heldout_mse", "heldout_mse_dropless", "r2", "ood_mse")}
                         for k in ("raw", "ema") if k in ev}
except Exception as e:  # noqa: BLE001
    out["C_start365"] = {"error": repr(e)}
print(json.dumps(out, indent=2))
