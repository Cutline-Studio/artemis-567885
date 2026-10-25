"""Aggregate runs/stage8b/*/final.json + error_vs_t.json into summary.csv / summary.json."""
import csv, glob, json, os, re, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "runs", "stage8b")
rows = []
for f in sorted(glob.glob(os.path.join(D, "*", "final.json"))):
    name = os.path.basename(os.path.dirname(f)); d = json.load(open(f))
    evt = json.load(open(os.path.join(os.path.dirname(f), "error_vs_t.json")))
    cfg = d["config"]
    rows.append(dict(run=name, config=re.sub(r"_s\d$", "", name), seed=cfg.get("seed", 567885), threads=cfg["threads"],
                     steps=d["steps"], kwargs=json.dumps({k: v for k, v in cfg.items() if k not in ("name", "steps", "threads", "seed")}),
                     rel_l2_t0=d["rel_l2_t0.0"], rel_l2_t05=d["rel_l2_t0.5"], rel_l2_t1=d["rel_l2_t1.0"], rel_l2_all=d["rel_l2_all"],
                     max_abs_err_t1=d["max_abs_err_t1.0"], pde_mse=d["pde_residual_mse"], ic_mse=d["ic_mse"], bc_mse=d["bc_mse"],
                     expert_usage=json.dumps(d["expert_usage_top1_total"]), experts_used=d["experts_used"],
                     usage_entropy=d["usage_entropy"], wall_s=d["wall_seconds"],
                     err_vs_t=json.dumps([r["rel_l2"] for r in evt])))
with open(os.path.join(D, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
agg = {}
for r in rows:
    if r["threads"] != 2:
        continue
    agg.setdefault(r["config"], []).append(r)
summ = {c: dict(n=len(v), seeds=[r["seed"] for r in v], rel_l2_t1=[r["rel_l2_t1"] for r in v],
                mean_rel_l2_t1=round(st.mean(r["rel_l2_t1"] for r in v), 6), max_rel_l2_t1=max(r["rel_l2_t1"] for r in v),
                mean_experts_used=round(st.mean(r["experts_used"] for r in v), 2),
                mean_usage_entropy=round(st.mean(r["usage_entropy"] for r in v), 4)) for c, v in agg.items()}
json.dump(dict(runs=rows, by_config_2threads=summ), open(os.path.join(D, "summary.json"), "w"), indent=2)
for c, s in sorted(summ.items(), key=lambda kv: kv[1]["mean_rel_l2_t1"]):
    print(f"{c:26s} n={s['n']} mean={s['mean_rel_l2_t1']:.4f} max={s['max_rel_l2_t1']:.4f} used={s['mean_experts_used']} H={s['mean_usage_entropy']}")
print("total wall s (sum of runs):", round(sum(r["wall_s"] for r in rows), 1))
