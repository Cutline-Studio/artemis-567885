"""Summarize the language-benchmark runs into results.json and a markdown table."""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "runs"
ARMS = ["dense", "moe-none", "moe-learned", "moe-sinusoidal", "moe-fourier"]
SEEDS = [42, 1, 2]


def load(arm, seed):
    p = ROOT / arm / f"seed{seed}" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else float("nan"))


rows, per = [], {}
for arm in ARMS:
    runs = {s: load(arm, s) for s in SEEDS}
    runs = {s: r for s, r in runs.items() if r}
    if not runs:
        continue
    per[arm] = {s: {"val": r["best_validation"]["neural_bpb"], "test": r["test"]["neural_bpb"],
                    "params": r["configuration"]["parameters"],
                    "train_bytes": r["training_target_bytes"],
                    "seconds": r["total_elapsed_seconds"],
                    "test_drop": r["test"].get("dropped_assignment_fraction", 0.0),
                    "unigram_test": r["test"]["unigram_bpb"]} for s, r in runs.items()}
    v = [x["val"] for x in per[arm].values()]
    t = [x["test"] for x in per[arm].values()]
    p = next(iter(per[arm].values()))
    rows.append((arm, len(runs), p["params"], *ms(v), *ms(t),
                 st.mean(x["test_drop"] for x in per[arm].values()),
                 st.mean(x["seconds"] for x in per[arm].values())))


def paired(a, b, key="test"):
    common = sorted(set(per.get(a, {})) & set(per.get(b, {})))
    d = [per[a][s][key] - per[b][s][key] for s in common]
    return d


out = {"per_run": per, "paired": {}}
lines = ["| arm | seeds | params | val bpb (mean ± sd) | test bpb (mean ± sd) | test drop frac | s/run |",
         "|---|---|---|---|---|---|---|"]
for arm, n, params, vm, vs, tm, ts, drop, secs in rows:
    lines.append(f"| {arm} | {n} | {params:,} | {vm:.3f} ± {vs:.3f} | {tm:.3f} ± {ts:.3f} | {drop:.3f} | {secs:.0f} |")
lines.append("")
lines.append("Paired differences on test bpb (same seed, same data order; negative = first arm better):")
for a, b in [("moe-fourier", "moe-sinusoidal"), ("moe-fourier", "moe-learned"), ("moe-sinusoidal", "moe-learned"),
             ("moe-learned", "moe-none"), ("moe-learned", "dense"), ("moe-fourier", "dense")]:
    for key in ("val", "test"):
        d = paired(a, b, key)
        if d:
            out["paired"][f"{a} - {b} ({key})"] = d
            if key == "test":
                lines.append(f"- {a} − {b}: " + ", ".join(f"{x:+.3f}" for x in d)
                             + (f"  (mean {st.mean(d):+.3f})" if len(d) > 1 else ""))
Path(ROOT / "results.json").write_text(json.dumps(out, indent=2))
Path(ROOT / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
