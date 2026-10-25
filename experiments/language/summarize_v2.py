"""Summarize protocol-v2 runs (runs_v2/) into results_v2.json and results_v2.md.

Bits per byte and routing diagnostics are reported in separate tables. Old-protocol runs are not read.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_v2"
PROTO = json.loads((ROOT / "protocol_v2.json").read_text())
ARMS = list(PROTO["arm_args"])
SEEDS = PROTO["seeds"]


def ledger():
    return [json.loads(l) for l in (RUNS / "ledger.jsonl").read_text().splitlines()]


def load(arm, seed):
    d = RUNS / arm / f"seed{seed}"
    if not (d / "metrics.json").exists():
        return None
    return {"metrics": json.loads((d / "metrics.json").read_text()),
            "val": [json.loads(l) for l in (d / "validation.jsonl").read_text().splitlines()]}


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


runs = {arm: {s: load(arm, s) for s in SEEDS} for arm in ARMS}
per, drops = {}, {}
for arm in ARMS:
    for s, r in runs[arm].items():
        if not r:
            continue
        m, val = r["metrics"], r["val"]
        sel = m["best_validation"]
        train_drops = sum((v["train_since_last_validation"]["routing"] or {}).get("dropped_assignments", 0)
                          for v in val)
        eval_drops = sum(v.get("dropped_assignments", 0) for v in val) + m["test"].get("dropped_assignments", 0)
        drops[(arm, s)] = (train_drops, eval_drops)
        last_train = val[-1]["train_since_last_validation"]["routing"]
        per[(arm, s)] = {
            "val_bpb": sel["neural_bpb"], "test_bpb": m["test"]["neural_bpb"],
            "unigram_val_bpb": sel["unigram_bpb"], "unigram_test_bpb": m["test"]["unigram_bpb"],
            "selected_step": sel["optimizer_steps"], "selected_train_bytes": sel["training_target_bytes"],
            "train_bytes_total": m["training_target_bytes"], "optimizer_steps": m["optimizer_steps"],
            "training_seconds": m["training_seconds"], "total_seconds": m["total_elapsed_seconds"],
            "params_total": m["configuration"]["parameter_counts"]["total"],
            "params_active": m["configuration"]["parameter_counts"]["active_per_byte"],
            "val_curve": [(v["optimizer_steps"], v["neural_bpb"]) for v in val],
            "routing_selected_val": sel.get("routing"),
            "routing_test": m["test"].get("routing"),
            "routing_train_last_window": last_train,
            "dropped_train": train_drops, "dropped_eval": eval_drops,
            "checkpoint_sha256": m["checkpoint_sha256"],
        }

L = []
L.append("# Artemis 567885 language benchmark, protocol v2 (development benchmark)\n")
L.append(f"Code commit `{PROTO['code_commit']}` · corpus manifest sha256 `{PROTO['corpus_manifest_sha256']}` · "
         f"protocol sha256 `{(ROOT / 'protocol_v2.sha256').read_text().split()[0]}`\n")
L.append("Budgets are matched in optimizer steps (5,000) and byte positions processed, not in compute; "
         "wall time is reported next to every result. This corpus is a development benchmark: its test scores "
         "have been viewed, so confirmation belongs to the untouched confirmation set.\n")
L.append("## Bits per byte (cross-entropy only; aux loss excluded)\n")
L.append("| arm | seed | val bpb | test bpb | selected step | train bytes (selected / total) | train s | total s |")
L.append("|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    for s in SEEDS:
        p = per.get((arm, s))
        if not p:
            L.append(f"| {arm} | {s} | missing | | | | | |")
            continue
        L.append(f"| {arm} | {s} | {fmt(p['val_bpb'])} | {fmt(p['test_bpb'])} | {p['selected_step']} | "
                 f"{p['selected_train_bytes']:,} / {p['train_bytes_total']:,} | {p['training_seconds']:.0f} | {p['total_seconds']:.0f} |")
any_p = next(iter(per.values()))
L.append(f"\nUnigram (train counts, add-one): val {fmt(any_p['unigram_val_bpb'])}, test {fmt(any_p['unigram_test_bpb'])}.\n")

L.append("Active = shared parameters plus selected-expert parameters per byte (embedding table, position module, "
         "attention, norms, router gates and head all counted; 2 of 4 experts per MoE layer). An accounting "
         "convention, not measured FLOPs or memory traffic. Dense: active = total.\n")
L.append("| arm | params total | shared + selected-expert params per byte | val bpb mean (sd) | test bpb mean (sd) | mean train s |")
L.append("|---|---|---|---|---|---|")
for arm in ARMS:
    ps = [per[(arm, s)] for s in SEEDS if (arm, s) in per]
    if not ps:
        continue
    v = [p["val_bpb"] for p in ps]; t = [p["test_bpb"] for p in ps]
    sd = lambda xs: st.stdev(xs) if len(xs) > 1 else float("nan")
    L.append(f"| {arm} | {ps[0]['params_total']:,} | {ps[0]['params_active']:,} | "
             f"{st.mean(v):.3f} ({sd(v):.3f}) | {st.mean(t):.3f} ({sd(t):.3f}) | "
             f"{st.mean(p['training_seconds'] for p in ps):.0f} |")

L.append("\n## Paired differences by seed (first arm minus second; negative = first arm lower bpb)\n")
pairs = [("moe-fourier", "moe-sinusoidal"), ("moe-fourier", "moe-learned"), ("moe-sinusoidal", "moe-learned"),
         ("moe-learned", "dense"), ("moe-sinusoidal", "dense"), ("moe-fourier", "dense")]
paired = {}
L.append("| comparison | split | seed 42 | seed 1 | seed 2 | mean | min | max |")
L.append("|---|---|---|---|---|---|---|---|")
for a, b in pairs:
    for split in ("val", "test"):
        d = {s: per[(a, s)][f"{split}_bpb"] - per[(b, s)][f"{split}_bpb"]
             for s in SEEDS if (a, s) in per and (b, s) in per}
        if not d:
            continue
        paired[f"{a} - {b} ({split})"] = d
        vals = list(d.values())
        cells = " | ".join(f"{d[s]:+.3f}" if s in d else "—" for s in SEEDS)
        L.append(f"| {a} − {b} | {split} | {cells} | {st.mean(vals):+.3f} | {min(vals):+.3f} | {max(vals):+.3f} |")
L.append("\nOnly moe-fourier − moe-sinusoidal is a parameter-matched comparison (same feature width, same projection, "
         "identical parameter counts). Comparisons with moe-learned (larger: 128-row position table) and with dense "
         "(different architecture and parameter count) are qualified accordingly.")
L.append("\nThree seeds: preliminary evidence. The sign agreement across seeds and the range are reported, not a "
         "confidence interval. Development benchmark: test scores of this corpus have been viewed.")

L.append("\n## Routing diagnostics (separate from bpb)\n")
L.append("Load = fraction of top-2 assignments of real (non-padding) bytes per expert. Entropy normalized by log 4. "
         "Aux = Switch-style balance loss summed over both layers (1.0 per layer at perfect balance, so 2.0 total).\n")
L.append("| arm | seed | layer | val load (selected ckpt) | val entropy | test entropy | train entropy, last 1000 steps | train aux, last 1000 steps |")
L.append("|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    if arm == "dense":
        continue
    for s in SEEDS:
        p = per.get((arm, s))
        if not p or not p["routing_selected_val"]:
            continue
        rv, rt, rtr = p["routing_selected_val"], p["routing_test"], p["routing_train_last_window"]
        for i in range(len(rv["per_layer"])):
            L.append(f"| {arm} | {s} | {i} | {', '.join(f'{x:.2f}' for x in rv['per_layer'][i]['load_fraction'])} | "
                     f"{rv['per_layer'][i]['normalized_entropy']:.3f} | {rt['per_layer'][i]['normalized_entropy']:.3f} | "
                     f"{rtr['per_layer'][i]['normalized_entropy']:.3f} | "
                     f"{rtr['mean_aux_loss_summed_over_layers']:.3f} |" if i == 0 else
                     f"| | | {i} | {', '.join(f'{x:.2f}' for x in rv['per_layer'][i]['load_fraction'])} | "
                     f"{rv['per_layer'][i]['normalized_entropy']:.3f} | {rt['per_layer'][i]['normalized_entropy']:.3f} | "
                     f"{rtr['per_layer'][i]['normalized_entropy']:.3f} | |")

tr = sum(x[0] for x in drops.values()); ev = sum(x[1] for x in drops.values())
L.append(f"\nDropped assignments across all MoE runs: training {tr}, evaluation {ev} "
         f"({'zero in both, as required for dropless routing' if tr == 0 and ev == 0 else 'NONZERO, investigate'}).")

led = ledger()
status = [e for e in led if "arm" in e and e["status"] != "launched"]
L.append("\n## Run ledger\n")
for e in status:
    extra = (f" (exit {e.get('exit_code')}, {e.get('wall_seconds')} s)" if "exit_code" in e
             else f" — {e.get('detail') or e.get('reasons', '')}")
    L.append(f"- {e['utc'][:16]}Z {e['arm']} seed {e['seed']}: {e['status']}{extra}")

out = {"protocol_sha256_file": (ROOT / "protocol_v2.sha256").read_text().split()[0],
       "code_commit": PROTO["code_commit"], "corpus_manifest_sha256": PROTO["corpus_manifest_sha256"],
       "per_run": {f"{a}/seed{s}": v for (a, s), v in per.items()},
       "paired_differences": paired, "dropped_assignments": {"training": tr, "evaluation": ev},
       "ledger": led}
(ROOT / "results_v2.json").write_text(json.dumps(out, indent=2))
(ROOT / "results_v2.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
