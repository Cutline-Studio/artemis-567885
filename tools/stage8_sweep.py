"""Stage 8: routing comparison on the stage-6/7 byte-level task.

Same model/training settings as the stage-7 sweep (1.06M params, 2000 steps, seed
567885, same 256 held-out Austen windows), dropless token-choice routing with
aux alpha 0.01 unless noted. Compares the baseline (top-2, renormalised top-k
softmax weights) against Switch-style weighting (full router probability; the
gate gets task gradient at top-1), router z-loss, and expert-choice routing.
Expert choice is NOT causal (an expert's top-C pick within a sequence looks at
future positions), so its rows are labelled LEAKY and are not comparable as a
causal byte-LM result.

Each config runs `train_bytes.py` in its own subprocess, `--workers` at a time with
`--threads` torch threads each (default 2 x 2 = 4 threads).
Outputs: runs/stage8/<run>/{curve.csv,curve.jsonl,summary.json,samples.txt,console.txt},
runs/stage8/sweep.csv, runs/stage8/sweep.json.
  python3 tools/stage8_sweep.py            # run missing configs, then aggregate
  python3 tools/stage8_sweep.py --report   # aggregate only
"""
import argparse, csv, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from stage7_sweep import COMMON as STAGE7_COMMON  # noqa: E402

# stage-7 settings minus --top-k (set per config)
_i = STAGE7_COMMON.index("--top-k")
COMMON = STAGE7_COMMON[:_i] + STAGE7_COMMON[_i + 2:] + ["--aux-alpha", "0.01"]

# name -> extra flags
CONFIGS = {
    "top2_renorm": ["--top-k", "2"],
    "top1_renorm": ["--top-k", "1"],
    "top1_switch": ["--top-k", "1", "--gate-weighting", "switch"],
    "top2_switch": ["--top-k", "2", "--gate-weighting", "switch"],
    "top2_renorm_z1e-3": ["--top-k", "2", "--z-loss", "1e-3"],
    "top2_renorm_z1e-2": ["--top-k", "2", "--z-loss", "1e-2"],
    "top1_switch_z1e-3": ["--top-k", "1", "--gate-weighting", "switch", "--z-loss", "1e-3"],
    "ec_k2_cf1_seq_LEAKY": ["--top-k", "2", "--routing", "expert_choice", "--capacity-factor", "1.0",
                            "--capacity-mode", "sequence"],
}


def run_one(name, a):
    out = ROOT / a.out_dir / name
    if (out / "summary.json").exists() and not a.force:
        return name, "cached"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "train_bytes.py"), *COMMON, *CONFIGS[name], "--max-steps", str(a.steps),
           "--threads", str(a.threads), "--out-dir", str(ROOT / a.out_dir), "--run-name", name]
    t0 = time.time()
    with open(out / "console.txt", "w") as f:
        f.write("$ " + " ".join(c if " " not in c else repr(c) for c in cmd[1:]) + "\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    msg = f"exit {r.returncode} in {time.time() - t0:.0f}s"
    print(f"[stage8] {name}: {msg}", flush=True)
    return name, msg


def aggregate(a):
    rows = []
    for name in CONFIGS:
        p = ROOT / a.out_dir / name / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        rows.append(dict(run=name, top_k=s["top_k"], gate_weighting=s["gate_weighting"], routing=s["routing"],
                         z_loss_coef=s["z_loss_coef"], capacity_factor=s["capacity_factor"] or "none",
                         leaky=s["leaky"], steps=s["steps"], train_time_s=s["train_time_s"],
                         heldout_bpb=s["final_heldout_bpb"], heldout_loss_nats=s["final_heldout_loss_nats"],
                         best_heldout_bpb=s["best_heldout_bpb"], heldout_drop_rate=s["heldout_drop_rate"],
                         entropy_routed_mean=s["heldout_entropy_routed_mean"],
                         entropy_routed_per_layer=s["heldout_entropy_routed"],
                         final_aux=s["final_aux"], train_z_mean=s["train_z_mean"],
                         heldout_z_loss=s["heldout_z_loss"], params=s["params"],
                         unigram_bpb=s["unigram_bpb_eval_windows"]))
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(dict(common_args=COMMON, configs=CONFIGS, steps=a.steps,
                                                    threads=a.threads, workers=a.workers, rows=rows),
                                               indent=2) + "\n")
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run"])
        w.writeheader()
        for r in rows:
            r = dict(r); r["entropy_routed_per_layer"] = " ".join(f"{e:.4f}" for e in r["entropy_routed_per_layer"])
            w.writerow(r)
    print(f"[stage8] aggregated {len(rows)} runs -> {out / 'sweep.csv'}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out-dir", default="runs/stage8")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if not a.report:
        names = a.only or list(CONFIGS)
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda n: run_one(n, a), names))
        print(f"[stage8] sweep wall time {time.time() - t0:.0f}s", flush=True)
    aggregate(a)


if __name__ == "__main__":
    main()
