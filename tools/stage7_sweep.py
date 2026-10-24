"""Stage 7: alpha x capacity_factor sweep on the stage-6 byte-level task.

Each config is one `train_bytes.py` run in its own subprocess (fixed step budget,
same seed, same held-out eval windows), `--workers` at a time with `--threads`
torch threads each. Capacity uses capacity_mode="sequence" (per-sequence,
position-ordered, causal-safe; see moe.py) unless --capacity-mode global is
given, in which case results are labelled leaky. A dropless row
(capacity_factor=None) is run per alpha as a reference.

Outputs: runs/stage7/<run>/{curve.csv,curve.jsonl,summary.json,samples.txt,console.txt}
and the aggregate runs/stage7/sweep.csv + runs/stage7/sweep.json.
  python3 tools/stage7_sweep.py            # run missing configs, then aggregate
  python3 tools/stage7_sweep.py --report   # aggregate only
"""
import argparse, csv, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALPHAS = [0.0, 0.01, 0.1, 0.3]
CFS = [0.8, 1.0, 1.25, 1.5, None]

# model/training settings shared by every config (smaller than the stage-6 default
# so 20 configs fit in ~30-45 min of CPU)
COMMON = ["--d-model", "128", "--n-heads", "4", "--d-ff", "256", "--experts", "4", "--layers", "3",
          "--top-k", "2", "--pos", "sinusoidal", "--seq-len", "128", "--batch", "32", "--lr", "2e-3",
          "--warmup", "100", "--eval-every", "250", "--eval-windows", "256", "--eval-batch", "64",
          "--seed", "567885", "--time-budget", "0", "--prompt", "It was a ", "--sample-len", "160"]


def run_name(alpha, cf, mode):
    cfs = "none" if cf is None else f"{cf:g}"
    return f"a{alpha:g}_cf{cfs}" + ("_GLOBAL_LEAKY" if mode == "global" and cf is not None else "")


def run_one(alpha, cf, a):
    name = run_name(alpha, cf, a.capacity_mode)
    out = ROOT / a.out_dir / name
    if (out / "summary.json").exists() and not a.force:
        return name, "cached"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "train_bytes.py"), *COMMON, "--max-steps", str(a.steps),
           "--threads", str(a.threads), "--aux-alpha", str(alpha),
           "--capacity-factor", "none" if cf is None else str(cf), "--capacity-mode", a.capacity_mode,
           "--out-dir", str(ROOT / a.out_dir), "--run-name", name]
    t0 = time.time()
    with open(out / "console.txt", "w") as f:
        f.write("$ " + " ".join(c if " " not in c else repr(c) for c in cmd[1:]) + "\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    msg = f"exit {r.returncode} in {time.time() - t0:.0f}s"
    print(f"[stage7] {name}: {msg}", flush=True)
    return name, msg


def aggregate(a):
    rows = []
    for alpha in ALPHAS:
        for cf in CFS:
            p = ROOT / a.out_dir / run_name(alpha, cf, a.capacity_mode) / "summary.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            rows.append(dict(run=s["run_name"], alpha=alpha, capacity_factor=cf if cf is not None else "none",
                             capacity_mode=s["capacity_mode"] if cf is not None else "dropless",
                             leaky=s["leaky"], steps=s["steps"], train_time_s=s["train_time_s"],
                             heldout_bpb=s["final_heldout_bpb"], heldout_loss_nats=s["final_heldout_loss_nats"],
                             best_heldout_bpb=s["best_heldout_bpb"],
                             train_drop_rate=s["train_drop_rate"], heldout_drop_rate=s["heldout_drop_rate"],
                             entropy_routed_mean=s["heldout_entropy_routed_mean"],
                             entropy_routed_per_layer=s["heldout_entropy_routed"],
                             entropy_kept_mean=s["heldout_entropy_kept_mean"],
                             final_aux=s["final_aux"], params=s["params"],
                             unigram_bpb=s["unigram_bpb_eval_windows"]))
    out = ROOT / a.out_dir
    (out / "sweep.json").write_text(json.dumps(dict(common_args=COMMON, steps=a.steps, threads=a.threads,
                                                    workers=a.workers, rows=rows), indent=2) + "\n")
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run"])
        w.writeheader()
        for r in rows:
            r = dict(r); r["entropy_routed_per_layer"] = " ".join(f"{e:.4f}" for e in r["entropy_routed_per_layer"])
            w.writerow(r)
    print(f"[stage7] aggregated {len(rows)} runs -> {out / 'sweep.csv'}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--capacity-mode", default="sequence", choices=["sequence", "global"])
    p.add_argument("--out-dir", default="runs/stage7")
    p.add_argument("--force", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if not a.report:
        jobs = [(al, cf) for cf in CFS for al in ALPHAS]
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda j: run_one(*j, a), jobs))
        print(f"[stage7] sweep wall time {time.time() - t0:.0f}s", flush=True)
    aggregate(a)


if __name__ == "__main__":
    main()
