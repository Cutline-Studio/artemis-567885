"""Stage 4: loop vs vectorized MoE dispatch benchmark (PLAN.md stage 4).

For EVERY config, parity is checked FIRST at that exact size (abort on failure),
using checks/test_check3_moe_dispatch.compare with identical weights and noise:
  * float64: routing, kept slots, drops, outputs, aux, input AND every parameter
    gradient allclose at rtol=atol=1e-12 (idle experts: loop grad None vs vectorized
    grad exactly zero is allowed, asserted to be exactly zero, and counted)
  * float32: routing, kept slots, drops, outputs, aux, input gradient allclose at
    rtol=1e-5, atol=1e-6; parameter-gradient max relative diff is REPORTED (not
    asserted) because batched-matmul reduction order differs from nn.Linear
    (see the strict xfail in checks/test_check3_moe_dispatch.py)
Only then are both paths timed in float32 on the same weights and the same input:
  * fwd      : eval mode, torch.no_grad(), forward only
  * fwd_bwd  : train mode (routing noise on), forward + (out^2).mean()+aux backward
WARMUP untimed calls, then REPS timed calls; loop and vectorized are interleaved
per repeat (order alternates) so slow drift on the host hits both paths equally.
Reports median and p90 (statistics.quantiles, n=10, 'inclusive') in ms.
Defaults never change: CapacitySparseMoELayer's dispatch default stays "loop".

Usage:
  python3 bench.py                         # full grid, 1 thread, writes JSON lines to stdout
  python3 bench.py --threads 8 --json out.json
  python3 bench.py --quick                 # tiny grid (used by tests)
"""
import argparse, json, os, platform, statistics, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from checks.test_check3_moe_dispatch import TOL, pair, run

FULL_GRID = [(N, E, d) for N in (1024, 4096, 16384) for E in (4, 16, 64) for d in (64, 256)]
QUICK_GRID = [(128, 4, 16), (256, 16, 16)]


def cpu_model():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def env_info(args):
    return dict(python=sys.version.split()[0], torch=torch.__version__, cpu=cpu_model(),
                cpu_count=os.cpu_count(), threads=torch.get_num_threads(),
                cuda_available=torch.cuda.is_available(), platform=platform.platform(),
                top_k=args.top_k, capacity_factor=args.cf, d_ff_mult=args.d_ff_mult,
                warmup=args.warmup, reps=args.reps, seed=args.seed, dtype_timed="float32")


def summarize(ts):
    q = statistics.quantiles(ts, n=10, method="inclusive")
    return dict(median_ms=round(statistics.median(ts), 3), p90_ms=round(q[8], 3),
                min_ms=round(min(ts), 3))


def step(m, x, mode, noise_seed):
    m.zero_grad(set_to_none=True)
    torch.manual_seed(noise_seed)
    t0 = time.perf_counter()
    if mode == "fwd":
        m.eval()
        with torch.no_grad():
            m(x)
    else:
        m.train()
        out, aux, _ = m(x)
        (out.pow(2).mean() + aux).backward()
    return (time.perf_counter() - t0) * 1000.0


def compare_bench(E, K, cf, N, d, d_ff, dtype, train, seed, assert_param_grads=True):
    """Same assertions as checks/test_check3_moe_dispatch.compare, except one measured,
    documented difference is allowed and COUNTED instead of aborting: an expert that
    receives no tokens has .grad None on the loop path but an all-zero .grad on the
    vectorized path (see tests/test_bench.py). Those vectorized grads must be exactly 0."""
    a, b = pair(E, K, cf, d, d_ff, dtype, seed)
    x = torch.randn(N, d, generator=torch.Generator().manual_seed(seed + 1), dtype=dtype)
    ra, rb = run(a, x, train, 77), run(b, x, train, 77)
    tol = TOL[dtype]
    assert torch.equal(ra[3]["topk_indices"], rb[3]["topk_indices"]), "routing differs"
    assert torch.equal(ra[3]["kept_slots"], rb[3]["kept_slots"]), "kept slots differ"
    assert ra[2] == rb[2] and ra[3]["dropped_tokens"] == rb[3]["dropped_tokens"], "drops differ"
    assert torch.allclose(ra[0], rb[0], **tol), "outputs differ"
    assert torch.allclose(ra[1], rb[1], **tol), "aux differs"
    assert torch.allclose(ra[4], rb[4], **tol), "input grads differ"
    assert set(ra[5]) <= set(rb[5]), "loop has grads the vectorized path lacks"
    extra = sorted(set(rb[5]) - set(ra[5]))
    for n in extra:
        assert torch.count_nonzero(rb[5][n]) == 0, f"{n}: vectorized grad nonzero for an idle expert"
    within = all(torch.allclose(ra[5][n], rb[5][n], **tol) for n in ra[5])
    if assert_param_grads:
        assert within, "parameter grads differ"
    return dict(out_max_abs=(ra[0] - rb[0]).abs().max().item(),
                aux_abs=(ra[1] - rb[1]).abs().item(),
                xgrad_max_abs=(ra[4] - rb[4]).abs().max().item(),
                param_grad_max_abs=max((ra[5][n] - rb[5][n]).abs().max().item() for n in ra[5]),
                param_grad_rel=max((ra[5][n] - rb[5][n]).abs().max().item()
                                   / max(ra[5][n].abs().max().item(), 1e-30) for n in ra[5]),
                param_grads_within_tol=within, dropped=ra[2],
                idle_experts=len({n.split(".")[1] for n in extra}))


def parity(N, E, d, d_ff, K, cf, seed):
    res = {}
    for train in (False, True):
        tag = "train" if train else "eval"
        res[f"f64_{tag}"] = compare_bench(E, K, cf, N, d, d_ff, torch.float64, train, seed)
        res[f"f32_{tag}"] = compare_bench(E, K, cf, N, d, d_ff, torch.float32, train, seed,
                                          assert_param_grads=False)
    return res


def bench_config(N, E, d, args):
    d_ff = args.d_ff_mult * d
    seed = args.seed + E * 1000 + d  # deterministic per config
    par = parity(N, E, d, d_ff, args.top_k, args.cf, seed)  # raises AssertionError on mismatch
    loop, vec = pair(E, args.top_k, args.cf, d, d_ff, torch.float32, seed)
    x = torch.randn(N, d, generator=torch.Generator().manual_seed(seed + 1))
    row = dict(N=N, E=E, d=d, d_ff=d_ff, parity=par)
    for mode in ("fwd", "fwd_bwd"):
        ts = {"loop": [], "vectorized": []}
        for i in range(args.warmup + args.reps):
            order = (("loop", loop), ("vectorized", vec)) if i % 2 == 0 else (("vectorized", vec), ("loop", loop))
            for name, m in order:
                dt = step(m, x, mode, noise_seed=args.seed + i)
                if i >= args.warmup:
                    ts[name].append(dt)
        sl, sv = summarize(ts["loop"]), summarize(ts["vectorized"])
        row[mode] = dict(loop=sl, vectorized=sv,
                         speedup_median=round(sl["median_ms"] / sv["median_ms"], 3))
    return row


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--seed", type=int, default=567885)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--cf", type=float, default=1.0)
    p.add_argument("--d-ff-mult", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--json", default=None, help="also write all rows to this JSON file")
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    env = env_info(args)
    print("ENV", json.dumps(env), flush=True)
    rows = []
    for N, E, d in (QUICK_GRID if args.quick else FULL_GRID):
        row = bench_config(N, E, d, args)
        rows.append(row)
        print("ROW", json.dumps(row), flush=True)
    print("\n| N | E | d | d_ff | fwd loop ms (med/p90) | fwd vec ms (med/p90) | fwd speedup | "
          "fwd+bwd loop ms (med/p90) | fwd+bwd vec ms (med/p90) | fwd+bwd speedup |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f, b = r["fwd"], r["fwd_bwd"]
        print(f"| {r['N']} | {r['E']} | {r['d']} | {r['d_ff']} | "
              f"{f['loop']['median_ms']}/{f['loop']['p90_ms']} | {f['vectorized']['median_ms']}/{f['vectorized']['p90_ms']} | "
              f"{f['speedup_median']}x | {b['loop']['median_ms']}/{b['loop']['p90_ms']} | "
              f"{b['vectorized']['median_ms']}/{b['vectorized']['p90_ms']} | {b['speedup_median']}x |")
    print("PARITY_ALL_OK", len(rows), "configs", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(env=env, argv=sys.argv, rows=rows), fh, indent=1)
    return rows


if __name__ == "__main__":
    main()
