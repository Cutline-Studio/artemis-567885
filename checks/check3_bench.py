"""Check 3 timing. For every benchmark config, FIRST re-run the equivalence
comparison from checks/test_check3_moe_dispatch.py at the benchmark size; abort
if any assertion fails:
  * float64: routing, kept slots, drops, outputs, aux, input AND all parameter
    gradients within rtol=1e-12, atol=1e-12
  * float32: routing, kept slots, drops, outputs, aux, input AND all parameter
    gradients within rtol=1e-5, atol=1e-6 (parameter gradients were printed but
    not asserted until the vectorized path reduced expert-weight gradients in the
    loop's row order; see moe._BatchedExpertFFN).
Only then time loop vs vectorized (median of REPS after WARMUP, CPU, 1 thread).

Usage: python3 checks/check3_bench.py
"""
import json, os, platform, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from checks.test_check3_moe_dispatch import compare, pair

torch.set_num_threads(1)
N, D, DFF, K, CF = 4096, 128, 512, 2, 1.0
WARMUP, REPS = 3, 15
print("ENV", json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, threads=torch.get_num_threads(),
                             cpu=platform.processor() or platform.machine(), N=N, d=D, d_ff=DFF, top_k=K, cf=CF,
                             warmup=WARMUP, reps=REPS)), flush=True)
for E in (4, 16, 64):
    for train in (False, True):
        r64 = compare(E, K, CF, N, D, DFF, torch.float64, train, seed=E)
        print("EQUIV_OK_FLOAT64", json.dumps(dict(E=E, train=train, **r64)), flush=True)
        r32 = compare(E, K, CF, N, D, DFF, torch.float32, train, seed=E)
        print("EQUIV_OK_FLOAT32", json.dumps(dict(E=E, train=train, **r32)), flush=True)
    a, b = pair(E, K, CF, D, DFF, torch.float32, E)
    x = torch.randn(N, D, generator=torch.Generator().manual_seed(1))
    row = dict(E=E)
    for name, m in (("loop", a), ("vectorized", b)):
        for mode in ("fwd_eval", "fwd_bwd_train"):
            ts = []
            for i in range(WARMUP + REPS):
                m.zero_grad(set_to_none=True)
                torch.manual_seed(i)
                t0 = time.perf_counter()
                if mode == "fwd_eval":
                    m.eval()
                    with torch.no_grad():
                        m(x)
                else:
                    m.train()
                    out, aux, _ = m(x)
                    (out.pow(2).mean() + aux).backward()
                dt = (time.perf_counter() - t0) * 1000
                if i >= WARMUP:
                    ts.append(dt)
            row[f"{name}_{mode}_median_ms"] = round(statistics.median(ts), 3)
            row[f"{name}_{mode}_min_ms"] = round(min(ts), 3)
    print("TIMING", json.dumps(row), flush=True)
