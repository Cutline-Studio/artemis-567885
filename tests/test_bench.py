"""Stage 4: bench.py parity gate and output shape (tiny sizes, CPU)."""
import pytest
import torch
import bench
from moe import CapacitySparseMoELayer


def test_default_dispatch_is_still_loop():
    assert CapacitySparseMoELayer(8, 16, 4).dispatch == "loop"


@pytest.mark.parametrize("N,E,d", [(64, 4, 8), (200, 16, 8), (1024, 64, 16)])
def test_parity_at_bench_style_sizes(N, E, d):
    res = bench.parity(N, E, d, 4 * d, 2, 1.0, seed=N + E)
    for k, v in res.items():
        assert v["out_max_abs"] <= 1e-5 and v["xgrad_max_abs"] <= 1e-5, (k, v)
    assert res["f64_train"]["param_grad_max_abs"] <= 1e-10


def test_parity_with_capacity_drops():
    res = bench.parity(256, 8, 8, 32, 2, 0.25, seed=3)
    assert res["f64_train"]["dropped"] > 0  # capacity actually binds, and parity still held


def test_parity_gate_aborts_on_mismatch(monkeypatch):
    orig = CapacitySparseMoELayer._forward_vectorized

    def broken(self, x):
        out, aux, dropped = orig(self, x)
        return out * 1.001, aux, dropped
    monkeypatch.setattr(CapacitySparseMoELayer, "_forward_vectorized", broken)
    with pytest.raises(AssertionError):
        bench.parity(64, 4, 8, 32, 2, 1.0, seed=0)


def test_quick_run_rows(tmp_path):
    out = tmp_path / "b.json"
    rows = bench.main(["--quick", "--warmup", "1", "--reps", "3", "--json", str(out)])
    assert len(rows) == len(bench.QUICK_GRID) and out.exists()
    for r in rows:
        for mode in ("fwd", "fwd_bwd"):
            for path in ("loop", "vectorized"):
                s = r[mode][path]
                assert 0 < s["min_ms"] <= s["median_ms"] <= s["p90_ms"]
            assert r[mode]["speedup_median"] > 0
    assert torch.get_num_threads() == 1


def test_idle_expert_grad_none_vs_zero_is_a_known_difference():
    """MEASURED while writing stage 4: when an expert receives NO tokens, the loop path
    leaves its parameter .grad as None while the vectorized path (torch.stack over all
    experts) gives an all-zero .grad. Numerically equal, but optimizers treat them
    differently (Adam skips None params; a zero grad still applies momentum/weight decay).
    checks' compare() asserts identical grad sets; bench.compare_bench allows exactly this
    case, asserts the extra vectorized grads are exactly zero, and reports idle_experts."""
    a, b = bench.pair(64, 2, 1.0, 16, 64, torch.float64, 214)
    x = torch.randn(150, 16, generator=torch.Generator().manual_seed(215), dtype=torch.float64)
    for m in (a, b):
        m.eval(); out, aux, _ = m(x); (out.pow(2).sum() + aux).backward()
    idle = [i for i in range(64) if a.experts[i].w1.weight.grad is None]
    assert idle, "expected at least one expert with no routed tokens at N=150, E=64"
    for i in idle:
        assert b.experts[i].w1.weight.grad is not None
        assert torch.count_nonzero(b.experts[i].w1.weight.grad) == 0


def test_bench_parity_counts_idle_experts():
    res = bench.parity(150, 64, 16, 64, 2, 1.0, seed=214)
    assert res["f64_eval"]["idle_experts"] > 0
