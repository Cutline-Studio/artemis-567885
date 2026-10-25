"""Check 3: loop vs vectorized MoE dispatch with IDENTICAL weights (one state_dict
copied into both), identical routing noise (RNG reset before each call), same
capacity. Asserts, in this order: routing assignments (top-k indices) equal,
kept/dropped slots equal, dropped counts equal, aux loss, outputs and ALL gradients
(input, gate, every expert param) allclose. Tolerances are stated in TOL and are
not loosened per case. Timing lives in checks/check3_bench.py and runs only after
these assertions pass."""
import pytest
import torch
from moe import CapacitySparseMoELayer

TOL = {torch.float32: dict(rtol=1e-5, atol=1e-6), torch.float64: dict(rtol=1e-12, atol=1e-12)}
CONFIGS = [  # (E, top_k, capacity_factor, N, d, d_ff)
    (4, 1, 1.0, 256, 16, 32), (4, 2, 1.0, 256, 16, 32), (4, 2, 0.5, 256, 16, 32),
    (8, 2, 0.25, 512, 16, 32), (16, 2, 1.0, 1024, 32, 64), (64, 2, 1.0, 1024, 32, 64),
    (4, 2, None, 256, 16, 32), (8, 3, 0.75, 300, 8, 16),
]


def pair(E, K, cf, d, d_ff, dtype, seed):
    torch.manual_seed(seed)
    a = CapacitySparseMoELayer(d, d_ff, E, K, cf, dispatch="loop").to(dtype)
    b = CapacitySparseMoELayer(d, d_ff, E, K, cf, dispatch="vectorized").to(dtype)
    b.load_state_dict(a.state_dict())
    return a, b


def run(m, x, train, noise_seed):
    m.train(train)
    xi = x.clone().requires_grad_(True)
    m.zero_grad(set_to_none=True)
    torch.manual_seed(noise_seed)
    out, aux, dropped = m(xi)
    (out.pow(2).sum() + aux).backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return out.detach(), aux.detach(), dropped, dict(m.last_stats), xi.grad.clone(), grads


def compare(E, K, cf, N, d, d_ff, dtype, train, seed=0, assert_param_grads=True):
    a, b = pair(E, K, cf, d, d_ff, dtype, seed)
    x = torch.randn(N, d, generator=torch.Generator().manual_seed(seed + 1), dtype=dtype)
    ra, rb = run(a, x, train, 77), run(b, x, train, 77)
    tol = TOL[dtype]
    res = {}
    res["topk_equal"] = torch.equal(ra[3]["topk_indices"], rb[3]["topk_indices"])
    res["kept_equal"] = torch.equal(ra[3]["kept_slots"], rb[3]["kept_slots"])
    res["dropped"] = (ra[2], rb[2])
    res["dropped_tokens"] = (ra[3]["dropped_tokens"], rb[3]["dropped_tokens"])
    res["out_max_abs"] = (ra[0] - rb[0]).abs().max().item()
    res["aux_abs"] = (ra[1] - rb[1]).abs().item()
    res["xgrad_max_abs"] = (ra[4] - rb[4]).abs().max().item()
    res["param_grad_max_abs"] = max((ra[5][n] - rb[5][n]).abs().max().item() for n in ra[5])
    assert set(ra[5]) == set(rb[5])
    assert res["topk_equal"] and res["kept_equal"]
    assert ra[2] == rb[2] and ra[3]["dropped_tokens"] == rb[3]["dropped_tokens"]
    assert torch.allclose(ra[0], rb[0], **tol)
    assert torch.allclose(ra[1], rb[1], **tol)
    assert torch.allclose(ra[4], rb[4], **tol)
    res["param_grads_within_tol"] = all(torch.allclose(ra[5][n], rb[5][n], **tol) for n in ra[5])
    res["param_grad_max_abs_over_max_grad"] = max(
        (ra[5][n] - rb[5][n]).abs().max().item() / max(ra[5][n].abs().max().item(), 1e-30) for n in ra[5])
    if assert_param_grads:
        for n in ra[5]:
            assert torch.allclose(ra[5][n], rb[5][n], **tol), n
    return res


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("cfg", CONFIGS)
def test_loop_vs_vectorized(cfg, train, dtype):
    res = compare(*cfg, dtype, train)
    print(f"E={cfg[0]} K={cfg[1]} cf={cfg[2]} N={cfg[3]} {str(dtype)[6:]} train={train}: {res}")


def test_drops_actually_happen_in_capacity_configs():
    a, _ = pair(8, 2, 0.25, 16, 32, torch.float32, 0)
    a.eval(); a(torch.randn(512, 16))
    assert a.last_stats["dropped_assignments"] > 0 and a.last_stats["dropped_tokens"] > 0


def test_vectorized_rejects_legacy():
    with pytest.raises(ValueError):
        CapacitySparseMoELayer(8, 16, 4, 2, 1.0, legacy=True, dispatch="vectorized")


def test_legacy_matches_fixed_only_without_drops():
    """Legacy semantics differ by design (token-order capacity, sum==0 passthrough,
    top-1-only aux). With dropless capacity the OUTPUTS coincide; with drops they don't."""
    torch.manual_seed(0)
    leg = CapacitySparseMoELayer(16, 32, 4, 2, 4.0, legacy=True).eval()
    fix = CapacitySparseMoELayer(16, 32, 4, 2, 4.0, dispatch="vectorized").eval()
    fix.load_state_dict(leg.state_dict())
    x = torch.randn(256, 16, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(leg(x)[0], fix(x)[0], rtol=1e-5, atol=1e-6)
    leg.capacity_factor = fix.capacity_factor = 0.25
    assert not torch.allclose(leg(x)[0], fix(x)[0], rtol=1e-5, atol=1e-6)


def test_bench_size_float32_param_grads_at_stated_tol():
    """Formerly a strict xfail: the vectorized path reduced expert-weight gradients
    over a padded [E, cap, d] buffer in priority order, so float32 rounding drifted
    up to ~3e-5 from the loop. Weight/bias grads are now reduced per expert over
    exactly the kept rows in the loop's order (moe._BatchedExpertFFN), and they
    must meet the unchanged float32 tolerance at the benchmark size."""
    compare(4, 2, 1.0, 4096, 128, 512, torch.float32, True, seed=4)


@pytest.mark.parametrize("E", [4, 16, 64])
def test_bench_size_float32_all_grads(E):
    compare(E, 2, 1.0, 4096, 128, 512, torch.float32, True, seed=E)


@pytest.mark.parametrize("E", [4, 16, 64])
def test_bench_size_float64(E):
    compare(E, 2, 1.0, 4096, 128, 512, torch.float64, True, seed=E)


@pytest.mark.parametrize("E", [4, 16, 64])
def test_bench_size_float32_routing_outputs_input_grads(E):
    res = compare(E, 2, 1.0, 4096, 128, 512, torch.float32, True, seed=E, assert_param_grads=False)
    print(f"E={E} float32 N=4096: {res}")


def test_vectorized_second_order_grads_match_loop():
    """The vectorized expert FFN uses a custom autograd.Function; its backward is
    built from differentiable ops, so create_graph / double backward must still
    work and agree with the loop path (float64, no drops)."""
    a, b = pair(4, 2, None, 8, 16, torch.float64, 0)
    x = torch.randn(64, 8, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    res = []
    for m in (a, b):
        m.eval()
        xi = x.clone().requires_grad_(True)
        out = m(xi)[0]
        g = torch.autograd.grad(out.pow(2).sum(), xi, create_graph=True)[0]
        m.zero_grad(set_to_none=True)
        g.pow(2).sum().backward()
        res.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
    assert set(res[0]) == set(res[1])
    for n in res[0]:
        assert torch.allclose(res[0][n], res[1][n], **TOL[torch.float64]), n
    assert torch.autograd.gradgradcheck(lambda t: b(t)[0], (x[:16].clone().requires_grad_(True),))
