"""Check 5: causal attention integrated into the existing model
(MultiHeadAttention(causal=True) -> MoETransformerBlock -> DynamicMultiBlockTransformer)
and future-input independence through attention AND MoE routing together.

Protocol: x of shape [B, L, d]; for each cut t, replace x[:, t+1:] with fresh
random values; outputs y[:, :t+1] must be bit-identical (torch.equal), and the
Jacobian of y[:, :t+1] w.r.t. x[:, t+1:] must be exactly zero. In train mode the
routing-noise RNG is reset before each forward, so the noise is the same draw.
Capacity-limited routing is EXPECTED to leak (capacity is shared over all B*L
tokens and ranked by router probability), so those configs are measured and the
leak is asserted to exist (a documented finding), not hidden."""
import json
import pytest
import torch
from transformer import MultiHeadAttention
from model import DynamicMultiBlockTransformer

B, L, D = 4, 24, 32
CUTS = [0, 5, 11, 17, 22]


def perturbed(x, t, seed):
    x2 = x.clone()
    x2[:, t + 1:] = torch.randn(x[:, t + 1:].shape, generator=torch.Generator().manual_seed(seed))
    return x2


def fwd(m, x, train):
    m.train(train)
    torch.manual_seed(1234)
    out = m(x)
    return out[0] if isinstance(out, tuple) else out


def leak_stats(m, train, trials=3):
    x = torch.randn(B, L, D, generator=torch.Generator().manual_seed(0))
    worst, leaks, n = 0.0, 0, 0
    with torch.no_grad():
        y = fwd(m, x, train)
        for t in CUTS:
            for s in range(trials):
                y2 = fwd(m, perturbed(x, t, 100 * t + s), train)
                d = (y[:, :t + 1] - y2[:, :t + 1]).abs().max().item()
                worst = max(worst, d); leaks += int(d > 0); n += 1
    return dict(max_abs_change_past=worst, leaking_trials=leaks, trials=n)


def cross_sample_leak(m):
    """perturb ONLY sequence 1 (all positions); does sequence 0 change?"""
    x = torch.randn(B, L, D, generator=torch.Generator().manual_seed(0))
    x2 = x.clone(); x2[1] = torch.randn(L, D, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        return (fwd(m, x, False)[0] - fwd(m, x2, False)[0]).abs().max().item()


def jac_past_wrt_future(m, t, train):
    x = torch.randn(B, L, D, generator=torch.Generator().manual_seed(0), requires_grad=True)
    y = fwd(m, x, train)
    g, = torch.autograd.grad(y[:, :t + 1].sum(), x)
    return g[:, t + 1:].abs().max().item()


def make(causal=True, cf=None, legacy=False, dispatch="loop", layers=2, E=4, K=2):
    torch.manual_seed(0)
    return DynamicMultiBlockTransformer(D, 4, 64, E, num_layers=layers, top_k=K, capacity_factor=cf,
                                        legacy=legacy, causal=causal, dispatch=dispatch)


def test_attention_alone_is_causal_and_original_is_not():
    torch.manual_seed(0)
    a = MultiHeadAttention(D, 4, causal=True); u = MultiHeadAttention(D, 4, causal=False)
    u.load_state_dict(a.state_dict())
    x = torch.randn(B, L, D)
    for t in CUTS:
        x2 = perturbed(x, t, t)
        assert torch.equal(a(x)[:, :t + 1], a(x2)[:, :t + 1])
    d = (u(x)[:, :1] - u(perturbed(x, 0, 0))[:, :1]).abs().max().item()
    print("non-causal (original) attention: max change at position 0 =", d)
    assert d > 0


@pytest.mark.parametrize("dispatch", ["loop", "vectorized"])
@pytest.mark.parametrize("train", [False, True])
def test_causal_dropless_moe_has_no_leak(dispatch, train):
    m = make(cf=None, dispatch=dispatch)
    st = leak_stats(m, train)
    jac = max(jac_past_wrt_future(m, t, train) for t in CUTS)
    cs = cross_sample_leak(m)
    print(f"CAUSAL dropless dispatch={dispatch} train={train}: {json.dumps(st)} max|dPast/dFuture|={jac} cross_sample={cs}")
    assert st["leaking_trials"] == 0 and st["max_abs_change_past"] == 0.0
    assert jac == 0.0 and cs == 0.0


@pytest.mark.parametrize("cf,legacy,dispatch", [(1.0, False, "loop"), (1.0, False, "vectorized"),
                                                (0.5, False, "loop"), (1.0, True, "loop"), (0.5, True, "loop")])
def test_FINDING_capacity_routing_leaks_future_into_past(cf, legacy, dispatch):
    m = make(cf=cf, legacy=legacy, dispatch=dispatch)
    st = leak_stats(m, False)
    cs = cross_sample_leak(m)
    print(f"CAUSAL capacity cf={cf} legacy={legacy} dispatch={dispatch}: {json.dumps(st)} cross_sample={cs}")
    assert st["leaking_trials"] > 0 and st["max_abs_change_past"] > 0


def test_noncausal_model_leaks_even_dropless():
    m = make(causal=False, cf=None)
    st = leak_stats(m, False)
    print("NON-causal model, dropless:", json.dumps(st))
    assert st["leaking_trials"] == st["trials"]
