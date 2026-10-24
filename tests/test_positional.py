"""Stage 3: positional encodings + causal mask acceptance checks."""
import math
import pytest
import torch
from model import DynamicMultiBlockTransformer
from positional import sinusoidal_table, SinusoidalPositionalEncoding, LearnedPositionalEncoding

D = 16


def make(pos="none", causal=False, cf=None, layers=2, seed=0):
    torch.manual_seed(seed)
    m = DynamicMultiBlockTransformer(D, 2, 32, 4, layers, top_k=2, capacity_factor=cf,
                                     causal=causal, pos_encoding=pos, max_len=64)
    m.eval()
    return m


def test_sinusoidal_known_values():
    pe = sinusoidal_table(8, 6)
    assert torch.allclose(pe[0], torch.tensor([0., 1., 0., 1., 0., 1.]))
    assert math.isclose(pe[1, 0].item(), math.sin(1.0), rel_tol=1e-6)
    assert math.isclose(pe[1, 1].item(), math.cos(1.0), rel_tol=1e-6)
    assert math.isclose(pe[3, 2].item(), math.sin(3 / 10000 ** (2 / 6)), rel_tol=1e-5)
    assert pe.abs().max() <= 1.0


def test_sinusoidal_odd_dim_and_bounds():
    pe = sinusoidal_table(10, 7)
    assert pe.shape == (10, 7) and torch.isfinite(pe).all()


@pytest.mark.parametrize("pos", ["none", "sinusoidal", "learned"])
def test_shape_and_gradients(pos):
    m = make(pos)
    m.train()
    x = torch.randn(2, 12, D, requires_grad=True)
    y, aux, dropped = m(x)
    assert y.shape == x.shape
    (y.pow(2).mean() + aux).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert m.blocks[0].moe.gate.weight.grad.abs().sum() > 0
    if pos == "learned":
        assert m.pos.emb.weight.grad[:12].abs().sum() > 0
        assert m.pos.emb.weight.grad[12:].abs().sum() == 0


def test_none_is_permutation_equivariant():
    m = make("none")
    x = torch.randn(2, 10, D)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        y = m(x)[0]
        yp = m(x[:, perm])[0]
    assert torch.allclose(yp, y[:, perm], atol=1e-5)


@pytest.mark.parametrize("pos", ["sinusoidal", "learned"])
def test_positional_breaks_permutation_equivariance(pos):
    m = make(pos)
    x = torch.randn(2, 10, D)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        y = m(x)[0]
        yp = m(x[:, perm])[0]
    assert not torch.allclose(yp, y[:, perm], atol=1e-4)


@pytest.mark.parametrize("pos", ["none", "sinusoidal", "learned"])
def test_causal_future_change_does_not_affect_past(pos):
    m = make(pos, causal=True, cf=None)  # dropless: capacity ranking would leak (check 5)
    x = torch.randn(3, 16, D, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        y = m(x)[0]
        for t in (0, 4, 9, 14):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randn(x[:, t + 1:].shape, generator=torch.Generator().manual_seed(t))
            y2 = m(x2)[0]
            assert torch.equal(y[:, :t + 1], y2[:, :t + 1])
            assert not torch.allclose(y[:, t + 1:], y2[:, t + 1:])


def test_max_len_enforced():
    for mod in (SinusoidalPositionalEncoding(D, 8), LearnedPositionalEncoding(D, 8)):
        with pytest.raises(ValueError):
            mod(torch.zeros(1, 9, D))


def test_bad_kind_rejected():
    with pytest.raises(ValueError):
        DynamicMultiBlockTransformer(D, 2, 32, 4, 1, pos_encoding="rope")


def test_default_state_dict_unchanged():
    a = make("none"); s = make("sinusoidal")
    assert set(a.state_dict()) == set(s.state_dict())  # sinusoidal buffer is non-persistent
    l = make("learned")
    assert set(l.state_dict()) - set(a.state_dict()) == {"pos.emb.weight"}
