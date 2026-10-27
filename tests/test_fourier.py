"""Stage 9(a): multiscale Fourier feature encoding (fourier.py)."""
import math
import pytest
import torch
from fourier import FourierFeatures


@pytest.mark.parametrize("in_dim", [1, 2, 3, 7])
@pytest.mark.parametrize("L", [1, 4, 10])
@pytest.mark.parametrize("include_input", [False, True])
def test_output_dim(in_dim, L, include_input):
    enc = FourierFeatures(in_dim, L, include_input=include_input)
    x = torch.rand(5, 3, in_dim)
    y = enc(x)
    expected = 2 * L * in_dim + (in_dim if include_input else 0)
    assert y.shape == (5, 3, expected) and enc.out_dim == expected


def test_any_leading_shape_and_bad_dim():
    enc = FourierFeatures(2, 3)
    assert enc(torch.rand(2)).shape == (12,)
    assert enc(torch.rand(4, 5, 6, 2)).shape == (4, 5, 6, 12)
    with pytest.raises(ValueError):
        enc(torch.rand(4, 3))
    with pytest.raises(ValueError):
        FourierFeatures(0, 3)
    with pytest.raises(ValueError):
        FourierFeatures(1, 0)


def test_include_input_passthrough():
    enc = FourierFeatures(3, 4, include_input=True)
    x = torch.randn(10, 3) * 5
    assert torch.equal(enc(x)[..., :3], x)


def test_values_bounded():
    enc = FourierFeatures(3, 12, include_input=True, gaussian_features=16, gaussian_scale=10.0)
    x = torch.randn(1000, 3) * 100
    y = enc(x)[..., 3:]  # sin/cos part only
    assert y.abs().max().item() <= 1.0


def test_known_values_t0():
    L = 6
    y = FourierFeatures(1, L)(torch.zeros(1, 1))[0]
    assert torch.equal(y[:L], torch.zeros(L))  # sin(0) = 0
    assert torch.equal(y[L:], torch.ones(L))   # cos(0) = 1


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
def test_known_values_t_half(dtype, atol):
    # sin(2^k pi/2) = 1, 0, 0, ...;  cos(2^k pi/2) = 0, -1, 1, 1, ...
    L = 6
    enc = FourierFeatures(1, L)
    y = enc(torch.full((1,), 0.5, dtype=dtype))
    exp_sin = torch.tensor([1, 0, 0, 0, 0, 0], dtype=dtype)
    exp_cos = torch.tensor([0, -1, 1, 1, 1, 1], dtype=dtype)
    torch.testing.assert_close(y[:L], exp_sin, atol=atol, rtol=0)
    torch.testing.assert_close(y[L:], exp_cos, atol=atol, rtol=0)


def test_layout_matches_reference_multi_dim():
    in_dim, L = 3, 4
    x = torch.randn(7, in_dim, dtype=torch.float64)
    enc = FourierFeatures(in_dim, L, include_input=True).double()
    y = enc(x)
    ref = [x]
    ref += [torch.sin(2 ** k * math.pi * x) for k in range(L)]
    ref += [torch.cos(2 ** k * math.pi * x) for k in range(L)]
    torch.testing.assert_close(y, torch.cat(ref, -1))


def test_frequency_scaling():
    """Band k at t equals band 0 at 2^k t; band k has period 2^(1-k)."""
    L = 8
    enc = FourierFeatures(1, L).double()
    t = torch.linspace(-1, 1, 101, dtype=torch.float64).unsqueeze(-1)
    y = enc(t)
    for k in range(L):
        y0 = enc(t * 2 ** k)
        torch.testing.assert_close(y[:, k], y0[:, 0])
        torch.testing.assert_close(y[:, L + k], y0[:, L])
        yp = enc(t + 2.0 ** (1 - k))
        torch.testing.assert_close(y[:, k], yp[:, k], atol=1e-9, rtol=0)
    # the frequency buffer itself is 2^k * pi
    torch.testing.assert_close(enc.frequencies(torch.float64), 2.0 ** torch.arange(L, dtype=torch.float64) * math.pi)


def test_differentiable_gradient_matches_analytic():
    L = 5
    enc = FourierFeatures(2, L, include_input=True, gaussian_features=4).double()
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    # analytic grad of sum: 1 + sum_k w_k (cos(w_k x) - sin(w_k x)) + gaussian terms
    xd = x.detach()
    w = 2.0 ** torch.arange(L, dtype=torch.float64) * math.pi
    wx = xd.unsqueeze(-1) * w
    g = 1 + (w * (torch.cos(wx) - torch.sin(wx))).sum(-1)
    gp = 2 * math.pi * (xd @ enc.B)
    g = g + ((torch.cos(gp) - torch.sin(gp)) * 2 * math.pi) @ enc.B.T
    torch.testing.assert_close(x.grad, g)
    assert torch.autograd.gradcheck(enc, (torch.randn(2, 2, dtype=torch.float64, requires_grad=True),))


def test_no_trainable_parameters():
    enc = FourierFeatures(2, 4, gaussian_features=8)
    assert list(enc.parameters()) == []


def test_gaussian_shape_and_determinism():
    a = FourierFeatures(3, 4, gaussian_features=16, gaussian_scale=2.0, seed=567885)
    b = FourierFeatures(3, 4, gaussian_features=16, gaussian_scale=2.0, seed=567885)
    c = FourierFeatures(3, 4, gaussian_features=16, gaussian_scale=2.0, seed=1)
    assert a.B.shape == (3, 16)
    assert torch.equal(a.B, b.B) and not torch.equal(a.B, c.B)
    x = torch.rand(10, 3)
    assert a(x).shape == (10, 2 * 4 * 3 + 2 * 16) and a.out_dim == 56
    assert torch.equal(a(x), b(x))
    # construction does not consume or depend on the global RNG
    torch.manual_seed(0); s0 = torch.rand(1)
    torch.manual_seed(0); FourierFeatures(3, 0, gaussian_features=16, seed=5); s1 = torch.rand(1)
    assert torch.equal(s0, s1)


def test_gaussian_only_and_reference_and_scale():
    enc = FourierFeatures(2, 0, gaussian_features=8, gaussian_scale=3.0, seed=7).double()
    x = torch.randn(4, 2, dtype=torch.float64)
    gp = 2 * math.pi * x @ enc.B
    torch.testing.assert_close(enc(x), torch.cat([torch.sin(gp), torch.cos(gp)], -1))
    big = FourierFeatures(2, 0, gaussian_features=4096, gaussian_scale=3.0, seed=7)
    assert abs(big.B.std().item() - 3.0) < 0.1


def test_gaussian_B_in_state_dict():
    a = FourierFeatures(2, 3, gaussian_features=5, seed=1)
    b = FourierFeatures(2, 3, gaussian_features=5, seed=2)
    assert list(a.state_dict()) == ["B"]
    assert list(FourierFeatures(2, 3).state_dict()) == []
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.B, b.B)
