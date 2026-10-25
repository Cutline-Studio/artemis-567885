import torch, pytest
from artemis_dense import DenseConfig, ArtemisDense, expected_param_count

TINY = dict(vocab_size=256, context_len=32, n_layers=2, d_model=32, n_heads=4, n_kv_heads=4, d_ff=64)


def test_default_param_count_exact():
    cfg = DenseConfig()
    m = ArtemisDense(cfg)
    assert m.num_params() == expected_param_count(cfg) == 6_885_632


@pytest.mark.parametrize("kv", [4, 2, 1])
def test_param_count_matches_formula_mha_and_gqa(kv):
    cfg = DenseConfig(**{**TINY, "n_kv_heads": kv})
    assert ArtemisDense(cfg).num_params() == expected_param_count(cfg)


def test_untied_param_count():
    cfg = DenseConfig(**TINY, tie_embeddings=False)
    assert ArtemisDense(cfg).num_params() == expected_param_count(cfg)


def test_shapes():
    m = ArtemisDense(DenseConfig(**TINY))
    x = torch.randint(0, 256, (3, 17))
    logits, loss = m(x, x)
    assert logits.shape == (3, 17, 256) and loss.ndim == 0
    y, aux, dropped = m.forward_vectors(torch.randn(3, 17, 32))
    assert y.shape == (3, 17, 32) and float(aux) == 0.0 and dropped == 0


def test_causal_mask():
    torch.manual_seed(0)
    m = ArtemisDense(DenseConfig(**TINY)).eval()
    x = torch.randint(0, 256, (1, 20))
    x2 = x.clone(); x2[0, 12:] = (x2[0, 12:] + 1) % 256       # change only the future
    with torch.no_grad():
        a, _ = m(x); b, _ = m(x2)
    assert torch.allclose(a[0, :12], b[0, :12], atol=1e-6)
    assert not torch.allclose(a[0, 12:], b[0, 12:])


def test_forward_backward_cpu():
    m = ArtemisDense(DenseConfig(**TINY))
    x = torch.randint(0, 256, (2, 16))
    _, loss = m(x[:, :-1], x[:, 1:])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_tiny_overfit():
    torch.manual_seed(0)
    m = ArtemisDense(DenseConfig(**TINY))
    x = torch.randint(0, 256, (4, 17))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        _, loss = m(x[:, :-1], x[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.25 * first, (first, loss.item())


def test_context_len_enforced():
    m = ArtemisDense(DenseConfig(**TINY))
    with pytest.raises(ValueError):
        m(torch.zeros(1, 33, dtype=torch.long))


def test_config_validation():
    with pytest.raises(ValueError):
        DenseConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError):
        DenseConfig(n_heads=8, n_kv_heads=3)
