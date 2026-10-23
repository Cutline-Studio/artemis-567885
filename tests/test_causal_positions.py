import math

import pytest
import torch
import torch.nn.functional as F

from byte_model import ArtemisByteLM
from model import DynamicMultiBlockTransformer
from moe import CapacitySparseMoELayer
from positions import PositionEncoding, fourier_features
from transformer import MultiHeadAttention

D = 16


def _prefix_unchanged(fn, x, perturb, positions=range(1, 12, 2), seed=None):
    """True if editing positions >= t never changes outputs at positions < t."""
    if seed is not None:
        torch.manual_seed(seed)
    base = fn(x)
    for t in positions:
        x2 = perturb(x.clone(), t)
        if seed is not None:
            torch.manual_seed(seed)
        if not torch.allclose(fn(x2)[:, :t], base[:, :t], atol=1e-6):
            return False
    return True


def _perturb_vectors(x, t):
    x[:, t:] = torch.randn_like(x[:, t:]) * 3
    return x


def _perturb_bytes(x, t):
    x[:, t:] = torch.randint(0, 256, x[:, t:].shape)
    return x


# ---- Codex stage-3 check 1: causality --------------------------------------------------
def test_causal_attention_ignores_future():
    torch.manual_seed(0)
    attn = MultiHeadAttention(D, 2, causal=True)
    assert _prefix_unchanged(attn, torch.randn(2, 12, D), _perturb_vectors)


def test_default_attention_is_still_bidirectional():
    torch.manual_seed(0)
    attn = MultiHeadAttention(D, 2)
    assert not _prefix_unchanged(attn, torch.randn(2, 12, D), _perturb_vectors)


def _moe(mode, cf=0.5):
    torch.manual_seed(0)
    return CapacitySparseMoELayer(D, 32, 4, top_k=2, capacity_factor=cf, capacity_mode=mode).eval()


def test_batch_capacity_leaks_future_tokens_documented():
    layer = _moe("batch")
    torch.manual_seed(1)
    x = torch.randn(1, 16, D)
    assert layer(x)[2] > 0
    assert not _prefix_unchanged(lambda v: layer(v)[0], x, _perturb_vectors, range(1, 16))


@pytest.mark.parametrize("mode", ["causal", "none"])
def test_causal_and_none_capacity_preserve_causality(mode):
    layer = _moe(mode)
    torch.manual_seed(1)
    x = torch.randn(3, 16, D)
    if mode == "causal":
        assert layer(x)[2] > 0, "test must exercise real drops"
    else:
        assert layer(x)[2] == 0
    assert _prefix_unchanged(lambda v: layer(v)[0], x, _perturb_vectors, range(1, 16))


def test_causal_capacity_holds_in_training_mode_with_noise():
    layer = _moe("causal").train()
    torch.manual_seed(1)
    x = torch.randn(2, 16, D)
    assert _prefix_unchanged(lambda v: layer(v)[0], x, _perturb_vectors, range(1, 16), seed=123)


def test_causal_capacity_is_independent_across_sequences():
    layer = _moe("causal")
    torch.manual_seed(1)
    x = torch.randn(2, 16, D)
    base = layer(x)[0]
    x2 = x.clone(); x2[1] = torch.randn(16, D) * 3
    assert torch.allclose(layer(x2)[0][0], base[0], atol=1e-6)


def test_causal_capacity_respects_prefix_cap():
    layer = _moe("causal", cf=0.5)
    torch.manual_seed(2)
    x = torch.randn(2, 16, D)
    flat = x.reshape(-1, D)
    idx = torch.topk(layer.gate(flat), 2, dim=-1).indices
    keep = layer._causal_slot_mask(x, idx).view(2, 16, 2)
    idx = idx.view(2, 16, 2)
    scale = layer.capacity_factor * layer.top_k / layer.num_experts
    for b in range(2):
        for e in range(4):
            accepted = 0
            for t in range(16):
                accepted += int(((idx[b, t] == e) & keep[b, t]).sum())
                assert accepted <= math.ceil((t + 1) * scale)


def test_capacity_mode_validation():
    with pytest.raises(ValueError):
        CapacitySparseMoELayer(D, 32, 4, capacity_mode="bogus")
    with pytest.raises(ValueError):
        CapacitySparseMoELayer(D, 32, 4, legacy=True, capacity_mode="causal")


def test_default_batch_mode_numbers_unchanged():
    torch.manual_seed(0)
    a = CapacitySparseMoELayer(D, 32, 4, capacity_factor=0.5).eval()
    torch.manual_seed(0)
    b = CapacitySparseMoELayer(D, 32, 4, capacity_factor=0.5, capacity_mode="batch").eval()
    x = torch.randn(2, 8, D)
    assert torch.equal(a(x)[0], b(x)[0])


def test_causal_stack_prefix_matches_full_sequence():
    torch.manual_seed(0)
    m = DynamicMultiBlockTransformer(D, 2, 32, 4, 2, capacity_factor=0.5, causal=True,
                                     capacity_mode="causal").eval()
    x = torch.randn(2, 16, D)
    full = m(x)[0]
    for t in (1, 5, 11):
        assert torch.allclose(m(x[:, :t])[0], full[:, :t], atol=1e-5)


# ---- Codex stage-3 check 2: positions ----------------------------------------------------
@pytest.mark.parametrize("kind", ["learned", "sinusoidal", "fourier"])
def test_reordering_changes_outputs_with_positions(kind):
    torch.manual_seed(0)
    m = ArtemisByteLM(kind, d_model=D, d_ff=32, context=16, capacity_mode="none").eval()
    x = torch.randint(0, 256, (1, 16))
    perm = torch.randperm(16)
    y, _, _ = m(x)
    yp, _, _ = m(x[:, perm])
    assert not torch.allclose(yp[:, -1], y[:, -1], atol=1e-5)


def test_sinusoidal_and_fourier_have_identical_parameter_counts():
    counts = {k: sum(p.numel() for p in ArtemisByteLM(k).parameters())
              for k in ("sinusoidal", "fourier")}
    assert counts["sinusoidal"] == counts["fourier"]


def test_fourier_known_values():
    f = fourier_features(torch.tensor([0.0, 0.5]), 4)  # k = 0, 1 -> [sin pi t, sin 2pi t, cos pi t, cos 2pi t]
    assert torch.allclose(f[0], torch.tensor([0.0, 0.0, 1.0, 1.0]), atol=1e-6)
    assert torch.allclose(f[1], torch.tensor([1.0, 0.0, 0.0, -1.0]), atol=1e-6)
    assert f.abs().max() <= 1.0


def test_learned_positions_refuse_beyond_table():
    p = PositionEncoding("learned", D, max_len=16)
    with pytest.raises(ValueError):
        p(17)
    assert PositionEncoding("fourier", D, max_len=16)(32).shape == (32, D)


def test_none_positions_are_zero():
    assert torch.count_nonzero(PositionEncoding("none", D, max_len=8)(8)) == 0


# ---- byte LM ----------------------------------------------------------------------------
def test_byte_lm_rejects_batch_capacity():
    with pytest.raises(ValueError):
        ArtemisByteLM("fourier", capacity_mode="batch")


@pytest.mark.parametrize("kind", ["none", "learned", "sinusoidal", "fourier"])
def test_byte_lm_is_causal_end_to_end(kind):
    torch.manual_seed(0)
    m = ArtemisByteLM(kind, d_model=D, d_ff=32, context=16, capacity_factor=0.5).eval()
    x = torch.randint(0, 256, (2, 16))
    assert _prefix_unchanged(lambda v: m(v)[0], x, _perturb_bytes, range(1, 16))


def test_byte_lm_shapes_and_gradients():
    torch.manual_seed(0)
    m = ArtemisByteLM("fourier", d_model=D, d_ff=32, context=16).train()
    x = torch.randint(0, 256, (2, 16))
    logits, aux, dropped = m(x)
    assert logits.shape == (2, 16, 256) and aux.dim() == 0 and isinstance(dropped, int)
    (F.cross_entropy(logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)) + 0.01 * aux).backward()
    assert m.position.proj.weight.grad.norm() > 0
    for b in m.body.blocks:
        assert b.moe.gate.weight.grad.norm() > 0
