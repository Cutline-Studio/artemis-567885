"""Stage 5: ArtemisModel (byte embedding -> N MoE blocks -> byte logits)."""
import math
import pytest
import torch
from model import ArtemisModel, DynamicMultiBlockTransformer, next_byte_loss, bits_per_byte

D, H, F, E = 32, 4, 64, 4


def make(seed=0, layers=3, cf=None, causal=True, pos="sinusoidal", **kw):
    torch.manual_seed(seed)
    return ArtemisModel(D, H, F, E, layers, top_k=2, capacity_factor=cf, causal=causal,
                        pos_encoding=pos, max_len=64, **kw)


def ids(B=2, T=16, seed=0):
    return torch.randint(0, 256, (B, T), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("B,T", [(1, 1), (2, 16), (3, 33)])
def test_output_shape(B, T):
    m = make()
    logits, aux, dropped = m(ids(B, T))
    assert logits.shape == (B, T, 256)
    assert aux.dim() == 0 and torch.isfinite(aux) and isinstance(dropped, int)
    assert m.num_layers == 3 and m.num_experts == [E, E, E]


def test_body_is_dynamic_multiblock_with_options():
    m = make(pos="learned")
    assert isinstance(m.body, DynamicMultiBlockTransformer)
    assert m.body.causal and m.body.pos_encoding == "learned"
    assert all(b.attn.causal for b in m.blocks)
    assert all(b.moe.capacity_factor is None for b in m.blocks)


def test_gradients_reach_embedding_gates_and_experts():
    m = make()
    m.train()
    x = ids(4, 32, seed=1)
    logits, aux, _ = m(x)
    (next_byte_loss(logits, x) + 0.01 * aux).backward()
    g = m.embed.weight.grad
    assert g is not None and g[x.unique()].abs().sum(-1).gt(0).all()  # every used byte row
    unused = torch.ones(256, dtype=torch.bool); unused[x.unique()] = False
    assert g[unused].abs().sum() == 0  # rows of unseen bytes get no gradient
    assert m.head.weight.grad.abs().sum() > 0
    for li, b in enumerate(m.blocks):
        assert b.moe.gate.weight.grad is not None and b.moe.gate.weight.grad.abs().sum() > 0, li
        for ei, e in enumerate(b.moe.experts):
            for p in e.parameters():
                assert p.grad is not None and p.grad.abs().sum() > 0, (li, ei)


def _per_layer_hooks(m):
    rec = []
    hs = [b.moe.register_forward_hook(lambda mod, inp, out: rec.append((out[1].detach().clone(), int(out[2]))))
          for b in m.blocks]
    return rec, hs


@pytest.mark.parametrize("legacy", [False, True])
def test_aux_and_dropped_summed_over_layers(legacy):
    m = make(cf=0.5, legacy=legacy)  # tight capacity so drops actually happen
    m.eval()
    rec, hs = _per_layer_hooks(m)
    with torch.no_grad():
        _, aux, dropped = m(ids(2, 24, seed=3))
    for h in hs:
        h.remove()
    assert len(rec) == 3
    assert torch.allclose(aux, sum(a for a, _ in rec))
    assert dropped == sum(d for _, d in rec) and dropped > 0
    print("per-layer aux", [round(float(a), 4) for a, _ in rec], "dropped", [d for _, d in rec],
          "sum", round(float(aux), 4), dropped)


def test_legacy_requires_float_capacity():
    with pytest.raises(ValueError):
        make(cf=None, legacy=True)


def test_dropless_default_has_no_drops():
    m = make()
    _, _, dropped = m(ids(4, 32))
    assert dropped == 0


@pytest.mark.parametrize("pos", ["none", "sinusoidal", "learned"])
@pytest.mark.parametrize("dispatch", ["loop", "vectorized"])
def test_causal_future_bytes_do_not_affect_past(pos, dispatch):
    m = make(pos=pos, dispatch=dispatch)  # default dropless routing
    m.eval()
    x = ids(3, 20, seed=5)
    with torch.no_grad():
        y = m(x)[0]
        for t in (0, 5, 11, 18):
            x2 = x.clone()
            x2[:, t + 1:] = (x[:, t + 1:] + 1 + ids(3, 20, seed=t)[:, t + 1:] % 255) % 256  # always differs
            assert (x2[:, t + 1:] != x[:, t + 1:]).all()
            y2 = m(x2)[0]
            assert torch.equal(y[:, :t + 1], y2[:, :t + 1])
            assert not torch.allclose(y[:, t + 1:], y2[:, t + 1:])


def test_noncausal_option_leaks():
    m = make(causal=False)
    m.eval()
    x = ids(1, 12)
    x2 = x.clone(); x2[:, -1] = (x[:, -1] + 1) % 256
    with torch.no_grad():
        assert not torch.allclose(m(x)[0][:, 0], m(x2)[0][:, 0])


@pytest.mark.parametrize("bad", [torch.tensor([[0, 256]]), torch.tensor([[-1, 3]])])
def test_rejects_out_of_range_bytes(bad):
    with pytest.raises(ValueError):
        make()(bad)


def test_rejects_float_and_wrong_rank():
    m = make()
    with pytest.raises(TypeError):
        m(torch.zeros(1, 4))
    with pytest.raises(ValueError):
        m(torch.zeros(4, dtype=torch.long))


def test_accepts_uint8_bytes():
    m = make(); m.eval()
    raw = torch.tensor([list(b"hello artemis")], dtype=torch.uint8)
    with torch.no_grad():
        assert torch.equal(m(raw)[0], m(raw.long())[0])


def test_determinism_with_fixed_seed():
    x = ids(2, 16, seed=7)
    outs = []
    for _ in range(2):
        m = make(seed=123)
        m.train()  # includes any training-mode randomness
        torch.manual_seed(9)
        logits, aux, dropped = m(x)
        outs.append((logits.detach(), float(aux.detach()), dropped))
    assert torch.equal(outs[0][0], outs[1][0]) and outs[0][1:] == outs[1][1:]
    m3 = make(seed=124)
    assert not torch.allclose(m3(x)[0], outs[0][0])


def test_loss_helpers():
    x = ids(2, 10)
    logits = torch.zeros(2, 10, 256)
    loss = next_byte_loss(logits, x)
    assert math.isclose(float(loss), math.log(256), rel_tol=1e-6)  # uniform -> ln 256 nats
    assert math.isclose(bits_per_byte(loss), 8.0, rel_tol=1e-6)    # = 8 bits per byte


def test_tiny_overfit_sanity():
    torch.manual_seed(0)
    data = torch.tensor([list(b"the quick brown fox jumps over the lazy dog. " * 2)], dtype=torch.long)
    m = ArtemisModel(32, 4, 64, 4, 2, top_k=2, max_len=128)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    losses = []
    for step in range(50):
        logits, aux, dropped = m(data)
        loss = next_byte_loss(logits, data)
        opt.zero_grad(); (loss + 0.01 * aux).backward(); opt.step()
        losses.append(float(loss.detach()))
    first, last = losses[0], losses[-1]
    print(f"overfit: step0 loss {first:.4f} ({bits_per_byte(first):.3f} bpb) -> step49 {last:.4f} "
          f"({bits_per_byte(last):.3f} bpb), min {min(losses):.4f}, dropped {dropped}")
    assert last < 0.5 * first
    assert dropped == 0
