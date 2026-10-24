"""Stage 7: causal-safe per-sequence capacity (capacity_mode="sequence"),
the expert-load entropy helper, and the train_bytes.py capacity/routing metrics."""
import json, math
import pytest
import torch
from moe import CapacitySparseMoELayer, expert_load_entropy
from model import DynamicMultiBlockTransformer, ArtemisModel
import train_bytes as tb

D = 16


def layer(cf=0.5, E=4, K=2, dispatch="loop", mode="sequence", seed=0):
    torch.manual_seed(seed)
    return CapacitySparseMoELayer(D, 32, E, K, cf, dispatch=dispatch, capacity_mode=mode).eval()


def reference_keep(topk, B, T, E, K, cap):
    """Brute force: per sequence and expert, keep the first `cap` assignments in (position, slot) order."""
    keep = torch.zeros(B * T, K, dtype=torch.bool)
    for b in range(B):
        used = [0] * E
        for t in range(T):
            for k in range(K):
                e = int(topk[b * T + t, k])
                if used[e] < cap:
                    keep[b * T + t, k] = True
                used[e] += 1
    return keep


@pytest.mark.parametrize("cf", [0.25, 0.5, 0.8, 1.0, 1.5])
@pytest.mark.parametrize("dispatch", ["loop", "vectorized"])
def test_sequence_capacity_drop_counts_match_reference(cf, dispatch):
    B, T, E, K = 3, 20, 4, 2
    m = layer(cf, E, K, dispatch)
    x = torch.randn(B, T, D, generator=torch.Generator().manual_seed(1))
    _, _, dropped = m(x)
    st = m.last_stats
    cap = math.ceil(T / E * cf * K)
    assert st["capacity"] == cap                        # per-sequence, uses T not B*T
    ref = reference_keep(st["topk_indices"], B, T, E, K, cap)
    assert torch.equal(st["kept_slots"], ref)
    assert dropped == int((~ref).sum()) == st["dropped_assignments"]
    # closed form: sum over (sequence, expert) of max(0, n - cap)
    idx = st["topk_indices"].reshape(B, T * K)
    closed = sum(max(0, int((idx[b] == e).sum()) - cap) for b in range(B) for e in range(E))
    assert dropped == closed
    if cf <= 0.5:
        assert dropped > 0


def test_sequence_capacity_loop_equals_vectorized_outputs_and_grads():
    B, T = 4, 24
    a, b = layer(0.5, dispatch="loop"), layer(0.5, dispatch="vectorized")
    b.load_state_dict(a.state_dict())
    x = torch.randn(B, T, D, generator=torch.Generator().manual_seed(2))
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ya, aa, da = a(xa); yb, ab, db = b(xb)
    assert da == db > 0
    assert torch.allclose(ya, yb, atol=1e-6) and torch.allclose(aa, ab)
    assert torch.equal(a.last_stats["kept_slots"], b.last_stats["kept_slots"])
    (ya.square().sum() + aa).backward(); (yb.square().sum() + ab).backward()
    assert torch.allclose(xa.grad, xb.grad, atol=1e-5)
    assert torch.allclose(a.gate.weight.grad, b.gate.weight.grad, atol=1e-5)


def test_sequence_capacity_dropped_slots_get_zero_output():
    m = layer(0.25, K=1)
    x = torch.randn(2, 16, D, generator=torch.Generator().manual_seed(3))
    y, _, d = m(x)
    gone = ~m.last_stats["kept_slots"].any(-1)
    assert d > 0 and gone.any()
    assert torch.equal(y.reshape(-1, D)[gone], torch.zeros(int(gone.sum()), D))
    # late positions are the ones that overflow (first-come-first-served)
    pos = torch.arange(16).repeat(2)
    assert pos[gone].float().mean() > pos[~gone].float().mean()


def test_sequence_mode_dropless_equals_global_dropless():
    a, b = layer(None, mode="sequence"), layer(None, mode="global")
    b.load_state_dict(a.state_dict())
    x = torch.randn(3, 10, D)
    ya, _, da = a(x); yb, _, db = b(x)
    assert da == db == 0 and torch.equal(ya, yb)


def test_sequence_mode_2d_input_is_one_sequence():
    m = layer(0.5)
    x = torch.randn(20, D, generator=torch.Generator().manual_seed(4))
    y2, _, d2 = m(x)
    y3, _, d3 = m(x[None])
    assert d2 == d3 and torch.equal(y2, y3[0])
    with pytest.raises(ValueError):
        m(torch.randn(2, 3, 4, D))


def test_sequence_independent_of_other_sequences():
    m = layer(0.5)
    x = torch.randn(4, 20, D, generator=torch.Generator().manual_seed(5))
    x2 = x.clone(); x2[1:] = torch.randn(3, 20, D)
    y, _, _ = m(x); y2, _, _ = m(x2)
    assert torch.equal(y[0], y2[0])
    y1, _, _ = m(x[:1])
    assert torch.equal(y[0], y1[0])


def test_global_mode_default_and_bad_options():
    assert CapacitySparseMoELayer(D, 32, 4).capacity_mode == "global"
    with pytest.raises(ValueError):
        CapacitySparseMoELayer(D, 32, 4, capacity_mode="batch")
    with pytest.raises(ValueError):
        CapacitySparseMoELayer(D, 32, 4, legacy=True, capacity_mode="sequence")


# ---------------- causal future-independence through attention + sequence capacity
B, L = 4, 24
CUTS = [0, 5, 11, 17, 22]


def fwd(m, x, train):
    m.train(train)
    torch.manual_seed(1234)
    return m(x)


@pytest.mark.parametrize("dispatch", ["loop", "vectorized"])
@pytest.mark.parametrize("cf", [0.5, 1.0])
@pytest.mark.parametrize("train", [False, True])
def test_causal_sequence_capacity_has_no_future_leak(dispatch, cf, train):
    torch.manual_seed(0)
    m = DynamicMultiBlockTransformer(32, 4, 64, 4, num_layers=2, top_k=2, capacity_factor=cf, causal=True,
                                     dispatch=dispatch, pos_encoding="sinusoidal", max_len=L,
                                     capacity_mode="sequence")
    x = torch.randn(B, L, 32, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        y, _, dropped = fwd(m, x, train)
        assert dropped > 0                              # capacity is actually binding
        for t in CUTS:
            for s in range(3):
                x2 = x.clone()
                x2[:, t + 1:] = torch.randn(x[:, t + 1:].shape, generator=torch.Generator().manual_seed(100 * t + s))
                y2, _, _ = fwd(m, x2, train)
                assert torch.equal(y[:, :t + 1], y2[:, :t + 1])
        x3 = x.clone(); x3[1] = torch.randn(L, 32, generator=torch.Generator().manual_seed(9))
        assert torch.equal(fwd(m, x, train)[0][0], fwd(m, x3, train)[0][0])   # no cross-sequence leak
    xg = x.clone().requires_grad_()
    yg = fwd(m, xg, train)[0]
    g, = torch.autograd.grad(yg[:, :12].sum(), xg)
    assert g[:, 12:].abs().max().item() == 0.0


def test_global_capacity_still_leaks_for_contrast():
    torch.manual_seed(0)
    m = DynamicMultiBlockTransformer(32, 4, 64, 4, num_layers=2, top_k=2, capacity_factor=0.5, causal=True)
    x = torch.randn(B, L, 32, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        y = fwd(m, x, False)[0]
        x2 = x.clone(); x2[:, 6:] = torch.randn(B, L - 6, 32, generator=torch.Generator().manual_seed(7))
        assert not torch.equal(y[:, :6], fwd(m, x2, False)[0][:, :6])


def test_artemis_model_byte_causality_with_sequence_capacity():
    torch.manual_seed(0)
    m = ArtemisModel(32, 4, 64, 4, 2, capacity_factor=0.8, max_len=40, capacity_mode="sequence").eval()
    ids = torch.randint(0, 256, (3, 40), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        lg, _, d = m(ids)
        assert d > 0
        for t in [0, 9, 25, 38]:
            ids2 = ids.clone(); ids2[:, t + 1:] = torch.randint(0, 256, (3, 39 - t))
            assert torch.equal(lg[:, :t + 1], m(ids2)[0][:, :t + 1])


def test_add_layer_keeps_capacity_mode():
    m = DynamicMultiBlockTransformer(32, 4, 64, 4, num_layers=1, capacity_factor=1.0, causal=True,
                                     capacity_mode="sequence")
    m.add_layer(3)
    assert [b.moe.capacity_mode for b in m.blocks] == ["sequence", "sequence"]


# ---------------- entropy helper
def test_expert_load_entropy_known_values():
    assert expert_load_entropy([5, 5, 5, 5]) == pytest.approx(1.0)
    assert expert_load_entropy([7, 0, 0, 0]) == 0.0
    assert expert_load_entropy([1, 1, 0, 0]) == pytest.approx(0.5)          # ln2 / ln4
    assert expert_load_entropy([1, 1, 0, 0], normalize=False) == pytest.approx(math.log(2))
    p = torch.tensor([0.5, 0.25, 0.25])
    assert expert_load_entropy(p * 8, normalize=False) == pytest.approx(1.5 * math.log(2))
    assert expert_load_entropy([0, 0, 0]) == 0.0 and expert_load_entropy([3]) == 0.0
    assert expert_load_entropy([2, 1]) == expert_load_entropy([200, 100])  # scale-free


# ---------------- train_bytes.py wiring
TEXT = (b"the quick brown fox jumps over the lazy dog. " * 40
        + b"pack my box with five dozen liquor jugs. " * 40)


def targs(tmp_path, **kw):
    a = tb.build_parser().parse_args([])
    base = dict(d_model=32, n_heads=4, d_ff=64, experts=4, layers=2, seq_len=32, batch=8, lr=3e-3,
                warmup=5, max_steps=20, time_budget=0, eval_every=10, eval_windows=16, sample_len=10,
                prompt=["the "], out_dir=str(tmp_path), run_name="t", seed=1, threads=1)
    base.update(kw)
    for k, v in base.items():
        setattr(a, k, v)
    return a


def test_parser_capacity_options():
    p = tb.build_parser()
    a = p.parse_args([])
    assert a.capacity_factor is None and a.capacity_mode == "sequence"
    assert p.parse_args(["--capacity-factor", "none"]).capacity_factor is None
    assert p.parse_args(["--capacity-factor", "1.25"]).capacity_factor == 1.25


def test_train_reports_drop_rate_and_entropy(tmp_path):
    s = tb.train(targs(tmp_path, capacity_factor=0.8, aux_alpha=0.1), TEXT, TEXT, log=lambda *_: None)
    assert s["capacity_mode"] == "sequence" and s["leaky"] is False
    assert 0 < s["train_drop_rate"] < 1 and 0 <= s["heldout_drop_rate"] < 1
    assert len(s["heldout_entropy_routed"]) == 2
    assert all(0 <= e <= 1 for e in s["heldout_entropy_routed"] + s["heldout_entropy_kept"])
    rows = [json.loads(l) for l in (tmp_path / "t" / "curve.jsonl").read_text().splitlines()]
    assert rows[-1]["heldout_drop_rate"] == s["heldout_drop_rate"]
    d = tb.train(targs(tmp_path, run_name="d"), TEXT, TEXT, log=lambda *_: None)
    assert d["train_drop_rate"] == 0 and d["heldout_drop_rate"] == 0 and d["capacity_factor"] is None


def test_global_capacity_is_labelled_leaky(tmp_path):
    msgs = []
    s = tb.train(targs(tmp_path, capacity_factor=1.0, capacity_mode="global", max_steps=5),
                 TEXT, TEXT, log=msgs.append)
    assert s["leaky"] is True and any("leak" in m for m in msgs)


def test_evaluate_routed_counts_cover_all_assignments():
    torch.manual_seed(0)
    m = ArtemisModel(32, 4, 64, 4, 2, capacity_factor=1.0, max_len=33, capacity_mode="sequence")
    w = tb.eval_windows(tb.to_tensor(TEXT), 33, 8)
    r = tb.evaluate(m, w, batch=3)
    assert r["assignments"] == 8 * 33 * 2 * 2
    assert all(sum(c) == 8 * 33 * 2 for c in r["load_routed"])
    assert r["drop_rate"] == r["dropped"] / r["assignments"]
