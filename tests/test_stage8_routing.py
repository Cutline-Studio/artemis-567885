"""Stage 8: Switch-style gate weighting (review #6), router z-loss, expert-choice routing."""
import json, math
import pytest
import torch
import torch.nn.functional as F
from moe import CapacitySparseMoELayer
from transformer import MoETransformerBlock
from model import DynamicMultiBlockTransformer, ArtemisModel
import train_bytes as tb

D = 16
TEXT = (b"the quick brown fox jumps over the lazy dog. " * 40
        + b"pack my box with five dozen liquor jugs. " * 40)


def layer(K=2, cf=None, gw="renorm", routing="token_choice", dispatch="loop", mode="global", E=4, seed=0):
    torch.manual_seed(seed)
    return CapacitySparseMoELayer(D, 32, E, K, cf, dispatch=dispatch, capacity_mode=mode,
                                  gate_weighting=gw, routing=routing).eval()


def xin(B=3, T=12, seed=1):
    return torch.randn(B, T, D, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- Switch weighting (review #6)
@pytest.mark.parametrize("dispatch", ["loop", "vectorized"])
@pytest.mark.parametrize("train", [False, True])
def test_top1_renorm_gate_gets_no_task_gradient_but_switch_does(dispatch, train):
    grads = {}
    for gw in ("renorm", "switch"):
        m = layer(K=1, gw=gw, dispatch=dispatch)
        m.train(train)
        torch.manual_seed(5)
        y, aux, _ = m(xin())
        y.pow(2).mean().backward()            # task loss only, no aux
        grads[gw] = m.gate.weight.grad
    assert grads["renorm"] is None or float(grads["renorm"].abs().max()) == 0.0   # dead (review #6)
    assert float(grads["switch"].norm()) > 1e-4


def test_switch_top1_gate_grad_through_artemis_model():
    torch.manual_seed(0)
    m = ArtemisModel(32, 4, 64, 4, 2, top_k=1, gate_weighting="switch")
    ids = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(0))
    logits, _, _ = m(ids)
    F.cross_entropy(logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1)).backward()
    for b in m.blocks:
        assert float(b.moe.gate.weight.grad.norm()) > 0


@pytest.mark.parametrize("K", [1, 2])
def test_switch_weights_are_full_router_probabilities(K):
    m = layer(K=K, gw="switch")
    x = xin()
    flat = x.reshape(-1, D)
    probs = F.softmax(m.gate(flat), -1)
    w = probs.topk(K, -1)
    y, _, _ = m(x)
    ref = torch.zeros_like(flat)
    for k in range(K):
        for e in range(4):
            sel = w.indices[:, k] == e
            ref[sel] += m.experts[e](flat[sel]) * w.values[sel, k:k + 1]
    assert torch.allclose(y.reshape(-1, D), ref, atol=1e-6)
    if K == 2:   # not renormalised: the two weights sum to < 1
        assert float(w.values.detach().sum(-1).max()) < 1.0


def test_renorm_default_unchanged():
    a = layer(K=2)
    torch.manual_seed(0)
    b = CapacitySparseMoELayer(D, 32, 4, 2, None).eval()   # constructor defaults
    b.load_state_dict(a.state_dict())
    assert a.gate_weighting == b.gate_weighting == "renorm" and b.routing == "token_choice"
    x = xin()
    flat = x.reshape(-1, D)
    lg = b.gate(flat)
    tl, ti = lg.topk(2, -1)
    w = F.softmax(tl, -1)
    ref = torch.zeros_like(flat)
    for k in range(2):
        for e in range(4):
            sel = ti[:, k] == e
            ref[sel] += b.experts[e](flat[sel]) * w[sel, k:k + 1]
    assert torch.allclose(b(x)[0].reshape(-1, D), ref, atol=1e-6)


@pytest.mark.parametrize("cf,mode", [(None, "global"), (0.5, "global"), (0.5, "sequence")])
@pytest.mark.parametrize("K", [1, 2])
def test_switch_loop_equals_vectorized(cf, mode, K):
    a = layer(K=K, cf=cf, gw="switch", mode=mode, dispatch="loop")
    b = layer(K=K, cf=cf, gw="switch", mode=mode, dispatch="vectorized")
    b.load_state_dict(a.state_dict())
    xa, xb = xin().requires_grad_(), xin().requires_grad_()
    ya, aa, da = a(xa); yb, ab, db = b(xb)
    assert da == db and torch.allclose(ya, yb, atol=1e-6) and torch.allclose(aa, ab)
    ya.pow(2).sum().backward(); yb.pow(2).sum().backward()
    assert torch.allclose(xa.grad, xb.grad, atol=1e-5)
    assert torch.allclose(a.gate.weight.grad, b.gate.weight.grad, atol=1e-5)


@pytest.mark.parametrize("kw", [dict(gate_weighting="switch"), dict(routing="expert_choice")])
def test_legacy_rejects_stage8_options(kw):
    with pytest.raises(ValueError):
        CapacitySparseMoELayer(D, 32, 4, 2, 1.0, legacy=True, **kw)


def test_bad_option_values_rejected():
    with pytest.raises(ValueError):
        CapacitySparseMoELayer(D, 32, 4, 2, None, gate_weighting="softmax")
    with pytest.raises(ValueError):
        CapacitySparseMoELayer(D, 32, 4, 2, None, routing="hash")


# ---------------------------------------------------------------- z-loss
def test_z_loss_closed_form_zero_gate():
    m = layer(E=4)
    with torch.no_grad():
        m.gate.weight.zero_()
    m(xin())
    assert math.isclose(float(m.last_z_loss.detach()), math.log(4) ** 2, rel_tol=1e-6)


@pytest.mark.parametrize("routing", ["token_choice", "expert_choice"])
@pytest.mark.parametrize("train", [False, True])
def test_z_loss_matches_python_reference_on_clean_logits(routing, train):
    m = layer(routing=routing)
    m.train(train)
    x = xin()
    m(x)
    lg = (x.reshape(-1, D) @ m.gate.weight.t()).double().tolist()
    ref = sum(math.log(sum(math.exp(v) for v in row)) ** 2 for row in lg) / len(lg)
    assert math.isclose(float(m.last_z_loss.detach()), ref, rel_tol=1e-5)   # noise never enters z


def test_z_loss_has_gradient_to_gate_and_sums_over_layers():
    torch.manual_seed(0)
    m = ArtemisModel(32, 4, 64, 4, 3, z_loss_coef=1e-3)
    ids = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(0))
    m(ids)
    per = [float(b.moe.last_z_loss.detach()) for b in m.blocks]
    assert math.isclose(float(m.last_z_loss.detach()), sum(per), rel_tol=1e-6)
    r = m.regularizer()
    assert math.isclose(float(r.detach()), 1e-3 * sum(per), rel_tol=1e-6)
    r.backward()
    for b in m.blocks:
        assert float(b.moe.gate.weight.grad.norm()) > 0


def test_z_loss_default_off_is_graph_free_zero():
    torch.manual_seed(0)
    m = ArtemisModel(32, 4, 64, 4, 2)
    assert m.z_loss_coef == 0.0 and m.body.gate_weighting == "renorm" and m.body.routing == "token_choice"
    m(torch.zeros(1, 8, dtype=torch.long))
    r = m.regularizer()
    assert float(r) == 0.0 and not r.requires_grad


def test_legacy_outputs_unchanged_and_z_recorded():
    # the legacy sandbox numbers (Total Loss 1.5083) come from this exact block config
    torch.manual_seed(42)
    blk = MoETransformerBlock(16, 2, 64, 4, top_k=2, capacity_factor=0.8, legacy=True)
    x = torch.randn(2, 8, 16, requires_grad=True)
    blk.train()
    y, aux, dropped = blk(x)
    loss = F.mse_loss(y, torch.zeros_like(y)) + 0.1 * aux
    assert round(loss.item(), 4) == 1.5083 and dropped == 6
    assert blk.moe.last_z_loss is not None and float(blk.moe.last_z_loss.detach()) > 0


# ---------------------------------------------------------------- expert choice
@pytest.mark.parametrize("mode", ["global", "sequence"])
@pytest.mark.parametrize("cf", [None, 0.5, 1.5])
def test_expert_choice_picks_top_c_tokens_per_expert(mode, cf):
    B, T, E, K = 3, 12, 4, 2
    m = layer(K=K, cf=cf, routing="expert_choice", mode=mode)
    x = xin(B, T)
    y, aux, dropped = m(x)
    flat = x.reshape(-1, D)
    probs = F.softmax(m.gate(flat), -1)
    n = T if mode == "sequence" else B * T
    G = B if mode == "sequence" else 1
    C = min(n, math.ceil(n / E * (cf or 1.0) * K))
    assert m.last_stats["capacity"] == C
    ref = torch.zeros_like(flat)
    covered = torch.zeros(B * T, dtype=torch.long)
    for e in range(E):
        for g in range(G):
            p = probs[g * n:(g + 1) * n, e]
            idx = torch.topk(p, C).indices + g * n
            ref[idx] += m.experts[e](flat[idx]) * probs[idx, e:e + 1]
            covered[idx] += 1
    assert torch.allclose(y.reshape(-1, D), ref, atol=1e-6)
    assert dropped == int((covered == 0).sum()) == m.last_stats["dropped_tokens"]
    assert torch.equal(m.last_stats["experts_per_token"], covered)
    assert math.isclose(float(aux.detach()), 1.0, rel_tol=1e-6)          # every expert gets exactly G*C tokens
    assert m.last_stats["kept_load"].tolist() == [float(G * C)] * E


@pytest.mark.parametrize("mode", ["global", "sequence"])
@pytest.mark.parametrize("train", [False, True])
def test_expert_choice_loop_equals_vectorized(mode, train):
    a = layer(cf=1.0, routing="expert_choice", mode=mode, dispatch="loop")
    b = layer(cf=1.0, routing="expert_choice", mode=mode, dispatch="vectorized")
    b.load_state_dict(a.state_dict())
    a.train(train); b.train(train)
    xa, xb = xin().requires_grad_(), xin().requires_grad_()
    torch.manual_seed(3); ya, aa, da = a(xa)
    torch.manual_seed(3); yb, ab, db = b(xb)
    assert da == db and torch.allclose(ya, yb, atol=1e-6)
    (ya.pow(2).sum() + aa).backward(); (yb.pow(2).sum() + ab).backward()
    assert torch.allclose(xa.grad, xb.grad, atol=1e-5)
    assert torch.allclose(a.gate.weight.grad, b.gate.weight.grad, atol=1e-5)
    assert float(a.gate.weight.grad.norm()) > 0                  # gate trained by the task loss


def test_expert_choice_sequence_mode_has_no_cross_sequence_dependence():
    m = layer(cf=1.0, routing="expert_choice", mode="sequence")
    x = xin(3, 12)
    y = m(x)[0]
    x2 = x.clone(); x2[1:] = torch.randn(2, 12, D)
    assert torch.allclose(m(x2)[0][0], y[0], atol=1e-6)


def test_expert_choice_is_not_causal_and_flagged():
    """Expert choice leaks: changing future bytes changes past logits. This is why it's labelled leaky."""
    torch.manual_seed(0)
    m = ArtemisModel(32, 4, 64, 4, 2, routing="expert_choice", capacity_mode="sequence").eval()
    assert m.causal_safe is False
    ids = torch.randint(0, 256, (1, 24), generator=torch.Generator().manual_seed(0))
    base = m(ids)[0][0, :12]
    leaked = False
    for s in range(20):
        ids2 = ids.clone()
        ids2[0, 12:] = torch.randint(0, 256, (12,), generator=torch.Generator().manual_seed(100 + s))
        if not torch.allclose(m(ids2)[0][0, :12], base, atol=1e-6):
            leaked = True
            break
    assert leaked


def test_expert_choice_leak_is_invisible_to_jacobian_check():
    """The top-C pick is discrete, so d(past)/d(future) is exactly 0 even though expert choice leaks
    (previous test): a zero-Jacobian test alone can't certify causality; the perturbation test can."""
    torch.manual_seed(0)
    body = DynamicMultiBlockTransformer(D, 2, 32, 4, 2, top_k=2, capacity_factor=1.0, causal=True,
                                        routing="expert_choice", capacity_mode="sequence",
                                        pos_encoding="sinusoidal").eval()
    x = torch.randn(1, 16, D, generator=torch.Generator().manual_seed(4), requires_grad=True)
    y = body(x)[0]
    y[:, :8].sum().backward()
    assert float(x.grad[:, 8:].abs().max()) == 0.0
    changed = False
    for s in range(20):
        x2 = x.detach().clone()
        x2[:, 8:] = torch.randn(1, 8, D, generator=torch.Generator().manual_seed(50 + s)) * 3
        if not torch.allclose(body(x2)[0][:, :8], y[:, :8].detach(), atol=1e-6):
            changed = True
            break
    assert changed


@pytest.mark.parametrize("kw", [dict(), dict(gate_weighting="switch", top_k=1),
                                dict(gate_weighting="switch", capacity_factor=0.8, capacity_mode="sequence"),
                                dict(z_loss_coef=1e-2)])
@pytest.mark.parametrize("dispatch", ["loop", "vectorized"])
def test_token_choice_options_stay_causal(kw, dispatch):
    torch.manual_seed(0)
    m = ArtemisModel(32, 4, 64, 4, 2, dispatch=dispatch, **kw)
    assert m.causal_safe
    for train in (False, True):
        m.train(train)
        ids = torch.randint(0, 256, (2, 20), generator=torch.Generator().manual_seed(0))
        ids2 = ids.clone(); ids2[:, 10:] = (ids2[:, 10:] + 7) % 256
        torch.manual_seed(1); a = m(ids)[0][:, :10]
        torch.manual_seed(1); b = m(ids2)[0][:, :10]
        # experts run on different token subsets -> float rounding only (~1e-7), same routing
        assert torch.allclose(a, b, atol=1e-5, rtol=0)
    # exact: zero Jacobian d(past outputs)/d(future inputs) through the body
    body = m.body.train()
    x = torch.randn(2, 20, 32, generator=torch.Generator().manual_seed(2), requires_grad=True)
    torch.manual_seed(1)
    body(x)[0][:, :10].sum().backward()
    assert float(x.grad[:, 10:].abs().max()) == 0.0 and float(x.grad[:, :10].abs().max()) > 0


def test_causal_safe_flags():
    assert ArtemisModel(16, 2, 32, 2, 1).causal_safe
    assert not ArtemisModel(16, 2, 32, 2, 1, capacity_factor=1.0).causal_safe          # global capacity
    assert ArtemisModel(16, 2, 32, 2, 1, capacity_factor=1.0, capacity_mode="sequence").causal_safe
    assert not ArtemisModel(16, 2, 32, 2, 1, causal=False).causal_safe


def test_add_layer_keeps_stage8_options():
    m = DynamicMultiBlockTransformer(D, 2, 32, 4, 1, top_k=1, capacity_factor=None,
                                     gate_weighting="switch", routing="expert_choice")
    i = m.add_layer(4)
    assert m.blocks[i].moe.gate_weighting == "switch" and m.blocks[i].moe.routing == "expert_choice"


# ---------------------------------------------------------------- train_bytes flags
def targs(tmp_path, **kw):
    a = tb.build_parser().parse_args([])
    base = dict(d_model=32, n_heads=4, d_ff=64, experts=4, layers=2, seq_len=32, batch=8, lr=3e-3,
                warmup=5, max_steps=20, time_budget=0, eval_every=10, eval_windows=16, sample_len=10,
                prompt=["the "], out_dir=str(tmp_path), run_name="t", seed=1, threads=1)
    base.update(kw)
    for k, v in base.items():
        setattr(a, k, v)
    return a


def test_parser_defaults_and_flags():
    a = tb.build_parser().parse_args([])
    assert (a.gate_weighting, a.routing, a.z_loss) == ("renorm", "token_choice", 0.0)
    b = tb.build_parser().parse_args(["--gate-weighting", "switch", "--routing", "expert_choice", "--z-loss", "1e-3"])
    assert (b.gate_weighting, b.routing, b.z_loss) == ("switch", "expert_choice", 1e-3)


def test_train_default_flags_reproduce_stage7_path(tmp_path):
    """Explicit stage-8 defaults give bit-identical results to not setting them."""
    a = tb.train(targs(tmp_path, run_name="a"), TEXT, TEXT, log=lambda *_: None)
    b = tb.train(targs(tmp_path, run_name="b", gate_weighting="renorm", routing="token_choice", z_loss=0.0),
                 TEXT, TEXT, log=lambda *_: None)
    assert a["final_heldout_bpb"] == b["final_heldout_bpb"] and a["samples"] == b["samples"]
    assert a["leaky"] is False and a["z_loss_coef"] == 0.0


def test_train_switch_top1_zloss_and_expert_choice(tmp_path):
    s = tb.train(targs(tmp_path, run_name="s", top_k=1, gate_weighting="switch", z_loss=1e-2),
                 TEXT, TEXT, log=lambda *_: None)
    assert s["leaky"] is False and s["gate_weighting"] == "switch" and s["z_loss_coef"] == 1e-2
    assert s["train_z_mean"] > 0 and s["heldout_z_loss"] > 0
    assert s["final_heldout_bpb"] < s["initial_heldout_bpb"]
    msgs = []
    e = tb.train(targs(tmp_path, run_name="e", routing="expert_choice", capacity_factor=1.0),
                 TEXT, TEXT, log=msgs.append)
    assert e["leaky"] is True and e["routing"] == "expert_choice"
    assert any("not causal" in m for m in msgs)
    assert 0.0 <= e["heldout_drop_rate"] <= 1.0
    assert abs(e["heldout_entropy_kept_mean"] - 1.0) < 1e-9
    row = json.loads((tmp_path / "e" / "curve.jsonl").read_text().splitlines()[-1])
    assert "heldout_z_loss" in row and "train_z_mean" in row
