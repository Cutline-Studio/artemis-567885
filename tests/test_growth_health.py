"""Tests for the growth-health fixes (2026-10-27 external review):

  fix 1  growth gated on per-layer load health (no expert added next to an idle expert)
  fix 2  the load-balance `me` term counts every top-k slot (already true on the fixed path; regression guard)
  fix 3  stall needs the last growth to pay for itself (pre- vs post-growth eval window), so a converged
         model stops growing instead of growing every cycle
  fix 4  new experts are not clones of the most-loaded expert (default init = mean of the layer's experts)
"""
import copy, math, os, shutil
import pytest
import torch
import torch.nn.functional as F

import expand
from model import DynamicMultiBlockTransformer
from moe import CapacitySparseMoELayer, expert_load_entropy, expert_min_load_ratio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = {"steps_per_cycle": 20, "batch_size": 4, "seq_len": 8, "eval_batch_size": 4}


def rec(experts, loss=0.03, rel=0.0, before=None, loads=None, mutation=None):
    return {"loss": loss, "dropped": 0, "aux": 1.0, "rel_improvement": rel, "num_experts": list(experts),
            "num_experts_before": None if before is None else list(before),
            "expert_load": loads if loads is not None else [[10] * e for e in experts],
            "mutation": mutation or {"name": "none", "changes": {}}}


def cfg_hist(records, **over):
    cfg = copy.deepcopy(expand.DEFAULT_CONFIG)
    cfg.update(over)
    cfg["num_experts"] = list(records[-1]["num_experts"])
    cfg["num_layers"] = len(cfg["num_experts"])
    cfg["history"] = records
    return cfg


# ------------------------------------------------------------------ fix 1: load-health gate
def test_load_metrics():
    assert expert_load_entropy([5, 5, 5, 5]) == pytest.approx(1.0)
    assert expert_load_entropy([8, 0, 0, 0]) == 0.0
    assert expert_min_load_ratio([5, 5, 5, 5]) == pytest.approx(1.0)
    assert expert_min_load_ratio([10, 10, 0, 4]) == 0.0
    assert expert_min_load_ratio([10, 10, 2, 8]) == pytest.approx(4 * 2 / 30)
    assert expert_min_load_ratio([]) == 0.0 and expert_load_entropy([0, 0]) == 0.0


def test_no_expert_added_to_layer_with_idle_expert():
    loads = [[10] * 6, [12, 12, 12, 0, 12]]  # layer 1 (deepest, below cap) has an idle expert
    cfg = cfg_hist([rec([6, 5], loads=loads)])
    m = expand.plan_mutations(cfg)[0]
    assert m["changes"]["num_experts"] == [7, 5]  # skips layer 1, grows the deepest HEALTHY layer
    assert any("layer 1 not load-healthy" in r for r in expand.plan_mutations(cfg)[-1]["reasons"])


def test_low_entropy_layer_is_not_grown():
    loads = [[10] * 4, [40, 1, 1, 1]]  # min ratio 4*1/43 < 0.25, entropy ~0.2
    cfg = cfg_hist([rec([4, 4], loads=loads)])
    assert expand.plan_mutations(cfg)[0]["changes"]["num_experts"] == [5, 4]


def test_all_layers_unhealthy_means_no_growth_at_all():
    loads = [[20, 0, 20, 20], [30, 30, 0, 0]]
    cfg = cfg_hist([rec([4, 4], loads=loads)])
    cands = expand.plan_mutations(cfg)
    assert cands[0]["name"] == "none"
    assert any("not load-healthy" in r for r in cands[0]["reasons"])


def test_layer_add_needs_enough_healthy_layers():
    loads = [[16] * 16, [16, 0] + [16] * 14, [0] * 15 + [16]]
    cfg = cfg_hist([rec([16, 16, 16], loads=loads)])  # all full, 1/3 healthy < 0.5
    assert expand.plan_mutations(cfg)[0]["name"] == "none"
    cfg["rules"]["growth_min_healthy_layer_frac"] = 0.3
    assert expand.plan_mutations(cfg)[0]["changes"]["num_layers"] == 4


def test_legacy_rules_reproduce_old_choice():
    loads = [[10] * 6, [12, 12, 12, 0, 12]]
    cfg = cfg_hist([rec([6, 5], loads=loads)])
    cfg["rules"].update(expand.LEGACY_GROWTH_RULES)
    assert expand.plan_mutations(cfg)[0]["changes"]["num_experts"] == [6, 6]


def test_cycle_record_has_load_entropy(tmp_path):
    for f in ("moe.py", "transformer.py", "model.py", "expand.py"):
        shutil.copy(os.path.join(ROOT, f), tmp_path / f)
    expand.init_config(str(tmp_path), overrides={**FAST, "num_layers": 2, "num_experts": [4, 5]})
    r = expand.run_cycle(str(tmp_path))
    assert len(r["expert_load_entropy"]) == 2 and len(r["expert_min_load_ratio"]) == 2
    for l, h, q in zip(r["expert_load"], r["expert_load_entropy"], r["expert_min_load_ratio"]):
        assert h == pytest.approx(expert_load_entropy(l), abs=1e-6)
        assert q == pytest.approx(expert_min_load_ratio(l), abs=1e-6)


# ------------------------------------------------------------------ fix 2: me over all top-k slots
def test_expand_model_uses_fixed_moe_path():
    cfg = {**copy.deepcopy(expand.DEFAULT_CONFIG), "num_layers": 2, "num_experts": [4, 4]}
    model, _, _ = expand.build_model("/nonexistent-root-for-test", cfg)
    assert all(not b.moe.legacy for b in model.blocks)


@pytest.mark.parametrize("dispatch", ["loop", "vectorized"])
def test_me_counts_second_slot(dispatch):
    torch.manual_seed(0)
    m = CapacitySparseMoELayer(8, 16, 4, top_k=2, capacity_factor=None, dispatch=dispatch).eval()
    x = torch.randn(64, 8)
    _, aux, _ = m(x)
    logits = m.gate(x)
    idx = logits.topk(2, -1).indices
    pe = F.softmax(logits, -1).mean(0)
    me_all = F.one_hot(idx, 4).float().sum(1).mean(0) / 2
    me_top1 = F.one_hot(idx[:, 0], 4).float().mean(0)
    assert torch.allclose(aux, 4 * (me_all * pe).sum(), atol=1e-6)
    assert not torch.allclose(me_all, me_top1)  # the second slot really changes the load estimate
    assert not torch.allclose(aux, 4 * (me_top1 * pe).sum(), atol=1e-6)


# ------------------------------------------------------------------ fix 3: convergence floor / patience
def simulate(rules_update, cycles=40, loss_fn=lambda c, n_growth: 0.03):
    """Drive plan_mutations + apply_mutation on synthetic records: every cycle reports rel_improvement 0
    (stalled, as a converged model does), balanced load, and eval loss loss_fn(cycle, growths_so_far)."""
    cfg = copy.deepcopy(expand.DEFAULT_CONFIG)
    cfg["caps"] = {"capacity_factor_max": 3.0, "num_experts_max": 16, "alpha_max": 1.0, "num_layers_max": 16}
    cfg["rules"].update(rules_update)
    cfg["history"] = [rec([4], loss=loss_fn(0, 0))]
    growths = 0
    for c in range(1, cycles):
        before = list(cfg["num_experts"])
        mut = expand.plan_mutations(cfg)[0]
        cfg = expand.apply_mutation(cfg, mut)
        grew = cfg["num_experts"] != before
        growths += grew
        cfg["history"].append(rec(cfg["num_experts"], loss=loss_fn(c, growths), before=before, mutation=mut))
    return growths, cfg


def test_legacy_stall_grows_every_converged_cycle():
    growths, _ = simulate(expand.LEGACY_GROWTH_RULES)
    assert growths == 39  # the runaway: rel_improvement < 0.01 forever -> growth every single cycle


def test_converged_model_stops_growing():
    growths, cfg = simulate({})  # flat eval loss: no growth ever pays
    # grow expert (no baseline yet) ... wait 3 ... unpaid -> layer ... wait 3 ... unpaid layer -> converged
    assert growths <= 4  # expert, layer, then (after growth_cooldown=25) one more expert + layer try
    last_reasons = expand.plan_mutations(cfg)[-1]["reasons"]
    assert any("converged" in r or "cooldown" in r or "waiting" in r for r in last_reasons)


def test_cooldown_allows_a_retry():
    growths_short, _ = simulate({"growth_cooldown": 1000}, cycles=80)
    growths_long, _ = simulate({"growth_cooldown": 10}, cycles=80)
    assert growths_long > growths_short


def test_paying_growth_keeps_growing():
    growths, _ = simulate({}, loss_fn=lambda c, n: 0.03 * (0.9 ** n))  # every growth gains 10%
    assert 8 <= growths <= 14  # once per stall_patience (3) cycles, never every cycle


def test_settling_blocks_growth_right_after_growth():
    h = [rec([4], loss=0.03), rec([4], loss=0.03), rec([5], loss=0.03, before=[4])]
    cfg = cfg_hist(h)
    cands = expand.plan_mutations(cfg)
    assert cands[0]["name"] == "none" and any("waiting" in r for r in cands[0]["reasons"])


def test_verdict_ignores_post_growth_rel_improvement():
    """The decision uses eval losses before vs after growth, so the optimizer restart's effect on the
    growth cycle's own rel_improvement cannot drive the next growth."""
    base = [rec([4], loss=0.030), rec([4], loss=0.030), rec([4], loss=0.030)]
    for rel in (-0.5, 0.0, 0.5):
        post = [rec([5], loss=0.029, before=[4], rel=rel), rec([5], loss=0.029), rec([5], loss=0.029)]
        v, _ = expand.growth_verdict(base + post, expand.GROWTH_RULE_DEFAULTS)
        assert v == "paid"
        post = [rec([5], loss=0.0299, before=[4], rel=rel), rec([5], loss=0.0299), rec([5], loss=0.0299)]
        v, _ = expand.growth_verdict(base + post, expand.GROWTH_RULE_DEFAULTS)
        assert v == "unpaid_expert"


def test_unpaid_expert_escalates_to_layer_then_converges():
    base = [rec([4], loss=0.03)] * 3
    h = base + [rec([5], loss=0.03, before=[4])] + [rec([5], loss=0.03)] * 2
    m = expand.plan_mutations(cfg_hist(h))[0]
    assert m["changes"].get("num_layers") == 2
    h2 = h + [rec([5, 4], loss=0.03, before=[5])] + [rec([5, 4], loss=0.03)] * 2
    assert expand.plan_mutations(cfg_hist(h2))[0]["name"] == "none"


def test_loss_floor_stops_growth():
    cfg = cfg_hist([rec([4], loss=0.001)])
    assert expand.plan_mutations(cfg)[0]["changes"] == {"num_experts": [5]}
    cfg["rules"]["loss_floor"] = 0.002
    assert expand.plan_mutations(cfg)[0]["name"] == "none"


# ------------------------------------------------------------------ fix 4: new-expert initialisation
def _routed(layer, x, e):
    idx = layer.gate(x).topk(layer.top_k, -1).indices
    return (idx == e).any(-1)


def test_clone_of_most_loaded_duplicates_its_tokens_mean_init_does_not():
    torch.manual_seed(0)
    x = torch.randn(2048, 16)
    base = CapacitySparseMoELayer(16, 32, 6, top_k=2, capacity_factor=None).eval()
    with torch.no_grad():
        base.gate.weight[2] += 0.8 * base.gate.weight[2] / base.gate.weight[2].norm() * 3  # saturate expert 2
    counts = torch.bincount(base.gate(x).topk(2, -1).indices.flatten(), minlength=6)
    src = int(counts.argmax())
    assert src == 2
    copy_l, mean_l = copy.deepcopy(base), copy.deepcopy(base)
    g = lambda: torch.Generator().manual_seed(0)
    nc = copy_l.add_expert(src, 1e-2, g(), init="copy")
    nm = mean_l.add_expert(None, 1e-2, g(), init="mean")
    src_tokens = _routed(base, x, src)
    clone_tokens = _routed(copy_l, x, nc)
    # the clone takes (almost) exactly the saturated expert's tokens: top-2 now picks src AND its twin
    overlap_clone = (clone_tokens & src_tokens).sum() / clone_tokens.sum()
    assert overlap_clone > 0.95 and clone_tokens.sum() > 0.6 * src_tokens.sum()
    # and the saturated expert stays saturated (its twin displaces OTHER experts' second slots)
    assert _routed(copy_l, x, src).sum() >= 0.6 * src_tokens.sum()
    # the mean-init expert does not inherit the saturated expert's token set
    mean_tokens = _routed(mean_l, x, nm)
    assert mean_tokens.sum() < 0.5 * src_tokens.sum()


def test_mean_init_is_layer_average_and_shapes_grow():
    torch.manual_seed(1)
    layer = CapacitySparseMoELayer(8, 16, 3, top_k=2, capacity_factor=None)
    w_before = layer.gate.weight.detach().clone()
    exp_mean = torch.stack([e.w1.weight for e in layer.experts]).mean(0).detach()
    n = layer.add_expert(None, 0.0, None, init="mean")
    assert n == 3 and layer.num_experts == 4 and layer.gate.weight.shape == (4, 8)
    assert torch.allclose(layer.gate.weight[:3], w_before)                      # trained rows untouched
    assert torch.allclose(layer.gate.weight[3], w_before.mean(0), atol=1e-6)
    assert torch.allclose(layer.experts[3].w1.weight, exp_mean, atol=1e-6)
    with pytest.raises(ValueError):
        layer.add_expert(None, 0.0, None, init="bogus")


def _original_add_expert(layer, source_idx, noise_std, generator):
    """The pre-fix CapacitySparseMoELayer.add_expert, verbatim (reference for bit-identity)."""
    from moe import Expert
    import torch.nn as nn
    with torch.no_grad():
        src = layer.experts[source_idx]
        new = Expert(layer.d_model, layer.d_ff)
        new.load_state_dict(src.state_dict())
        for p in new.parameters():
            p.add_(torch.randn(p.shape, generator=generator, dtype=p.dtype) * noise_std)
        layer.experts.append(new)
        old_w = layer.gate.weight.data
        row = old_w[source_idx:source_idx + 1].clone()
        row.add_(torch.randn(row.shape, generator=generator, dtype=row.dtype) * noise_std)
        g = nn.Linear(layer.d_model, layer.num_experts + 1, bias=False)
        g.weight.data.copy_(torch.cat([old_w, row], dim=0))
        layer.gate = g
        layer.num_experts += 1


def test_copy_init_is_bit_identical_to_original():
    torch.manual_seed(2)
    a = CapacitySparseMoELayer(8, 16, 3, top_k=2)
    b = copy.deepcopy(a)
    _original_add_expert(a, 1, 1e-2, torch.Generator().manual_seed(5))
    b.add_expert(1, 1e-2, torch.Generator().manual_seed(5), init="copy")
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.parametrize("init,needle", [(None, "mean of the layer's experts"),
                                         ("most_loaded", "copy of most-loaded expert")])
def test_build_model_growth_uses_configured_init(tmp_path, init, needle):
    for f in ("moe.py", "transformer.py", "model.py", "expand.py"):
        shutil.copy(os.path.join(ROOT, f), tmp_path / f)
    root = str(tmp_path)
    over = {**FAST, "num_layers": 1, "num_experts": [4]}
    expand.init_config(root, overrides=over)
    if init:
        cfg = expand.load_config(root); cfg["rules"]["expert_init"] = init; expand.save_config(root, cfg)
    expand.run_cycle(root)
    cfg = expand.load_config(root); cfg["num_experts"] = [5]; expand.save_config(root, cfg)
    r = expand.run_cycle(root)
    assert r["num_experts"] == [5] and any(needle in g for g in r["grown"])
