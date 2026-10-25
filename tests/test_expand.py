import copy, json, os, shutil
import pytest
import torch
import expand

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = {"steps_per_cycle": 20, "batch_size": 4, "seq_len": 8, "eval_batch_size": 4}


def cfg_with(last, **over):
    cfg = copy.deepcopy(expand.DEFAULT_CONFIG)
    cfg.update(over)
    base = {"dropped": 0, "aux": 1.0, "rel_improvement": 0.5, "mutation": {"name": "none", "changes": {}}}
    base.update(last)
    cfg["history"] = [base]
    return cfg


def first(cfg):
    return expand.plan_mutations(cfg)[0]


def test_no_history_is_noop():
    assert first(copy.deepcopy(expand.DEFAULT_CONFIG))["name"] == "none"


def test_dropped_raises_capacity_factor():
    m = first(cfg_with({"dropped": 3}))
    assert m["changes"] == {"capacity_factor": 0.85}


def test_capacity_factor_capped():
    m = first(cfg_with({"dropped": 3}, capacity_factor=2.48))
    assert m["changes"]["capacity_factor"] == 2.5
    assert first(cfg_with({"dropped": 3}, capacity_factor=2.5))["name"] == "none"
    # the old 2.0 ceiling no longer blocks the +0.05 step
    assert first(cfg_with({"dropped": 3}, capacity_factor=2.0))["changes"] == {"capacity_factor": 2.05}


def test_high_aux_raises_alpha_with_cap():
    assert first(cfg_with({"aux": 1.3}))["changes"] == {"alpha": 0.15}
    assert first(cfg_with({"aux": 1.3}, alpha=0.8))["changes"] == {"alpha": 1.0}
    assert first(cfg_with({"aux": 1.3}, alpha=1.0))["name"] == "none"
    assert first(cfg_with({"aux": 1.2}))["name"] == "none"  # strictly greater than 1.2


def test_stall_adds_expert():
    m = first(cfg_with({"rel_improvement": 0.005}))
    assert m["changes"] == {"num_experts": [5]}


def test_stall_after_expert_addition_adds_layer():
    last = {"rel_improvement": 0.001, "mutation": {"name": "structural+soft", "changes": {"num_experts": [5]}}}
    m = first(cfg_with(last, num_experts=[5]))
    assert m["changes"] == {"num_layers": 2, "num_experts": [5, 4]}


def test_expert_cap_triggers_layer_and_layer_cap_stops():
    m = first(cfg_with({"rel_improvement": 0.0}, num_experts=[16]))
    assert m["changes"]["num_layers"] == 2
    cfg = cfg_with({"rel_improvement": 0.0}, num_experts=[16] * 12, num_layers=12)
    assert first(cfg)["name"] == "none"


def test_candidates_offer_different_paths_and_end_with_noop():
    c = expand.plan_mutations(cfg_with({"dropped": 2, "aux": 1.5, "rel_improvement": 0.0}))
    names = [x["name"] for x in c]
    assert names == ["structural+soft", "soft", "soft_half_step", "none"]
    assert c[2]["changes"]["capacity_factor"] == pytest.approx(0.825)


def test_apply_mutation_enforces_caps():
    cfg = copy.deepcopy(expand.DEFAULT_CONFIG)
    out = expand.apply_mutation(cfg, {"name": "x", "changes": {"capacity_factor": 9, "alpha": 9,
                                                               "num_experts": [99], "num_layers": 99}, "reasons": []})
    assert out["capacity_factor"] == 2.5 and out["alpha"] == 1.0 and out["num_layers"] == 12
    assert out["num_experts"] == [16]


def test_default_caps_are_the_approved_limits():
    caps = expand.DEFAULT_CONFIG["caps"]
    assert caps["capacity_factor_max"] == 2.5
    assert caps["num_layers_max"] == 12
    assert caps["num_experts_max"] == 16  # per layer
    repo = json.load(open(os.path.join(ROOT, "config.json")))["caps"]
    assert (repo["capacity_factor_max"], repo["num_layers_max"], repo["num_experts_max"]) == (2.5, 12, 16)


LIVE = [6, 9, 5, 8, 5, 5, 5, 19]  # layer 7 grew to 19 before the per-layer cap of 16 existed


def test_stall_skips_capped_layer_and_grows_deepest_open_layer():
    m = first(cfg_with({"rel_improvement": 0.0}, num_experts=list(LIVE), num_layers=8))
    assert m["changes"]["num_experts"] == [6, 9, 5, 8, 5, 5, 6, 19]  # layer 6, not layer 7
    assert "num_layers" not in m["changes"]
    # layer 6 also full -> next open layer down (layer 5)
    m = first(cfg_with({"rel_improvement": 0.0}, num_experts=[6, 9, 5, 8, 5, 5, 16, 19], num_layers=8))
    assert m["changes"]["num_experts"] == [6, 9, 5, 8, 5, 6, 16, 19]


def test_stall_never_adds_to_layer_at_or_above_cap():
    import random
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 12)
        ex = [rng.randint(4, 20) for _ in range(n)]
        cfg = cfg_with({"rel_improvement": 0.0}, num_experts=list(ex), num_layers=n)
        ch = first(cfg)["changes"]
        if "num_experts" in ch and "num_layers" not in ch:
            diff = [i for i, (a, b) in enumerate(zip(ex, ch["num_experts"])) if a != b]
            assert len(diff) == 1 and ex[diff[0]] < 16 and ch["num_experts"][diff[0]] == ex[diff[0]] + 1


def test_all_layers_capped_adds_layer_until_layer_cap():
    m = first(cfg_with({"rel_improvement": 0.0}, num_experts=[16] * 7 + [19], num_layers=8))
    assert m["changes"] == {"num_layers": 9, "num_experts": [16] * 7 + [19, 4]}
    m = first(cfg_with({"rel_improvement": 0.0}, num_experts=[16] * 11 + [19], num_layers=12))
    assert m["name"] == "none"


def test_stall_after_expert_addition_still_adds_layer_with_new_layer_cap():
    last = {"rel_improvement": 0.0, "mutation": {"name": "structural+soft", "changes": {"num_experts": LIVE}}}
    m = first(cfg_with(last, num_experts=list(LIVE), num_layers=8))
    assert m["changes"] == {"num_layers": 9, "num_experts": LIVE + [4]}
    m = first(cfg_with(last, num_experts=list(LIVE) + [4] * 4, num_layers=12))
    assert m["changes"]["num_experts"] == LIVE + [4, 4, 4, 5]  # layer cap hit -> falls back to an open layer


def test_apply_mutation_keeps_trained_experts_above_cap():
    cfg = cfg_with({"rel_improvement": 0.0}, num_experts=list(LIVE), num_layers=8)
    out = expand.apply_mutation(cfg, first(cfg))
    assert out["num_experts"][7] == 19 and out["num_experts"][6] == 6  # no pruning of layer 7
    # an attempted growth on a capped layer is clamped back to its current size, not to 16
    out = expand.apply_mutation(cfg, {"name": "x", "changes": {"num_experts": LIVE[:7] + [25]}, "reasons": []})
    assert out["num_experts"][7] == 19


def test_capped_layer_checkpoint_loads_and_grows_other_layer(tmp_path):
    for f in ("moe.py", "transformer.py", "model.py", "expand.py"):
        shutil.copy(os.path.join(ROOT, f), tmp_path / f)
    root = str(tmp_path)
    expand.init_config(root, overrides={**FAST, "num_layers": 2, "num_experts": [4, 18]})
    expand.run_cycle(root)
    cfg = expand.load_config(root)
    cfg["history"][-1]["rel_improvement"] = 0.0
    cfg["history"][-1]["mutation"] = {"name": "none", "changes": {}}
    m = expand.plan_mutations(cfg)[0]
    assert m["changes"]["num_experts"] == [5, 18]
    expand.save_config(root, expand.apply_mutation(cfg, m))
    r = expand.run_cycle(root)
    assert r["num_experts"] == [5, 18] and r["output_shape_ok"]


def test_cycle_persists_state_resumes_and_grows(tmp_path):
    for f in ("moe.py", "transformer.py", "model.py", "expand.py"):
        shutil.copy(os.path.join(ROOT, f), tmp_path / f)
    root = str(tmp_path)
    expand.init_config(root, overrides=FAST)
    r0 = expand.run_cycle(root)
    assert r0["output_shape_ok"] and r0["cycle"] == 0
    cfg = expand.load_config(root)
    assert cfg["cycle"] == 1 and len(cfg["history"]) == 1
    lines = open(tmp_path / "runs" / "expand_log.jsonl").read().splitlines()
    assert json.loads(lines[0])["cycle"] == 0
    # grow by one expert and one layer via config only; trained weights must be kept
    before = torch.load(tmp_path / "runs" / "checkpoint.pt", weights_only=False)["model"]
    cfg["num_experts"] = [5, 4]; cfg["num_layers"] = 2
    expand.save_config(root, cfg)
    r1 = expand.run_cycle(root)
    assert r1["num_experts"] == [5, 4] and r1["num_layers"] == 2 and r1["optimizer_reset"]
    assert any("layer 1 added" in g for g in r1["grown"])
    after = torch.load(tmp_path / "runs" / "checkpoint.pt", weights_only=False)
    assert after["arch"]["num_experts"] == [5, 4]
    assert set(before).issubset(set(after["model"]))
