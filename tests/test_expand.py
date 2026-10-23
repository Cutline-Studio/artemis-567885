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
    m = first(cfg_with({"dropped": 3}, capacity_factor=1.98))
    assert m["changes"]["capacity_factor"] == 2.0
    assert first(cfg_with({"dropped": 3}, capacity_factor=2.0))["name"] == "none"


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
    m = first(cfg_with({"rel_improvement": 0.0}, num_experts=[32]))
    assert m["changes"]["num_layers"] == 2
    cfg = cfg_with({"rel_improvement": 0.0}, num_experts=[32] * 8, num_layers=8)
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
    assert out["capacity_factor"] == 2.0 and out["alpha"] == 1.0 and out["num_layers"] == 8
    assert out["num_experts"] == [32]


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
