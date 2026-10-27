"""Tests for the 2026-10-27 expand-loop upgrades (lr schedule, optimizer carry-over, EMA, device, evaluate)."""
import copy, json, os
import pytest
import torch
import expand
import evaluate

FAST = {"steps_per_cycle": 20, "batch_size": 4, "seq_len": 8, "eval_batch_size": 4}


def fresh(tmp_path, **over):
    cfg = expand.init_config(str(tmp_path), force=True, overrides={**FAST, **over})
    return str(tmp_path), cfg


def test_lr_constant_is_default():
    cfg = copy.deepcopy(expand.DEFAULT_CONFIG)
    assert all(expand.lr_at(cfg, s) == cfg["lr"] for s in range(0, 300, 37))


def test_lr_cosine_warmup_and_decay():
    cfg = {**copy.deepcopy(expand.DEFAULT_CONFIG), "lr_schedule": "cosine", "warmup_steps": 10, "lr_min": 1e-5}
    assert expand.lr_at(cfg, 0) == pytest.approx(cfg["lr"] / 10)
    assert expand.lr_at(cfg, 9) == pytest.approx(cfg["lr"])
    assert expand.lr_at(cfg, 10) == pytest.approx(cfg["lr"])
    assert expand.lr_at(cfg, cfg["steps_per_cycle"]) == pytest.approx(1e-5)
    vals = [expand.lr_at(cfg, s) for s in range(10, cfg["steps_per_cycle"])]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_optimizer_state_carried_on_growth(tmp_path):
    root, cfg = fresh(tmp_path, carry_optimizer_on_growth=True)
    expand.run_cycle(root)
    cfg = expand.load_config(root)
    cfg = expand.apply_mutation(cfg, {"name": "grow", "changes": {"num_experts": [5]}, "reasons": []})
    expand.save_config(root, cfg)
    rec = expand.run_cycle(root)
    assert rec["optimizer_reset"] and rec["optimizer_carried"] > 0
    # gate grew 4->5 rows, new expert is new: those are the only params without carried state
    n_params = sum(1 for _ in torch.load(expand.paths(root)["ckpt"], weights_only=False)["model"])
    assert rec["optimizer_carried"] < n_params


def test_optimizer_not_carried_by_default(tmp_path):
    root, cfg = fresh(tmp_path)
    expand.run_cycle(root)
    cfg = expand.apply_mutation(expand.load_config(root), {"name": "grow", "changes": {"num_experts": [5]}, "reasons": []})
    expand.save_config(root, cfg)
    rec = expand.run_cycle(root)
    assert rec["optimizer_reset"] and rec["optimizer_carried"] == 0


def test_ema_saved_and_scored(tmp_path):
    root, cfg = fresh(tmp_path, ema_decay=0.9, eval_extra=True)
    rec = expand.run_cycle(root)
    assert "ema_loss" in rec and "loss_dropless" in rec
    ck = torch.load(expand.paths(root)["ckpt"], weights_only=False)
    assert ck["ema"] is not None and set(ck["ema"]) == set(ck["model"])
    res = evaluate.main(["--root", root, "--n-seqs", "16", "--device", "cpu"])
    for name in ("raw", "ema"):
        r = res[name]
        assert 0 <= r["heldout_mse"] and r["baseline_zero_mse"] > 0 and "|x|>=2" in r["bucket_mse"]


def test_checkpoint_is_cpu_and_device_cpu_default(tmp_path):
    root, _ = fresh(tmp_path)
    rec = expand.run_cycle(root)
    assert rec["device"] == "cpu"
    ck = torch.load(expand.paths(root)["ckpt"], weights_only=False)
    assert all(v.device.type == "cpu" for v in ck["model"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_cycle_with_growth(tmp_path):
    root, _ = fresh(tmp_path, device="cuda", carry_optimizer_on_growth=True, ema_decay=0.9, eval_extra=True)
    expand.run_cycle(root)
    cfg = expand.apply_mutation(expand.load_config(root),
                                {"name": "grow", "changes": {"num_experts": [5], }, "reasons": []})
    expand.save_config(root, cfg)
    rec = expand.run_cycle(root)
    assert rec["device"].startswith("cuda") and rec["grown"]
    cfg = expand.apply_mutation(expand.load_config(root),
                                {"name": "grow", "changes": {"num_layers": 2, "num_experts": [5, 4]}, "reasons": []})
    expand.save_config(root, cfg)
    assert expand.run_cycle(root)["num_layers"] == 2
