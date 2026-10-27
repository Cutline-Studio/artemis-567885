"""Stage 8b: physics sandbox options (lr schedule, noisy routing, causal-in-time training,
curriculum, expert-usage metric, error vs t). All options are off by default."""
import inspect, json, math, os
import pytest
import torch
import physics_simulation as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_train_defaults_are_off():
    sig = inspect.signature(P.train).parameters
    assert sig["lr"].default == 1e-3 and sig["alpha"].default == 0.01
    assert sig["lr_schedule"].default == "constant" and sig["noise_std"].default == 0.0
    assert sig["causal_eps"].default == 0.0 and sig["curriculum_frac"].default == 0.0
    assert P.PhysicsMoELayer(4, 2, hidden=8).router.noise_std == 0.0


def test_defaults_bit_identical_to_explicit_off_options():
    kw = dict(seconds=60, max_steps=25, n=128, hidden=16, log_every=10)
    _, a, ha = P.train(**kw)
    _, b, hb = P.train(**kw, lr_schedule="constant", noise_std=0.0, causal_eps=0.0, curriculum_frac=0.0)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("seconds", "sec")}
    assert strip(a) == strip(b) and [strip(h) for h in ha] == [strip(h) for h in hb]


def test_default_path_matches_pre_stage8b_loss_formula():
    """losses() with default args = plain mean residual over the full t range (no reweighting)."""
    torch.manual_seed(0)
    m = P.PhysicsMoELayer(4, 2, hidden=16)
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    l_pde = P.losses(m, 64, g1)[0]
    r, _ = P.pde_residual(m, P.sample(64, g2))
    assert torch.equal(l_pde, r.pow(2).mean())


def test_iteration1_baseline_artifact_reproduced():
    """runs/stage8b/baseline_8thr reran the defaults (8 threads, 6358 steps) and matched runs/physics."""
    a = json.load(open(os.path.join(ROOT, "runs", "physics", "final.json")))
    b = json.load(open(os.path.join(ROOT, "runs", "stage8b", "baseline_8thr", "final.json")))
    for k in ("rel_l2_t0.0", "rel_l2_t0.5", "rel_l2_t1.0", "rel_l2_all", "pde_residual_mse", "expert_usage_top1_total"):
        assert a[k] == b[k], k


@pytest.mark.parametrize("step,expect", [(0, 1e-3), (50, 5e-4), (100, 0.0), (25, 1e-3 * 0.5 * (1 + math.cos(math.pi / 4)))])
def test_cosine_schedule_values(step, expect):
    assert P.lr_at(step, 100, 1e-3, "cosine") == pytest.approx(expect, abs=1e-12)


def test_schedule_constant_and_floor_and_bad_name():
    assert P.lr_at(77, 100, 3e-4) == 3e-4
    assert P.lr_at(100, 100, 1e-3, "cosine", min_lr_frac=0.1) == pytest.approx(1e-4)
    assert P.lr_at(500, 100, 1e-3, "cosine", min_lr_frac=0.1) == pytest.approx(1e-4)  # clamps past the end
    with pytest.raises(ValueError):
        P.lr_at(0, 10, 1e-3, "linear")


def test_cosine_schedule_logged_in_training():
    _, _, hist = P.train(seconds=60, max_steps=20, n=64, hidden=8, log_every=10, lr_schedule="cosine")
    lrs = {h["step"]: h["lr"] for h in hist}
    # lr logged after the step that used it: step k (1-based) used lr_at(k-1)
    assert lrs[10] == pytest.approx(P.lr_at(9, 20, 1e-3, "cosine"))
    assert lrs[20] == pytest.approx(P.lr_at(19, 20, 1e-3, "cosine"))


def test_noisy_routing_only_in_train_mode():
    torch.manual_seed(0)
    m = P.PhysicsMoELayer(4, 2, hidden=8, noise_std=5.0)
    x = torch.rand(400, 3)
    m.eval()
    w_a, i_a, p_a = m.router(x); w_b, i_b, p_b = m.router(x)
    assert torch.equal(i_a, i_b) and torch.equal(w_a, w_b)  # deterministic in eval
    m.train()
    _, i_c, p_c = m.router(x); _, i_d, p_d = m.router(x)
    assert not torch.equal(i_c, i_d)  # noise changes routing in train mode
    assert torch.equal(p_c, p_a) and torch.equal(p_d, p_a)  # aux probs use the clean logits
    clean = P.PhysicsMoELayer(4, 2, hidden=8); clean.load_state_dict(m.state_dict())
    clean.train()
    assert torch.equal(clean.router(x)[1], i_a)  # noise_std 0 => train mode is clean


def test_noise_does_not_consume_rng_when_off():
    m = P.PhysicsMoELayer(4, 2, hidden=8)
    torch.manual_seed(5); m(torch.rand(10, 3)); a = torch.rand(1)
    torch.manual_seed(5); torch.rand(10, 3); b = torch.rand(1)
    assert torch.equal(a, b)


def test_evaluate_restores_mode_and_is_noise_free():
    torch.manual_seed(0)
    m = P.PhysicsMoELayer(4, 2, hidden=8, noise_std=3.0)
    m.train()
    a = P.evaluate(m, res=9); b = P.evaluate(m, res=9)
    assert m.training and a == b
    e1 = P.error_vs_t(m, [0.0, 1.0], res=9); e2 = P.error_vs_t(m, [0.0, 1.0], res=9)
    assert m.training and e1 == e2


def test_expert_usage_metric():
    assert P.expert_usage_stats([0, 0, 1129, 8132]) == {"experts_used": 2, "usage_entropy": pytest.approx(0.267412, abs=1e-6)}
    assert P.expert_usage_stats([5, 5, 5, 5]) == {"experts_used": 4, "usage_entropy": 1.0}
    assert P.expert_usage_stats([9261, 0, 0, 0]) == {"experts_used": 1, "usage_entropy": 0.0}
    torch.manual_seed(0)
    out = P.evaluate(P.PhysicsMoELayer(4, 2, hidden=8), res=9)
    assert out["experts_used"] == sum(c > 0 for c in out["expert_usage_top1_total"])


def test_causal_weights_math():
    L = torch.tensor([0.5, 0.2, 0.1, 0.0])
    w = P.causal_weights(L, 2.0)
    expect = torch.exp(-2.0 * torch.tensor([0.0, 0.5, 0.7, 0.8]))
    assert torch.allclose(w, expect)
    assert w[0] == 1.0 and (w[1:] <= w[:-1]).all()  # monotone non-increasing
    assert torch.equal(P.causal_weights(L, 0.0), torch.ones(4))
    Lg = L.clone().requires_grad_(True)
    assert not P.causal_weights(Lg, 1.0).requires_grad  # weights carry no gradient


def test_causal_pde_loss_binning_and_reference():
    t = torch.tensor([0.05, 0.1, 0.3, 0.6, 0.7, 0.99, 1.0])
    r = torch.tensor([1.0, 3.0, 2.0, 1.0, 1.0, 0.5, 0.5]).unsqueeze(1)
    loss, w, L = P.causal_pde_loss(r, t, eps=1.0, n_bins=4)
    # bins of width 0.25: [0.05,0.1] -> 0, [0.3] -> 1, [0.6,0.7] -> 2, [0.99, 1.0] -> 3 (t=1 clamped)
    L_ref = torch.tensor([(1 + 9) / 2, 4.0, 1.0, 0.25])
    assert torch.allclose(L, L_ref)
    w_ref = torch.exp(-torch.tensor([0.0, 5.0, 9.0, 10.0]))
    assert torch.allclose(w, w_ref)
    assert torch.allclose(loss, (w_ref * L_ref).mean())
    # eps = 0 => mean of per-bin means; with t_max the bins cover [0, t_max]
    l0, w0, _ = P.causal_pde_loss(r, t * 0.5, eps=0.0, n_bins=4, t_max=0.5)
    assert torch.allclose(l0, L_ref.mean()) and torch.equal(w0, torch.ones(4))


def test_causal_loss_gradient_flows_only_through_bin_losses():
    torch.manual_seed(0)
    m = P.PhysicsMoELayer(4, 2, hidden=8)
    pts = P.sample(128, torch.Generator().manual_seed(0))
    r, _ = P.pde_residual(m, pts)
    loss, w, _ = P.causal_pde_loss(r, pts[:, 2], eps=10.0)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_curriculum_tmax_and_sampling():
    assert P.curriculum_tmax(0, 100, 0.0) == 1.0
    assert P.curriculum_tmax(0, 100, 0.5, 0.1) == pytest.approx(0.1)
    assert P.curriculum_tmax(25, 100, 0.5, 0.1) == pytest.approx(0.55)
    assert P.curriculum_tmax(50, 100, 0.5, 0.1) == 1.0 and P.curriculum_tmax(90, 100, 0.5) == 1.0
    _, _, hist = P.train(seconds=60, max_steps=20, n=64, hidden=8, log_every=10, curriculum_frac=0.5)
    assert hist[0]["t_max"] == pytest.approx(0.1) and hist[-1]["t_max"] == 1.0


def test_new_options_train_and_stay_finite():
    _, f, _ = P.train(seconds=60, max_steps=15, n=64, hidden=8, log_every=5, lr=1e-4, lr_schedule="cosine",
                      alpha=0.1, noise_std=1.0, causal_eps=1.0, curriculum_frac=0.5)
    assert f["steps"] == 15 and all(math.isfinite(f[k]) for k in ("pde_residual_mse", "rel_l2_t1.0"))
    assert sum(f["expert_usage_top1_total"]) == 21 ** 3


def test_error_vs_t_matches_evaluate_at_shared_times():
    torch.manual_seed(1)
    m = P.PhysicsMoELayer(4, 2, hidden=8)
    ev = P.evaluate(m)
    rows = P.error_vs_t(m)
    assert [r["t"] for r in rows] == [round(i / 10, 4) for i in range(11)]
    by_t = {r["t"]: r for r in rows}
    for t in (0.0, 0.5, 1.0):
        assert by_t[t]["rel_l2"] == pytest.approx(ev[f"rel_l2_t{t:.1f}"], abs=2e-6)
        assert by_t[t]["max_abs_err"] == pytest.approx(ev[f"max_abs_err_t{t:.1f}"], abs=2e-6)


def test_cli_flags_parse_defaults(monkeypatch, tmp_path):
    seen = {}
    def fake_train(*a, **kw):
        seen.update(kw); return None, {"ok": 1}, []
    monkeypatch.setattr(P, "train", fake_train)
    monkeypatch.setattr("sys.argv", ["physics_simulation.py", "--out", str(tmp_path)])
    P.main()
    assert seen["lr"] == 1e-3 and seen["alpha"] == 0.01 and seen["lr_schedule"] == "constant"
    assert seen["noise_std"] == 0.0 and seen["causal_eps"] == 0.0 and seen["curriculum_frac"] == 0.0
    assert seen["threads"] is None and seen["seed"] == 567885
