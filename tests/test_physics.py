import math
import torch
import physics_simulation as P


def test_siren_init_ranges():
    first = P.SineLayer(3, 64, is_first=True, omega0=30.0)
    hid = P.SineLayer(64, 64, omega0=30.0)
    assert first.linear.weight.abs().max() <= 1 / 3 + 1e-7
    b = math.sqrt(6 / 64) / 30
    assert hid.linear.weight.abs().max() <= b + 1e-7
    assert hid.linear.weight.abs().max() > 0.5 * b  # actually spans the range


def test_shapes_and_routing():
    torch.manual_seed(0)
    m = P.PhysicsMoELayer(4, 2, hidden=16)
    h, aux, idx = m(torch.rand(50, 3))
    assert h.shape == (50, 1) and idx.shape == (50, 2) and torch.isfinite(aux)


def test_inplace_and_index_add_agree_with_second_order_grads():
    torch.manual_seed(0)
    a = P.PhysicsMoELayer(4, 2, hidden=16, scatter="index_add")
    b = P.PhysicsMoELayer(4, 2, hidden=16, scatter="inplace"); b.load_state_dict(a.state_dict())
    coords = P.sample(64, torch.Generator().manual_seed(0))
    grads = []
    for m in (a, b):
        r, _ = P.pde_residual(m, coords)
        m.zero_grad(); r.pow(2).mean().backward()  # backprop through d h / d coords (create_graph)
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
        assert torch.isfinite(grads[-1]).all() and grads[-1].norm() > 0
    ra, _ = P.pde_residual(a, coords); rb, _ = P.pde_residual(b, coords)
    assert torch.allclose(ra, rb, atol=1e-6)
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_pde_only_loss_has_trivial_constant_solution_but_ic_rejects_it():
    class Const(torch.nn.Module):
        num_experts = 1
        def forward(self, c):
            return c[:, :1] * 0 + 0.3, torch.tensor(1.0), torch.zeros(c.shape[0], 1, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    l_pde, l_ic, l_bc, *_ = P.losses(Const(), 256, gen)
    assert l_pde.item() == 0.0  # residual is exactly zero for a constant field
    assert l_ic.item() > 1e-2   # the initial-condition term is not


def test_exact_solution_has_zero_residual():
    c = P.sample(128, torch.Generator().manual_seed(1)).requires_grad_(True)
    h = P.exact(c[:, :1], c[:, 1:2], c[:, 2:3])
    g = torch.autograd.grad(h.sum(), c)[0]
    assert (g[:, 2] + P.C * (g[:, 0] + g[:, 1])).abs().max() < 1e-5


def test_short_training_reduces_loss():
    _, final, hist = P.train(seconds=30, max_steps=60, n=256, hidden=32, log_every=59)
    assert final["steps"] == 60
    first, last = hist[0], hist[-1]
    assert last["ic"] + last["bc"] < first["ic"] + first["bc"]
    assert sum(final["expert_usage_top1_total"]) == 21 ** 3
