"""Check 4a: validate the finite-difference solver (physics_solver.py) against
the analytic travelling-wave solution and its convergence order under grid
refinement (dx halves each level, CFL ~0.4, t_end = 1). Thresholds stated here,
before running: Lax-Wendroff observed order on the two finest refinements in
[1.8, 2.2]; upwind observed order on the finest refinement in [0.8, 1.2]."""
import json
import pytest
import physics_solver as S
import physics_simulation as P


@pytest.fixture(scope="module")
def conv():
    return {s: S.convergence(s) for s in ("upwind", "laxwendroff")}


def test_print_table(conv):
    for s, rows in conv.items():
        for r in rows:
            print(s, json.dumps(r))


def test_laxwendroff_second_order(conv):
    rows = conv["laxwendroff"]
    for r in rows[-2:]:
        assert 1.8 <= r["order_l2"] <= 2.2 and 1.8 <= r["order_max"] <= 2.2, r


def test_upwind_first_order(conv):
    r = conv["upwind"][-1]
    assert 0.8 <= r["order_l2"] <= 1.2 and 0.8 <= r["order_max"] <= 1.2, r


def test_errors_decrease_monotonically(conv):
    for rows in conv.values():
        assert all(b["l2"] < a["l2"] for a, b in zip(rows, rows[1:]))


def test_heldout_reference_close_to_exact():
    ref = P.heldout_reference()
    print("heldout set", tuple(ref["coords"].shape), "fd_vs_exact_rel_l2", ref["fd_vs_exact_rel_l2"])
    assert ref["fd_vs_exact_rel_l2"] < 2e-3


def test_solver_uses_same_problem_as_pinn():
    import numpy as np, torch
    pts = np.array([[-0.3, 0.2, 0.4], [0.5, -0.9, 0.9]])
    for x, y, t in pts:
        a = S.exact_np(x, y, t)
        b = P.exact(torch.tensor(x), torch.tensor(y), torch.tensor(t)).item()
        assert abs(a - b) < 1e-12
