"""Finite-difference reference solver for the PINN's PDE (repo-checks, check 4).

    dh/dt + c (dh/dx + dh/dy) = 0,  (x, y) in [-1, 1]^2,  t in [0, T]
    h(x, y, 0) = h0(x, y);  inflow Dirichlet data on x = -1 and y = -1.

This is the SAME problem as physics_simulation.py (C, h0 and exact are imported
from there). Two explicit schemes on a uniform (N x N) node grid, float64:

  * "upwind":       first-order donor-cell, expected order 1
  * "laxwendroff":  unsplit 2-D Lax-Wendroff (incl. the cross term), expected order 2

Boundaries: inflow nodes (x=-1 or y=-1) take the exact inflow data at the new
time level; outflow nodes (x=+1 or y=+1) use second-order linear extrapolation
(u_N = 2 u_{N-1} - u_{N-2}), i.e. no exact data is used there.
The validated solver provides a reference solution where no analytic one exists;
here the analytic one also exists, which is what lets us validate it.
"""
import math
import numpy as np

from physics_simulation import C, X0, Y0, SIGMA


def h0_np(x, y):
    return np.exp(-((x - X0) ** 2 + (y - Y0) ** 2) / (2 * SIGMA ** 2))


def exact_np(x, y, t):
    return h0_np(x - C * t, y - C * t)


def solve(n: int, t_end: float = 1.0, scheme: str = "laxwendroff", cfl: float = 0.4, snapshots=()):
    """Return (x, y, {t: field}) with fields on the n x n node grid (indexing='ij').

    dt is chosen so that t_end and every snapshot time are hit exactly
    (snapshot times must be multiples of the step).
    """
    x = np.linspace(-1.0, 1.0, n)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="ij")
    steps = int(math.ceil(t_end * C / (cfl * dx)))
    # make every snapshot time land on a step
    while any(abs(round(s * steps / t_end) - s * steps / t_end) > 1e-9 for s in snapshots):
        steps += 1
    dt = t_end / steps
    nu = C * dt / dx
    u = h0_np(X, Y)
    out = {}
    if 0.0 in snapshots:
        out[0.0] = u.copy()
    for k in range(1, steps + 1):
        t_new = k * dt
        new = u.copy()
        if scheme == "upwind":
            new[1:, 1:] = u[1:, 1:] - nu * (u[1:, 1:] - u[:-1, 1:]) - nu * (u[1:, 1:] - u[1:, :-1])
        elif scheme == "laxwendroff":
            c_ = u[1:-1, 1:-1]
            ux = u[2:, 1:-1] - u[:-2, 1:-1]
            uy = u[1:-1, 2:] - u[1:-1, :-2]
            uxx = u[2:, 1:-1] - 2 * c_ + u[:-2, 1:-1]
            uyy = u[1:-1, 2:] - 2 * c_ + u[1:-1, :-2]
            uxy = u[2:, 2:] - u[2:, :-2] - u[:-2, 2:] + u[:-2, :-2]
            new[1:-1, 1:-1] = (c_ - 0.5 * nu * (ux + uy) + 0.5 * nu ** 2 * (uxx + uyy)
                               + 0.25 * nu * nu * uxy)
            # outflow: second-order extrapolation (no exact data)
            new[-1, 1:-1] = 2 * new[-2, 1:-1] - new[-3, 1:-1]
            new[1:, -1] = 2 * new[1:, -2] - new[1:, -3]
        else:
            raise ValueError(scheme)
        # inflow Dirichlet data at the new time level
        new[0, :] = exact_np(X[0, :], Y[0, :], t_new)
        new[:, 0] = exact_np(X[:, 0], Y[:, 0], t_new)
        u = new
        for s in snapshots:
            if abs(t_new - s) < 1e-12:
                out[s] = u.copy()
    out[t_end] = u
    return x, X, Y, out, dict(steps=steps, dt=dt, dx=dx, cfl=nu)


def errors(n, scheme, t_end=1.0, cfl=0.4):
    x, X, Y, out, info = solve(n, t_end, scheme, cfl)
    e = out[t_end] - exact_np(X, Y, t_end)
    return dict(n=n, dx=info["dx"], steps=info["steps"], cfl=info["cfl"],
                max_abs=float(np.abs(e).max()), l2=float(np.sqrt(np.mean(e ** 2))))


def convergence(scheme, ns=(21, 41, 81, 161, 321), t_end=1.0, cfl=0.4):
    rows = [errors(n, scheme, t_end, cfl) for n in ns]
    for a, b in zip(rows, rows[1:]):
        b["order_l2"] = math.log(a["l2"] / b["l2"]) / math.log(a["dx"] / b["dx"])
        b["order_max"] = math.log(a["max_abs"] / b["max_abs"]) / math.log(a["dx"] / b["dx"])
    return rows
