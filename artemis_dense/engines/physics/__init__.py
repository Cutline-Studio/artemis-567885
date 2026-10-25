"""Physics engine (separate from math): units + dimensional analysis (pint), CODATA constants
(scipy.constants), kinematics/mechanics solvers, ODE simulation (scipy.integrate) and simple sims
(projectile, oscillator, orbit). Every compute() answer carries units and passes a dimensional check.
Symbolic work goes through the math engine's public interface (MathEngine), never its internals.
Needs pint + scipy (installed in /workspace/artemis-dense-venv on this box); raises ImportError otherwise.
"""
import math

import numpy as np

try:
    import pint
    from scipy import constants as C
    from scipy.integrate import solve_ivp
except ImportError as e:            # pragma: no cover
    raise ImportError("physics engine needs pint and scipy (see Physics.md)") from e

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


class DimensionError(ValueError):
    pass


def check_dims(q, expected: str):
    """Raise DimensionError unless quantity q has the dimensionality of `expected` units."""
    if not isinstance(q, ureg.Quantity) or q.dimensionality != Q_(1, expected).dimensionality:
        raise DimensionError(f"expected [{Q_(1, expected).dimensionality}], got "
                             f"[{getattr(q, 'dimensionality', 'dimensionless number')}]")
    return q


class PhysicsEngine:
    def __init__(self, math_engine=None):
        if math_engine is None:
            from ..math import MathEngine
            math_engine = MathEngine()
        self.math = math_engine

    # constants (CODATA via scipy.constants)
    def constant(self, name: str):
        table = {"g": (C.g, "m/s**2"), "c": (C.c, "m/s"), "G": (C.G, "m**3/(kg*s**2)"),
                 "h": (C.h, "J*s"), "k_B": (C.k, "J/K"), "e": (C.e, "C"), "N_A": (C.N_A, "1/mol"),
                 "m_e": (C.m_e, "kg"), "epsilon_0": (C.epsilon_0, "F/m")}
        v, u = table[name]
        return Q_(v, u)

    # kinematics
    def projectile_range(self, v0, angle_deg, g=None):
        g = g or self.constant("g")
        v0 = check_dims(v0, "m/s")
        return check_dims((v0 ** 2 * math.sin(2 * math.radians(angle_deg)) / g).to("m"), "m")

    def simulate_projectile(self, v0, angle_deg, g=None, drag_per_m=0.0):
        g = (g or self.constant("g")).to("m/s**2").magnitude
        v = check_dims(v0, "m/s").to("m/s").magnitude
        th = math.radians(angle_deg)

        def f(t, s):
            x, y, vx, vy = s
            sp_ = math.hypot(vx, vy)
            return [vx, vy, -drag_per_m * sp_ * vx, -g - drag_per_m * sp_ * vy]
        hit = lambda t, s: s[1] if t > 0 else 1.0
        hit.terminal, hit.direction = True, -1
        sol = solve_ivp(f, (0, 1e4), [0, 0, v * math.cos(th), v * math.sin(th)], events=hit,
                        rtol=1e-10, atol=1e-10, max_step=0.5)
        return Q_(sol.y_events[0][0][0], "m")

    def oscillator(self, k, m, x0, t):
        """Undamped spring: x(t) by ODE, compared with x0 cos(wt)."""
        w = np.sqrt((check_dims(k, "N/m") / check_dims(m, "kg")).to("1/s**2").magnitude)
        x0m = check_dims(x0, "m").to("m").magnitude
        tt = check_dims(t, "s").to("s").magnitude
        sol = solve_ivp(lambda _, s: [s[1], -w * w * s[0]], (0, tt), [x0m, 0], rtol=1e-10, atol=1e-12)
        return Q_(sol.y[0, -1], "m"), Q_(x0m * math.cos(w * tt), "m")

    def orbit_period(self, a, M):
        T = 2 * math.pi * (check_dims(a, "m") ** 3 / (self.constant("G") * check_dims(M, "kg"))) ** 0.5
        return check_dims(T.to("s"), "s")

    def kinetic_energy(self, m, v):
        return check_dims((0.5 * check_dims(m, "kg") * check_dims(v, "m/s") ** 2).to("J"), "J")

    def solve_symbolic(self, equation, var):
        return self.math.compute("solve", {"equation": equation, "var": var})

    def compute(self, op: str, args: dict):
        q = lambda d: Q_(d["value"], d["unit"])
        if op == "constant":
            r = self.constant(args["name"])
        elif op == "projectile_range":
            r = self.projectile_range(q(args["v0"]), args["angle_deg"])
        elif op == "kinetic_energy":
            r = self.kinetic_energy(q(args["m"]), q(args["v"]))
        elif op == "orbit_period":
            r = self.orbit_period(q(args["a"]), q(args["M"]))
        else:
            raise ValueError(f"unknown physics op {op!r}")
        return {"value": float(r.magnitude), "unit": f"{r.units:~}"}


def problem_set(n: int = 30, seed: int = 0) -> list:
    import random
    rng, p, out = random.Random(seed), PhysicsEngine(), []
    for _ in range(n):
        v, ang = rng.randint(5, 50), rng.choice([15, 30, 45, 60, 75])
        r = p.projectile_range(Q_(v, "m/s"), ang)
        sim = p.simulate_projectile(Q_(v, "m/s"), ang)
        out.append({"q": f"Range of a projectile launched at {v} m/s and {ang} degrees on Earth (no drag)?",
                    "a": f"{r.magnitude:.3f} m", "verified": bool(abs((r - sim).to("m").magnitude) < 1e-3)})
    return out
