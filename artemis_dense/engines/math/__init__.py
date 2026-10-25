"""Mathematics engine: exact/symbolic (SymPy), numeric (NumPy, SciPy if installed), arbitrary precision
(mpmath), step-by-step worked solutions and answer verification. Proof checks: Z3 for polynomial
real-arithmetic claims when z3-solver is installed; Lean = planned.

compute(op, args, sandboxed=False):
  ops: simplify, diff, integrate, solve, matrix_inverse, linsolve, limit, series, evalf, verify_root,
       numeric_integrate, prove
  sandboxed=True runs the same call inside the code_exec sandbox (used for model tool calls, because
  SymPy's parser evaluates strings).
"""
import json, os, sys
from typing import Optional

import sympy as sp
import mpmath


class MathEngine:
    def __init__(self, code_exec=None):
        self.code_exec = code_exec

    # ---------------------------------------------------------------- core ops
    @staticmethod
    def _sym(s):
        return sp.sympify(s, convert_xor=True)

    def simplify(self, expr):
        return str(sp.simplify(self._sym(expr)))

    def diff(self, expr, var="x", n=1):
        return str(sp.diff(self._sym(expr), sp.Symbol(var), int(n)))

    def integrate(self, expr, var="x", lower=None, upper=None):
        x = sp.Symbol(var)
        r = sp.integrate(self._sym(expr), (x, self._sym(lower), self._sym(upper))) if lower is not None \
            else sp.integrate(self._sym(expr), x)
        return str(r)

    def solve(self, equation, var="x"):
        x = sp.Symbol(var)
        lhs, _, rhs = equation.partition("=")
        eq = sp.Eq(self._sym(lhs), self._sym(rhs or "0"))
        return sorted([str(s) for s in sp.solve(eq, x)])

    def solve_steps(self, equation, var="x") -> dict:
        """Worked solution for polynomial equations up to degree 2 (other cases: plain solve)."""
        x = sp.Symbol(var)
        lhs, _, rhs = equation.partition("=")
        poly = sp.expand(self._sym(lhs) - self._sym(rhs or "0"))
        steps = [f"Move everything to one side: {poly} = 0"]
        p = sp.Poly(poly, x)
        if p.degree() == 2:
            a, b, c = p.all_coeffs()
            d = sp.simplify(b ** 2 - 4 * a * c)
            steps += [f"Quadratic with a={a}, b={b}, c={c}", f"Discriminant b^2-4ac = {d}",
                      "x = (-b ± sqrt(disc)) / (2a)"]
        elif p.degree() == 1:
            a, b = p.all_coeffs()
            steps.append(f"Linear: x = {-b}/{a}")
        sols = self.solve(equation, var)
        steps.append(f"Solutions: {sols}")
        checks = [self.verify_root(equation, s, var) for s in sols]
        steps.append(f"Check by substitution: {checks}")
        return {"solutions": sols, "steps": steps, "verified": all(checks)}

    def verify_root(self, equation, value, var="x") -> bool:
        lhs, _, rhs = equation.partition("=")
        e = self._sym(lhs) - self._sym(rhs or "0")
        exact = sp.simplify(e.subs(sp.Symbol(var), self._sym(value)))
        if exact == 0:
            return True
        return abs(complex(sp.N(exact, 30))) < 1e-12           # numeric check

    def matrix_inverse(self, rows):
        return [[str(v) for v in r] for r in sp.Matrix(rows).inv().tolist()]

    def linsolve(self, rows, rhs):
        sol = sp.Matrix(rows).LUsolve(sp.Matrix(rhs))
        return [str(v) for v in sol]

    def limit(self, expr, var="x", point="0"):
        return str(sp.limit(self._sym(expr), sp.Symbol(var), self._sym(point)))

    def series(self, expr, var="x", point="0", n=6):
        return str(sp.series(self._sym(expr), sp.Symbol(var), self._sym(point), int(n)).removeO())

    def evalf(self, expr, digits=50):
        mpmath.mp.dps = int(digits)
        return str(sp.N(self._sym(expr), int(digits)))

    def numeric_integrate(self, expr, var="x", lower=0, upper=1):
        f = sp.lambdify(sp.Symbol(var), self._sym(expr), "mpmath")
        return float(mpmath.quad(f, [float(lower), float(upper)]))

    def prove(self, claim: str, var: str = "x") -> dict:
        """Prove a universally quantified polynomial claim over the reals, e.g. 'x**2 + 1 > 0', with Z3."""
        try:
            import z3
        except ImportError:
            return {"status": "planned", "detail": "z3-solver not installed"}
        sym = self._sym(claim)
        X = z3.Real(var)

        def conv(e):
            if e.is_Symbol:
                return X
            if e.is_Number:
                return z3.RealVal(str(e))
            if e.is_Add:
                r = conv(e.args[0])
                for a in e.args[1:]:
                    r = r + conv(a)
                return r
            if e.is_Mul:
                r = conv(e.args[0])
                for a in e.args[1:]:
                    r = r * conv(a)
                return r
            if e.is_Pow and e.exp.is_Integer and e.exp > 0:
                b = conv(e.base); r = b
                for _ in range(int(e.exp) - 1):
                    r = r * b
                return r
            raise ValueError(f"unsupported term {e}")
        rel = {sp.StrictGreaterThan: lambda a, b: a > b, sp.GreaterThan: lambda a, b: a >= b,
               sp.StrictLessThan: lambda a, b: a < b, sp.LessThan: lambda a, b: a <= b,
               sp.Equality: lambda a, b: a == b}[type(sym)]
        s = z3.Solver()
        s.add(z3.Not(rel(conv(sym.lhs), conv(sym.rhs))))
        r = s.check()
        return {"status": "proved" if r == z3.unsat else ("counterexample" if r == z3.sat else "unknown"),
                "model": str(s.model()) if r == z3.sat else None}

    OPS = ("simplify", "diff", "integrate", "solve", "solve_steps", "matrix_inverse", "linsolve", "limit",
           "series", "evalf", "verify_root", "numeric_integrate", "prove")

    # ---------------------------------------------------------------- dispatch
    def compute(self, op: str, args: dict, sandboxed: bool = False):
        if op not in self.OPS:
            raise ValueError(f"unknown math op {op!r}")
        if not sandboxed:
            return getattr(self, op)(**args)
        if self.code_exec is None:
            raise RuntimeError("sandboxed math needs a code_exec engine")
        sb = getattr(self.code_exec, "sandbox", None)
        if sb is not None:                       # sandbox.env denies /home/box: use a staged copy under /tmp/artemis-sandbox
            libs = [sb.stage_libs("sympy", "mpmath")]
        else:
            libs = sorted({os.path.dirname(os.path.dirname(m.__file__)) for m in (sp, mpmath)})
        src = open(os.path.abspath(__file__), encoding="utf-8").read()     # engine code shipped as a file,
        code = (f"import sys, json, os\nsys.path[:0] = [os.getcwd()] + {json.dumps(libs)}\n"  # so the repo needn't be visible
                "from math_engine import MathEngine\n"
                f"print(json.dumps(MathEngine().compute({op!r}, json.loads({json.dumps(json.dumps(args))}))))\n")
        L = self.code_exec.limits
        old = (L.ro_paths, L.timeout_s, L.cpu_s, L.mem_bytes)
        L.ro_paths = tuple(sorted(set(L.ro_paths) | set(libs)))
        L.timeout_s, L.cpu_s, L.mem_bytes = max(L.timeout_s, 30), max(L.cpu_s, 30), max(L.mem_bytes, 2 * 1024 ** 3)
        try:
            r = self.code_exec.run(code, files={"math_engine.py": src})
        finally:
            L.ro_paths, L.timeout_s, L.cpu_s, L.mem_bytes = old
        if not r.ok:
            raise RuntimeError(f"sandboxed math failed: {r.stderr[-400:]}")
        return json.loads(r.stdout)


def problem_set(n: int = 50, seed: int = 0) -> list:
    """Generated problems with answers verified by the engine (for the combined corpus)."""
    import random
    rng, m, out = random.Random(seed), MathEngine(), []
    for i in range(n):
        a, b = rng.randint(1, 9), rng.randint(-9, 9)
        r1, r2 = rng.randint(-9, 9), rng.randint(-9, 9)
        kind = i % 3
        if kind == 0:
            eq = f"x**2 - ({r1 + r2})*x + ({r1 * r2}) = 0"
            ans = m.solve(eq)
            ok = all(m.verify_root(eq, s) for s in ans)
            out.append({"q": f"Solve {eq}", "a": ans, "verified": bool(ok)})
        elif kind == 1:
            e = f"{a}*x**{rng.randint(2, 5)} + {b}*sin(x)"
            d = m.diff(e)
            ok = sp.simplify(sp.integrate(sp.sympify(d), sp.Symbol("x")) - sp.sympify(e)).is_constant()
            out.append({"q": f"Differentiate {e}", "a": d, "verified": bool(ok)})
        else:
            e = f"{a}*x + {b}"
            v = m.integrate(e, "x", 0, 2)
            ok = abs(float(sp.sympify(v)) - m.numeric_integrate(e, "x", 0, 2)) < 1e-9
            out.append({"q": f"Integrate {e} from 0 to 2", "a": v, "verified": bool(ok)})
    return out
