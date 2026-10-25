# Mathematics engine

**Status:** built (SymPy, mpmath, Z3 proofs for polynomial real arithmetic). Lean proofs: planned.

`engines/math/` is its own engine. The physics engine calls it only through `MathEngine.compute`.

## What it does
| op | backend | example |
|---|---|---|
| `simplify`, `diff`, `integrate` (indefinite or definite), `limit`, `series` | SymPy | `diff("sin(x)") -> "cos(x)"`, `integrate("sin(x)") -> "-cos(x)"` |
| `solve`, `solve_steps` (worked steps for degree 1-2, then substitution check) | SymPy | `solve("x**2 - 5*x + 6 = 0") -> ["2", "3"]` |
| `matrix_inverse`, `linsolve` | SymPy (exact rationals) | `[[2,1],[1,1]]⁻¹ = [[1,-1],[-1,2]]` |
| `evalf` (arbitrary precision) | mpmath / SymPy N | `evalf("pi", 30)` |
| `numeric_integrate` | mpmath quad | ∫₀^π sin = 2 to 1e-12 |
| `verify_root` | exact substitution, then a numeric check at 30 digits | |
| `prove` | Z3 (if installed): looks for a counterexample to the negated claim | `x**2 + 1 > 0` proved; `x**2 - 1 > 0` gives counterexample x = 0 |

NumPy is available. SciPy is installed in `/workspace/artemis-dense-venv` (used by physics).

## Sandboxing
`compute(op, args, sandboxed=True)` runs the call in the code_exec sandbox, which the tool router
always uses for `math.compute`, because SymPy's parser evaluates strings. Under
[sandbox.env](Sandbox.md), SymPy and mpmath are staged into `/tmp/artemis-sandbox/libs` and the
engine file is shipped into the task dir. The child can see neither `/home/box` nor `/workspace`.

## Connections
- Tool head: `math.compute` in `TOOLS`. The router dispatches it sandboxed.
- Autonomous: a queued `{"name": "math.compute", ...}` problem becomes a `solve_problem` action,
  and the result is logged as an execution signal for self_train.
- Corpus: `python -m artemis_dense.data.build_problems` writes `data/problems/math.jsonl`. The
  run on 2026-10-25 produced 200 generated and 200 verified problems (quadratics checked by
  substitution, derivatives checked by integrating back, definite integrals checked numerically).
  `build_corpus` adds them as `<problem>` blocks.

## Example dialogues
<!-- example -->
**User:** Solve x^2 - 5x + 6 = 0 and show your work.
**Artemis:** Move everything to one side: x² − 5x + 6 = 0. It's a quadratic with a=1, b=−5, c=6, so the discriminant is 25 − 24 = 1 and x = (5 ± 1)/2. That gives x = 2 or x = 3. Both check out when substituted back.
