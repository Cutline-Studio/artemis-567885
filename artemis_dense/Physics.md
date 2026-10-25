# Physics engine

**Status:** built (pint units, CODATA constants via scipy.constants, SciPy ODE simulation).

`engines/physics/` is a separate module from math. For symbolic work it calls
`MathEngine.compute` (`solve_symbolic`), never the math engine's internals.

## What it does
- **Units and dimensional analysis** (pint): every input to the solvers is a `Quantity`.
  `check_dims` raises `DimensionError` on a mismatch, for example a projectile speed given in
  metres, or kinetic energy given a time instead of a speed. `compute()` returns
  `{"value", "unit"}`.
- **Constants** (CODATA via `scipy.constants`): g = 9.80665 m/s² (standard gravity, exact by
  definition), c, G, h, k_B, e, N_A, m_e, ε₀.
- **Kinematics and mechanics**: projectile range `v₀² sin 2θ / g`, kinetic energy, Kepler orbit
  period `2π √(a³/GM)`.
- **Simulation** (`scipy.integrate.solve_ivp`, rtol 1e-10): projectile with optional quadratic
  drag (ground-hit event), undamped oscillator, checked against `x₀ cos ωt`.

## Verified in tests
- 20 m/s at 45° gives a range of 40.79 m. The formula and the ODE simulation agree to under 1 mm.
- The oscillator ODE matches the analytic solution to under 1e-6 m.
- Earth's orbit (a = 1.496e11 m, M = 1.989e30 kg) gives a period of 365.2 days.
- `projectile_range(20 m, 45°)` raises `DimensionError`.

## Connections
- Tool head: `physics.compute` (ops `constant`, `projectile_range`, `kinetic_energy`,
  `orbit_period`).
- Autonomous: `solve_problem`, the same path as math.
- Corpus: `data/problems/physics.jsonl`. The run on 2026-10-25 produced 50 generated and 50
  verified projectile problems (formula vs. simulation).

## Planned
Rigid-body dynamics, thermodynamics tables, an n-body orbit integrator with energy-drift checks.

## Example dialogues
<!-- example -->
**User:** How far does a ball go at 20 m/s and 45 degrees, ignoring drag?
**Artemis:** About 40.8 m. The range is v²·sin(2θ)/g = 400 / 9.80665 = 40.79 m, and a numerical simulation agrees to within a millimetre.
