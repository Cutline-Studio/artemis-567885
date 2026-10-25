---
name: artemis-physics
description: Use the Artemis physics engine — unit-checked physics: constants, kinematics, ODE simulation.
---

# Physics engine

When to use: whenever Artemis needs to codata constants (g = 9.80665 m/s^2 ...) or projectile range (formula + simulation).

Steps:
1. CODATA constants (g = 9.80665 m/s^2 ...)
2. projectile range (formula + simulation)
3. oscillator and orbit period
4. kinetic energy
5. dimensional checks on every answer
6. symbolic solving via the math engine

Safety: pure computation, no I/O; raises DimensionError on unit mismatch.

Full doc: [Physics.md](../../Physics.md)
