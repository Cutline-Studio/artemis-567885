---
name: artemis-math
description: Use the Artemis mathematics engine — exact, symbolic, numeric and arbitrary-precision math with verification.
---

# Mathematics engine

When to use: whenever Artemis needs to simplify / diff / integrate / limit / series or solve with worked steps + substitution check.

Steps:
1. simplify / diff / integrate / limit / series
2. solve with worked steps + substitution check
3. matrix inverse / linear solve
4. high-precision evalf
5. numeric integration
6. Z3 proofs of polynomial inequalities

Safety: SymPy parses strings, so tool calls run inside the code_exec sandbox.

Full doc: [Mathematics.md](../../Mathematics.md)
