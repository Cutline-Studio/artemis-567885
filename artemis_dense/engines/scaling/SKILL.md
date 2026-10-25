---
name: artemis-scaling
description: Use the Artemis neural scaling laws engine — Kaplan/Chinchilla formulas, compute-optimal planning, Artemis's own fitted curve.
---

# Neural scaling laws engine

When to use: whenever Artemis needs to predict loss from n and d (published fits) or compute-optimal n, d for a flop budget.

Steps:
1. predict loss from N and D (published fits)
2. compute-optimal N, D for a FLOP budget
3. budget -> plan (with user-supplied GPU price/MFU)
4. fit a power law to measured points
5. tiny CPU sweep

Safety: plans only; never rents or spends.

Full doc: [ScalingLaws.md](../../ScalingLaws.md)
