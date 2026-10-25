---
name: artemis-heart
description: Use the Artemis heart engine — applies the ranked values from Heart.md.
---

# Heart engine

When to use: whenever Artemis needs to score a reply or action on each value or choose among candidates by rank (lexicographic).

Steps:
1. score a reply or action on each value
2. choose among candidates by rank (lexicographic)
3. send value scores to self_train's signal log

Safety: heuristic scorer; cannot override the constitution or approvals.

Full doc: [Heart.md](../../Heart.md)
