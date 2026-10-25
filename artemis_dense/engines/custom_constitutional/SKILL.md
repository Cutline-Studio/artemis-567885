---
name: artemis-custom-constitutional
description: Use the Artemis custom constitutional ai engine (promotion gate) — Latoya's constitution: loose warnings, conditional strict mode, monitoring.
---

# Custom Constitutional AI engine (promotion gate)

When to use: whenever Artemis needs to check text/action/paths against the constitution or switch to strict mode on escalation.

Steps:
1. check text/action/paths against the constitution
2. switch to strict mode on escalation
3. block promotion while strict (can_promote)
4. resume only after approve_resume(human)
5. log checks/alerts, prune, build a dashboard

Safety: operator-owned files; editing them trips strict mode.

Full doc: [CustomConstitutionalAI.md](../../CustomConstitutionalAI.md)
