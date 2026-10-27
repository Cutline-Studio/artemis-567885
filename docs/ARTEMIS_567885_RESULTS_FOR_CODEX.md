# Artemis 567885 — Iteration 1 Results (handoff for Codex)

Date: 2026-10-23 (America/New_York). All numbers below are real measured output from the box.

## Repo
- GitHub (PRIVATE): https://github.com/lpittman-blip/artemis-567885 — owner lpittman-blip (do not use lpittman-g)
- Local: /workspace/artemis-567885
- `main` = baseline commit `ffc29a7` (the user's original README.md, moe.py, transformer.py, sandbox.py, unchanged)
- PR #1 `stage-1-tests-fixes` -> main: https://github.com/lpittman-blip/artemis-567885/pull/1 (bug fixes + tests)
- PR #2 `stage-2-expand-engine`, stacked on #1: https://github.com/lpittman-blip/artemis-567885/pull/2 (expand engine, supervisor, multi-block, physics sandbox)
- Neither PR is merged. Merge order: #1 then #2.
- Env: PyTorch 2.14.1 (CPU), pytest 9.1.1

## Tests
47 passed in 36.95 s — test_moe 18, test_model 6, test_expand 10, test_supervisor 7, test_physics 6.

## Sandbox output
Original (legacy) — reproduced exactly with `legacy=True`, enforced by a test:
```
Total Loss Value: 1.5083
Enforced Drop-out Vector Count: 6
Input Structural Gradient Norm (\nabla_x L): 0.1581
Router Gate Matrix Gradient Norm (\nabla_W_g L): 0.0546
```
Fixed mode, regressing X -> sin(X): total loss 0.3415, dropped slots 6, grad norm x 0.0663, gate 0.0228, output shape == input shape (2, 8, 16), aux 1.0281, capacity 7/expert, 0 of 16 tokens fully dropped.

## Bugs found in the original (docs/REVIEW.md, each verified by tools/investigate_baseline.py)
1. Capacity overflow dropped tokens by batch order, not router priority (kept a 0.1-weight token, dropped 5.0 and 6.0). Fixed: priority ranking.
2. Dropped tokens detected via `sum == 0` and their input copied into the output, then added again by the residual (double count); also clobbered processed tokens whose output summed to 0. Fixed: explicit mask, zero output.
3. Aux loss used noisy logits and top-1 only while routing is top-2 (same input, two noise seeds: 1.0076 vs 1.0055). Fixed: clean logits, all top-k slots.
4. With top_k=1 the gate gets exactly 0 gradient from the task loss (top-2: 0.0140).
5. dropped_count counts token-slot pairs, can exceed token count (12 for 8 tokens). Dropped tokens now reported separately.
6. Attention has no causal mask and no positional info (permutation-equivariant). Not fixed yet — next stage.
7. Per-expert Python loop is slow on CPU: 64 experts 21.7–38.7 ms/forward vs ~10–13 ms at 4.

## Expand engine — 10 cycles via supervisor.py
Rules: dropped>0 -> capacity +0.05; aux>1.2 -> alpha up; loss stall -> add expert (weights kept); still stalled after expert add -> add layer (identity init). Caps: capacity<=2.0, experts<=32/layer, layers<=8, alpha<=1.0. State in config.json; logs in runs/expand_log.jsonl and runs/supervisor_log.jsonl.

| cycle | loss | aux | dropped | capacity | alpha | experts | layers | change |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.096482 | 1.0013 | 204 | 0.80 | 0.1 | [4] | 1 | none |
| 1 | 0.090077 | 1.0011 | 152 | 0.85 | 0.1 | [4] | 1 | capacity +0.05 |
| 2 | 0.085289 | 1.0012 | 100 | 0.90 | 0.1 | [4] | 1 | capacity +0.05 |
| 3 | 0.081592 | 1.0028 | 52 | 0.95 | 0.1 | [4] | 1 | capacity +0.05 |
| 4 | 0.080722 | 0.9993 | 39 | 1.00 | 0.1 | [5] | 1 | capacity + add expert |
| 5 | 0.079299 | 0.9990 | 18 | 1.05 | 0.1 | [5] | 1 | capacity +0.05 |
| 6 | 0.077656 | 1.0006 | 2 | 1.10 | 0.1 | [6] | 1 | capacity + add expert |
| 7 | 0.073885 | 1.0019 | 0 | 1.15 | 0.1 | [6, 4] | 2 | capacity + add layer |
| 8 | 0.061494 | 1.0021 | 8 | 1.15 | 0.1 | [6, 4] | 2 | none |
| 9 | 0.052061 | 1.0024 | 0 | 1.20 | 0.1 | [6, 4] | 2 | capacity +0.05 |

Final grad norms: x 0.003734, gate 0.008992. No rollbacks in the live run. Rollback/auto-correct verified by injected-fault tests (syntax error, output-shape break, NaN weights, d_model drift 512 vs 16, n_heads not dividing d_model, sabotaged layer add) — all passed. Example autocorrect log: realigned d_model 512 -> 16 from checkpoint, weights kept.

## Physics sandbox (physics_simulation.py)
SIREN experts, coordinate router, PDE dh/dt + c(dh/dx + dh/dy) = 0 with c = 0.5, plus Gaussian initial condition and inflow boundary terms (PDE-only loss is trivially solved by a constant). 120 s CPU run, 6,358 steps:
- PDE residual MSE 3.40e-4, IC MSE 1.53e-4, boundary MSE 1.18e-5
- Relative error vs analytic traveling wave: 0.055 (t=0), 0.157 (t=0.5), 0.250 (t=1), 0.173 overall
- Router collapse: top-1 counts [0, 0, 1129, 8132]; losses rose after ~step 250 at lr 1e-3 (PLAN stage 8b)

## Next stages (PLAN.md)
3. Causal mask + positional encoding options
4. Vectorized expert dispatch + CPU benchmark
5+. Alpha/capacity sweep; physics router-collapse fix (8b)
9a fourier.py · 9b webstream.py (real HTTP Range, 206 check, labeled fallback) · 9c byte-level next-byte model with causal mask, bits-per-byte on held-out text · 9d optional hypernetwork · 9e HTTP serving deferred (no Vercel; no hosting without the user's OK)

A routine continues the loop every 3 hours, 9 AM–9 PM ET, opening one PR per stage (never merging).
