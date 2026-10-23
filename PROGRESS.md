# Progress log (dates in America/New_York)

## Iteration 1 — 2026-10-23
- Baseline: the four files committed exactly as provided. `python sandbox.py` → Total Loss 1.5083, dropped 6, ∇x 0.1581, ∇W_g 0.0546 (torch 2.14.1+cpu).
- Stage 1: verified review findings (`docs/REVIEW.md`, `tools/investigate_baseline.py`); fixed aux loss, capacity priority, dropped-token fallback and drop stats behind `legacy` flag (default fixed); 18 pytest tests; sandbox runs both modes; legacy reproduces baseline exactly.
- Stage 2 (same run, user specs folded in): `model.py` DynamicMultiBlockTransformer; `expand.py` expand engine (sin(X) regression, config.json as single source of truth, caps, `runs/expand_log.jsonl`); `supervisor.py` ArtemisSelfFixingSupervisor with backups, ast check, subprocess timeout (default 300 s), failure parsing, shape auto-correct against the checkpoint + one validation re-run, rollback + alternate mutation path, capped attempts; `physics_simulation.py` separate SIREN-MoE PINN sandbox; sandbox.py fixed mode now regresses sin(X) (legacy keeps the original zeros target and numbers). 47 tests pass.
- Real 10-cycle supervised run (2026-10-23 ~00:15 ET): eval loss 0.0965 → 0.0521; CF 0.8 → 1.2; experts [4] → [6, 4]; layers 1 → 2; alpha stayed 0.1 (aux never > 1.2); no failures/rollbacks in the real run (rollback + auto-correct exercised by tests). Full table: `config.json` history / `runs/expand_log.jsonl`.
- Physics run (120 s, 6358 Adam steps): PDE residual MSE 3.40e-4, IC MSE 1.53e-4, BC MSE 1.18e-5, rel. L2 vs analytic 0.055 (t=0), 0.157 (t=0.5), 0.250 (t=1), overall 0.173. Router collapsed onto experts 2 and 3 (experts 0, 1 never top-1); training losses rose again after ~step 250 at lr 1e-3. Results in `runs/physics/`.
- Concept specs copied verbatim to `docs/` (CONTINUOUS_LANGUAGE_FIELD.md is truncated before its code); user's streaming wrapper saved to `docs/reference/clf_stream_original.py`; later stages 9(a)–(e) added to PLAN.md.

### Next
- Stage 3: causal mask + positional encoding options (PLAN.md). Physics follow-ups: lower lr / cosine schedule, stronger load balancing to stop expert collapse, longer error-vs-time study.
