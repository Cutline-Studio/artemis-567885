# Growth-health fixes: evidence, tests and A/B (2026-10-27, measured)

Branch `expand-loop-growth-health`, base `expand-loop` @ `7be7914` (PR #21 merged). Caps (`config.json`) unchanged.
Raw output: `runs/growth-health-ab-20261027/`. Script: `tools/ab_growth_health/run_ab.sh` + `summarize.py`.

## The four review points, checked against the code (base file:line)

| # | Claim | Real? | Evidence (base `7be7914`) | Change |
|---|---|---|---|---|
| 1 | Experts are added next to idle experts | **Yes** | `expand.py:121` picks `open_layers[-1]` with no load check. In the 2026-10-27 run (`runs/session-20261027/vast-main/final_config.json`): 195 expert additions, **28 into a layer that already had an expert with zero load**, 75 into a layer with an expert < 0.25x its uniform share. At cycle 624, layer 3 has 9 such experts. | `growth_min_load_ratio` (0.25), `growth_min_load_entropy` (0.8, uses `moe.expert_load_entropy`, same definition as PR #11), `growth_min_healthy_layer_frac` (0.5) for new layers. Per-layer entropy / min-load ratio now logged each cycle. |
| 2 | `me` counts only `topk_indices[:, 0]` | **Not on the path the loop uses** | `moe.py:88` is the `legacy=True` path (kept verbatim for reproducibility, docs/REVIEW.md row 1). The default path counts all slots: `moe.py:134-135` and `moe.py:188-189` (`one_hot(topk_indices).sum(1).mean(0) / top_k`). `expand.build_model` never sets `legacy`. | No code change. Regression tests: expand-built models are non-legacy, and aux equals the all-slot formula (and differs from top-1-only) for loop and vectorized dispatch. |
| 3 | Stall fires forever once converged; reset feeds it | **First half yes, second half not measured as claimed** | `expand.py:106` uses the within-cycle `rel_improvement` (`expand.py:343`) vs 0.01. Cycles 125-624: 318/500 stalled. The cycle after a growth had mean rel_improvement 0.00825 vs 0.00701 otherwise (carry-over on), so growth did not *depress* the next cycle's improvement; the runaway comes from the 0.01 test being true most of the time. | Growth must pay: after a growth wait `stall_patience` (3) cycles, compare mean eval loss (dropless when logged) of the 3 post-growth cycles with the 3 pre-growth cycles. Gain < `min_growth_gain` (1%): an expert growth escalates to a layer; a layer growth (or no room for a layer) means converged, so no growth until `growth_cooldown` (25) cycles pass. Optional `loss_floor`. The post-growth cycle's own rel_improvement no longer drives the decision. |
| 4 | New experts clone the most-loaded expert | **Yes** | `expand.py:237-239`. Unit test: a clone's gate row ties the source's logit, so it is routed >95% the same tokens and the source stays saturated. | `expert_init: "mean"` (parameter mean of the layer's experts + noise, gate row = mean row + noise): it starts at the router's average logit instead of twinning one expert, and aux pulls load to it. `"most_loaded"` keeps the old behaviour (bit-identical, tested). |

`expand.LEGACY_GROWTH_RULES` reproduces the old planner; a unit test asserts it still grows on every stalled cycle.

## Tests (same instance, before = `7be7914`, after = this branch)

| | passed | failed | xfailed |
|---|---|---|---|
| before | 143 | 5 | 1 |
| after | 166 (+23 new in `tests/test_growth_health.py`) | 5 | 1 |

The same 5 fail on both: `checks/test_check3_moe_dispatch.py::test_loop_vs_vectorized[cfg5-True-dtype0]` and the four `checks/test_check5_causal.py::test_causal_dropless_moe_has_no_leak[*]`. They assert exact float equality (`== 0.0`) and passed on the 32-core Azure/Vast hosts earlier today (148 passed, 1 xfailed). This host was a Xeon E5-2680 v4 with 5 training arms running in parallel. They are pre-existing and environment-dependent, not caused by this change. All of `tests/` passes before and after.

## A/B (one Vast.ai instance, RTX 2060 SUPER + 28-core Xeon E5-2680 v4, torch 2.5.1+cu124, py 3.11.10)

The recipe is the 2026-10-27 winner (cosine + warmup 30, lr_min 1e-5, Adam carry-over, batch 64, EMA 0.99). Caps are 16 layers / 16 experts / CF 3.0, the seed is 567885, and old and new use the same per-cycle seeds. Held-out numbers come from `evaluate.py` (1024 sequences, its own seed, CPU).

From cycle 624 neither version can grow: every layer is at the per-layer cap and there are 16 layers. So the continuation arm starts from `checkpoint_cycle365.pt` (draft release `ckpt-expand-loop-upgrades-20261027`, sha256 `c268feec...`), where growth was still active.

| arm | cycles | growth events (layers) | final layers / experts | layers with an idle expert | mean load entropy | held-out MSE capped (raw / EMA) | held-out MSE dropless (raw / EMA) | s/cycle |
|---|---|---|---|---|---|---|---|---|
| F_old (fresh, old code) | 0-149 | 38 (6) | 7 / 60 | 0 | 0.9957 | 0.022870 / 0.022845 | 0.022870 / 0.022845 | 10.9 |
| **F_new** (fresh, all fixes) | 0-149 | **13 (5)** | 6 / **32** | 0 | 0.9952 | **0.022822** / **0.022793** | **0.022686** / **0.022666** | **9.3** |
| F_new_ml (new rules, old clone init) | 0-149 | 12 (5) | 6 / 31 | 0 | 0.9978 | 0.022922 / 0.022897 | 0.022922 / 0.022897 | 9.5 |
| start (cycle 365 ckpt) | – | – | 16 / 255 | – | – | 0.026252 / 0.025240 | 0.021763 / 0.021747 | – |
| C_old (365 → 379, old) | 15 | 4 (0) | 16 / 259 | 6 | 0.8274 | 0.025798 / 0.023784 | 0.021780 / 0.021738 | 147.5 |
| C_new (365 → 379, new) | 15 | **0** | 16 / 255 | 7 | 0.8366 | 0.028569 / 0.023867 | 0.021763 / 0.021730 | 145.4 |

What this shows:
- **Runaway growth is cut and quality holds.** From scratch, the fixes made 66% fewer growths (13 vs 38) and ended with about half the experts (32 vs 60). Held-out MSE was equal or slightly better (dropless 0.02269 vs 0.02287), and each cycle was 15% faster.
- **A converged model stops growing.** From cycle 365, the old loop added 4 more experts to layer 0 in 15 cycles. The new loop added none: its last growth gained -0.54% (it needs +1%), and no layer could be added. Dropless and EMA held-out MSE are the same within noise (0.021763 vs 0.021780 raw dropless). The raw capped number for C_new (0.02857) is worse, but capped eval of the raw weights swings a lot from cycle to cycle (the 32-sequence eval moved 0.0273-0.0306 over cycles 619-624), and EMA capped is equal (0.02387 vs 0.02378). Treat it as noise, not a measured regression; one seed only.
- **Fix 4 is not shown to help yet.** With the new rules, mean-init (F_new) vs clone-init (F_new_ml) is within noise on held-out MSE (0.02282 vs 0.02292 capped, dropless 0.02269 vs 0.02292). Clone-init even had the lower in-loop eval loss over the last 5 cycles. The duplication problem is real at the mechanism level (unit test), but this short single-seed A/B can't separate the two inits. Keep `"mean"` as the default on the mechanism argument, or flip it to `"most_loaded"` if a multi-seed run disagrees.
- **Fix 1 doesn't revive the already-collapsed layers.** The gate stops adding experts there, but layers 3-6, 11 and 14 still have idle experts at cycle 379 in both arms. Reviving or pruning them is separate work.

## Cost and teardown
- Vast instance 57991927 (offer 25468214, $0.0696/h), rented 13:28:52 and destroyed 14:10:15 EDT. It had an on-instance self-destroy at rent+60 min plus a box-side spend/time killer.
- Vast credit $14.743560 → $14.695343 = **$0.0482** (`credit_before.txt`, `credit_after.txt`, `spend.log`). After teardown the Vast account showed 0 instances.
