# Repo checks (the five checks required before integration / GPU training)

All numbers below are **MEASURED** on CPU on the box (no GPU, no cloud), copied from the raw,
unedited consoles in `checks/results/`. Nothing here is an estimate unless it says *estimate*.

Environment (`results/env_console.txt`): Python 3.13.5, torch 2.14.1+cpu, numpy 2.5.4, pytest 9.1.1,
Linux 6.12 x86_64, 8 logical CPUs.
Re-run everything: `bash checks/run_all.sh /path/to/artemis-corpus-bench` (that dir holds the three
real `runs/seed*/best.pt` and `corpus/`; neither is in git).
Full suite after the changes: `python3 -m pytest -q tests checks` -> **134 passed, 1 xfailed** (strict xfail, see check 3).
**Update (check 3 fix):** the xfail's root cause is fixed; suite is now **139 passed, 0 xfailed** (see "Check 3 update" below).

Base: this branch is stacked on `stage-physics-frequency-study` (PR #3 + merged PR #4), because the code under
review (MoE fixes, multi-block model, physics PINN, pilot study, byte LM) is on the PR #1-#4 stack, not on `main`.

| # | Check | Result | Key numbers (measured) | Console |
|---|---|---|---|---|
| 1 | Historical target / normalisation / init | **PASS** + 1 finding | `checks/historical.py` restores sin(k(x-0.5t)), x/pi-1, y/pi-1, t-1, SIREN U(-b,b) on weight+bias. Bitwise (`torch.equal`) equal to `frequency_capacity_study` (k=1,2,4,8; seeds 0-4) and to the pilot/benchmark models (seeds 0-4). **Finding:** the study's output layer weight is re-initialised, so its "original k=1 setup" ablation is NOT bit-identical to the pilot model (hidden layers and output bias identical). `physics_simulation.py` deviates (Gaussian advection target, raw [-1,1] inputs, default-init biases) and is now labelled EXPLORATORY. | `check1_historical_console.txt` |
| 2 | Save / reload checkpoints | **PASS** | 10 model configs (byte LM, ledger LM, historical SIREN, physics MoE, MoE transformer legacy/fixed x loop/vectorized x causal/non-causal): trained 3 steps, state_dict to disk, fresh instance with another seed, eval-mode predictions `torch.equal`, max abs diff 0. **Real checkpoints** (seeds 42/1/2 `best.pt`): sha256 matches metrics.json; reloaded val/test bits-per-byte re-computed on the full splits equal the recorded values exactly (abs diff 0.0; e.g. seed 42 val 3.619183483419506, test 3.5859531551986064); re-save logits equal. | `check2_checkpoints_console.txt`, `check2_real_checkpoints_console.txt` |
| 3 | MoE loop vs vectorized dispatch | **PASS** with one stated-tolerance **FAIL** (float32 weight grads at N=4096) | New `dispatch="vectorized"` (sort + padded `[E,cap,d]` bmm). Same weights (state_dict copy), same noise RNG. 8 configs x {eval,train} x {fp32,fp64}: top-k assignments and kept/dropped slots identical, dropped counts identical, outputs **bit-identical (0.0)**, input grads <= 1.5e-8 (fp32) / 2.8e-17 (fp64), param grads <= 5.7e-6 (fp32) / 1.4e-14 (fp64); tolerance fp32 rtol 1e-5 atol 1e-6, fp64 1e-12. At bench size N=4096,d=128,d_ff=512: fp64 all grads <= 8.5e-14 (PASS); fp32 expert-weight grads differ up to 3.05e-5 with max grad ~69 (<= 7.8e-7 of max), which **exceeds the stated fp32 tolerance** -> recorded as strict xfail, tolerance not loosened (reduction-order rounding; fp64 agreement supports this interpretation). | `check3_moe_dispatch_console.txt` |
| 3t | Timing (only after equivalence) | measured | 1 thread, N=4096, d=128, d_ff=512, top-2, cf=1, median of 15: fwd eval loop/vec ms: E=4 17.0/27.0, E=16 20.9/30.2, E=64 41.8/30.0; fwd+bwd train: E=4 42.7/73.9, E=16 46.4/72.2, E=64 82.4/73.3. Vectorized is slower at E<=16 and faster at E=64 on this CPU. | `check3_bench_console.txt` |
| 4 | Solver validation, then IC/BC + held-out error | Solver **PASS**; PINN trains but is **not converged/unstable** | `physics_solver.py` (same PDE as the PINN). t=1 error vs analytic, dx halving 0.1 -> 0.00625: Lax-Wendroff L2 1.60e-2 -> 6.84e-5, observed order 1.92, 1.97, 1.98, **2.00** (max-norm 2.01); upwind order 0.75 -> **0.96**. Held-out set: 6724 FD grid nodes (161^2 grid, stride 4, t=0.25/0.5/0.75/1), FD vs exact rel L2 8.6e-4. PINN (PDE + 10 IC + 10 BC + 0.01 aux), seed 567885, 180 s, 12,582 steps: final held-out rel L2 vs FD **0.1425** (vs exact 0.1423), IC MSE 7.6e-4, BC MSE 4.5e-5, PDE residual MSE 1.0e-2. Trajectory unstable (held-out 0.32 at step 1000, 0.99 at 2500, 0.13 at 10000, 0.24 at 12500); top-1 expert usage [0, 0, 335, 8926] (collapse). | `check4_solver_console.txt`, `check4_pinn_train_console.txt` |
| 5 | Causal attention + MoE future independence | **PASS dropless / FAIL (leak) with capacity** | `causal=True` added to `MultiHeadAttention` -> `MoETransformerBlock` -> `DynamicMultiBlockTransformer`. 2 layers, 4 experts, top-2, B=4, L=24, 5 cut points x 3 perturbations. Dropless (`capacity_factor=None`), loop and vectorized, eval and train: 0/15 leaking trials, max change 0.0, d(past)/d(future) = 0.0 exactly, cross-sample 0.0. **Capacity routing leaks**: cf=1.0 fixed 12/15 trials, max change 0.366, cross-sample 0.185; cf=0.5 15/15, 0.605; legacy cf=1.0 12/15, 2.65 (legacy leaks across batch rows, not within row 0). Original non-causal model leaks 15/15 (0.554). | `check5_causal_console.txt` |

## Check 3 update: float32 parameter gradients now match at the stated tolerance
Root cause of the former strict xfail `test_bench_size_float32_param_grads_at_stated_tol`: the vectorized
path computed each expert's weight/bias gradient as a reduction over its padded `[cap, d]` buffer in
*priority* order, while the loop reduces over exactly the kept rows in *its* processing order (`nonzero()`
order, or priority order when the expert overflows). Different summation order/length -> different float32
rounding (up to 3.05e-5 at N=4096). Fix (`moe._BatchedExpertFFN`): buffer rows are laid out in the loop's
order; the forward and the input/activation gradients stay batched (row-wise, so already bit-identical);
expert weight/bias gradients are reduced per expert over exactly the first n_e rows with the same
`mm`/`sum` calls nn.Linear's autograd uses. Measured: float32 param-grad max abs diff loop vs vectorized is
now **0.0** (bit-identical) for all 8 small configs x {eval,train} and at N=4096 for E=4 (seed 4), 16, 64;
was 5.7e-6 / 3.05e-5 / 1.3e-5 / 4.3e-6. Tolerances unchanged; the xfail marker is removed, three float32
bench-size all-gradient tests and a double-backward (create_graph) test are added. `checks/check3_bench.py`
now asserts float32 parameter gradients too. Cost (measured, noisy shared CPU): vectorized fwd+bwd train is
~20-35% slower than before (per-expert weight-grad reductions instead of one padded bmm); fwd eval ~2-8%.
Loop / legacy / expand-engine paths are untouched: `sandbox.py --mode legacy` still prints Total Loss 1.5083,
and a 2-cycle `expand.py loop` from `init` gives a bit-identical checkpoint and identical log except wall seconds.
Consoles: `results/check3_fix_console.txt`.

## Code changes (all opt-in; defaults unchanged, the 47 pre-existing tests still pass)
- `moe.py`: `dispatch="vectorized"` path (fixed semantics only); `capacity_factor=None` = dropless; fixed/vectorized
  `last_stats` now also expose `topk_indices` and `kept_slots`. Legacy path untouched.
- `transformer.py` / `model.py`: `causal` and `dispatch` options plumbed through (also into `add_layer`).
- `physics_solver.py` (new): upwind + 2-D Lax-Wendroff FD solver with convergence study.
- `physics_simulation.py`: EXPLORATORY label; `heldout_reference()` / `heldout_error()`; `train(..., heldout=)` logs held-out error.
- `checks/`: historical spec, five check files, bench/training/real-checkpoint scripts, `run_all.sh`, raw consoles.

## Implications
- Causal MoE language modelling must use `capacity_factor=None` (dropless) or a per-position causal capacity; any
  finite capacity shared over the flattened batch lets future tokens change past outputs.
- Interpretation (not measured): the fp32 gradient mismatch is reduction-order rounding, not a dispatch bug.
