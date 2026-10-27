# Stage 8b — physics sandbox follow-ups (measured)

`physics_simulation.py`: coordinate-routed SIREN mixture-of-experts PINN (4 experts, top-2,
hidden 64, depth 3, ω0 30) for 2-D linear advection on [-1,1]² × [0,1] with IC + inflow BC,
compared against the analytic travelling wave. All numbers below are real measurements from
`runs/stage8b/` (CPU, torch 2.14.1+cpu, Python 3.13.5, 2026-10-25 12:01–12:20 ET).

## New options (all off by default)

| option | `train(...)` / CLI | what it does |
|---|---|---|
| lr | `lr=` / `--lr` (default 1e-3) | Adam learning rate (already a train arg; now on the CLI) |
| cosine schedule | `lr_schedule="cosine"`, `min_lr_frac` / `--lr-schedule cosine --min-lr-frac` | lr·(f + (1−f)·½(1+cos(π·step/max_steps))), defined over `max_steps` |
| load-balance weight | `alpha=` / `--alpha` (default 0.01) | weight of the switch-style aux loss |
| noisy routing | `noise_std=` / `--noise-std` (default 0) | N(0, σ²) added to the router logits before top-k, **train mode only**; aux loss uses the clean softmax; `evaluate`, `error_vs_t`, `heldout_error` and the final-loss pass run in eval mode (clean routing) |
| causal residual weighting | `causal_eps=`, `n_t_bins=16` / `--causal-eps --t-bins` | Wang, Sankaran & Perdikaris (2022): residual points binned by t; L_i = mean r² in bin i; w_i = exp(−ε·Σ_{j<i} L_j) (detached); L_pde = mean_i w_i·L_i |
| time curriculum | `curriculum_frac`, `curriculum_t0=0.1` / `--curriculum-frac --curriculum-t0` | residual and BC points use t ∈ [0, t_max], t_max = min(1, t0 + (1−t0)·step/(frac·max_steps)) |
| metrics | `expert_usage_stats`, `error_vs_t` | `experts_used` (top-1 count > 0) and normalised usage entropy added to `evaluate`; rel. L2 / max-abs error vs analytic on a 41×41 grid at t = 0, 0.1, …, 1 |
| threads | `threads=` / `--threads` | default keeps `min(8, cpu_count)` |

Runner: `tools/stage8b_physics_study.py NAME --steps 6358 --threads T [k=v ...]`; aggregation:
`tools/stage8b_summarize.py` → `runs/stage8b/summary.{csv,json}`.

## Baseline: how it was measured, and reproduction

Iteration 1 (`runs/physics/final.json`): `python physics_simulation.py --seconds 120`, defaults
(lr 1e-3 constant, alpha 0.01, seed 567885, 1024 points, 8 threads); the 120 s budget ended at
**6358 steps**; rel. L2 at t=1 = **0.249662** (41×41 grid at t=1 vs analytic), top-1 usage
[0, 0, 1129, 8132] on the 21³ grid.

All stage-8b runs use a fixed **6358-step** budget instead of a time budget (so results don't
depend on box load). Re-running the defaults with 8 threads (`runs/stage8b/baseline_8thr`)
reproduced iteration 1 **exactly**: rel. L2 t=1 0.249662, t=0.5 0.1568, t=0 0.054542, PDE MSE
3.3986e-4, usage [0, 0, 1129, 8132] (103.6 s). `tests/test_stage8b_physics.py` checks this.

**Variance warning.** The same default config with 2 threads (only the BLAS thread count
differs, so float summation order changes) gave rel. L2 t=1 **0.6272** for seed 567885, and
0.0268 / 0.1824 / 0.0481 for seeds 1 / 2 / 3. At lr 1e-3 training has loss spikes (e.g. PDE
MSE 4e-5 at step 1750 → 0.11 at step 3500 in the 2-thread baseline), so the final number
depends on where in a spike the run stops. A single baseline number is therefore not a
reliable comparison point; configs were compared over 4 seeds (567885, 1, 2, 3) at 2 threads
(4 runs in parallel), plus the best config at the exact iteration-1 setting (8 threads, seed 567885).

## Results

### Per config (2 threads, 6358 steps, sorted by mean rel. L2 at t=1)

| config | options | rel. L2 t=1 per seed (567885, 1, 2, 3) | mean | max | mean experts used (of 4) | mean usage entropy |
|---|---|---|---|---|---|---|
| **lr1e-4_a0.1_n1_cur0.5** | lr 1e-4, alpha 0.1, noise 1.0, curriculum 0.5 | 0.0082, 0.0098, 0.0077, 0.0071 | **0.0082** | 0.0097 | 4 | 0.892 |
| lr1e-4_n1 | lr 1e-4, noise 1.0 | 0.0117, 0.0104, 0.0127, 0.0099 | 0.0112 | 0.0127 | 4 | 0.885 |
| lr1e-4_a0.1_n1 | lr 1e-4, alpha 0.1, noise 1.0 | 0.0111, 0.0128, 0.0126, 0.0096 | 0.0115 | 0.0128 | 4 | 0.902 |
| alpha0.1_noise1 | alpha 0.1, noise 1.0 (lr 1e-3) | 0.0719, 0.1310, 0.0481, 0.0239 | 0.0687 | 0.1310 | 1.75 | 0.317 |
| curriculum0.5 | curriculum 0.5 (lr 1e-3) | 0.0425, 0.2600, 0.0538, 0.0536 | 0.1025 | 0.2600 | 1.75 | 0.115 |
| alpha1.0 | alpha 1.0 (1 seed) | 0.2128 | 0.2128 | — | 4 | 0.798 |
| baseline | defaults | 0.6272, 0.0268, 0.1824, 0.0481 | 0.2211 | 0.6272 | 1.75 | 0.192 |
| cosine | cosine from 1e-3 (1 seed) | 0.5894 | 0.5894 | — | 4 | 0.865 |
| lr1e-4_a0.1 | lr 1e-4, alpha 0.1 | 0.2663, 0.6522, 0.8412, 0.6344 | 0.5985 | 0.8412 | 4 | 0.886 |
| lr1e-4 | lr 1e-4 | 0.3064, 0.6144, 0.8710, 0.8638 | 0.6639 | 0.8710 | 4 | 0.945 |
| lr1e-4_causal1 | lr 1e-4, causal ε 1 | 0.2894, 0.6085, 0.8732, 0.8910 | 0.6655 | 0.8910 | 4 | 0.945 |
| cosine3e-4 | cosine from 3e-4 | 0.8978, 0.6186, 0.8814, 0.8440 | 0.8104 | 0.8978 | 4 | 0.899 |
| causal_eps100 | causal ε 100 (lr 1e-3, 1 seed) | 1.0124 | 1.0124 | — | 4 | 0.693 |
| causal_eps10 | causal ε 10 (lr 1e-3, 1 seed) | 1.0178 | 1.0178 | — | 4 | 0.634 |

### Exact iteration-1 setting (8 threads, seed 567885, 6358 steps)

| run | rel. L2 t=0 | t=0.5 | t=1 | all | top-1 usage | experts used | entropy | wall |
|---|---|---|---|---|---|---|---|---|
| baseline_8thr (defaults) | 0.0545 | 0.1568 | **0.2497** | 0.1731 | [0, 0, 1129, 8132] | 2 | 0.267 | 103.6 s |
| best_8thr (lr 1e-4, alpha 0.1, noise 1.0, curriculum 0.5) | 0.0073 | 0.0070 | **0.0077** | 0.0074 | [3015, 2507, 2579, 1160] | 4 | 0.963 | 104.6 s |

best_8thr final losses (fresh 8192-point sample): PDE MSE 3.10e-5, IC MSE 2.48e-6, BC MSE 5.10e-7
(baseline_8thr: 3.40e-4, 1.53e-4, 1.18e-5).

### Error vs t (rel. L2 vs analytic, 41×41 grid)

| t | baseline_8thr | best_8thr | baseline mean (2 thr, 4 seeds) | lr1e-4_a0.1_n1_cur0.5 mean | lr1e-4_n1 mean | curriculum0.5 mean | lr1e-4 mean |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.0545 | 0.0073 | 0.1004 | 0.0065 | 0.0080 | 0.0444 | 0.0053 |
| 0.1 | 0.0671 | 0.0071 | 0.1209 | 0.0065 | 0.0079 | 0.0444 | 0.0085 |
| 0.2 | 0.0943 | 0.0072 | 0.1404 | 0.0068 | 0.0084 | 0.0445 | 0.0250 |
| 0.3 | 0.1110 | 0.0073 | 0.1568 | 0.0071 | 0.0088 | 0.0443 | 0.0638 |
| 0.4 | 0.1314 | 0.0071 | 0.1721 | 0.0070 | 0.0094 | 0.0443 | 0.1398 |
| 0.5 | 0.1568 | 0.0070 | 0.1835 | 0.0071 | 0.0097 | 0.0469 | 0.3354 |
| 0.6 | 0.1741 | 0.0068 | 0.1932 | 0.0071 | 0.0103 | 0.0584 | 0.4601 |
| 0.7 | 0.2012 | 0.0067 | 0.2020 | 0.0071 | 0.0107 | 0.0734 | 0.5578 |
| 0.8 | 0.2170 | 0.0068 | 0.2092 | 0.0074 | 0.0108 | 0.0872 | 0.6081 |
| 0.9 | 0.2402 | 0.0071 | 0.2155 | 0.0078 | 0.0110 | 0.0969 | 0.6415 |
| 1.0 | 0.2497 | 0.0077 | 0.2211 | 0.0082 | 0.0112 | 0.1025 | 0.6639 |

Per-run error-vs-t rows: `runs/stage8b/<run>/error_vs_t.json` (and the `err_vs_t` column of `summary.csv`).

### All runs

| run | seed | threads | options | rel L2 t=0 | t=0.5 | **t=1** | all | top-1 usage (21³ grid) | used | entropy | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha0.1_noise1 | 567885 | 2 | `{"alpha": 0.1, "noise_std": 1.0}` | 0.0399 | 0.0577 | **0.0719** | 0.0580 | [3521, 0, 0, 5740] | 2 | 0.479 | 75.9 |
| alpha0.1_noise1_s1 | 1 | 2 | `{"alpha": 0.1, "noise_std": 1.0}` | 0.0569 | 0.1120 | **0.1310** | 0.1048 | [2963, 3436, 2862, 0] | 3 | 0.790 | 77.6 |
| alpha0.1_noise1_s2 | 2 | 2 | `{"alpha": 0.1, "noise_std": 1.0}` | 0.0526 | 0.0489 | **0.0481** | 0.0499 | [9261, 0, 0, 0] | 1 | 0.000 | 76.3 |
| alpha0.1_noise1_s3 | 3 | 2 | `{"alpha": 0.1, "noise_std": 1.0}` | 0.0192 | 0.0204 | **0.0239** | 0.0213 | [0, 0, 0, 9261] | 1 | 0.000 | 72.9 |
| alpha1.0 | 567885 | 2 | `{"alpha": 1.0}` | 0.0927 | 0.1855 | **0.2128** | 0.1716 | [2193, 3731, 57, 3280] | 4 | 0.798 | 76.4 |
| baseline | 567885 | 2 | `defaults` | 0.1496 | 0.4742 | **0.6272** | 0.4622 | [9261, 0, 0, 0] | 1 | 0.000 | 76.4 |
| baseline_s1 | 1 | 2 | `defaults` | 0.0259 | 0.0275 | **0.0268** | 0.0267 | [8782, 479, 0, 0] | 2 | 0.147 | 71.9 |
| baseline_s2 | 2 | 2 | `defaults` | 0.1742 | 0.1826 | **0.1824** | 0.1798 | [3828, 4932, 501, 0] | 3 | 0.619 | 75.5 |
| baseline_s3 | 3 | 2 | `defaults` | 0.0520 | 0.0498 | **0.0481** | 0.0500 | [0, 9261, 0, 0] | 1 | 0.000 | 67.0 |
| baseline_8thr | 567885 | 8 | `defaults` | 0.0545 | 0.1568 | **0.2497** | 0.1731 | [0, 0, 1129, 8132] | 2 | 0.267 | 103.6 |
| best_8thr | 567885 | 8 | `{"lr": 0.0001, "alpha": 0.1, "noise_std": 1.0, "curriculum_frac": 0.5}` | 0.0073 | 0.0070 | **0.0077** | 0.0074 | [3015, 2507, 2579, 1160] | 4 | 0.963 | 104.6 |
| causal_eps10 | 567885 | 2 | `{"causal_eps": 10.0}` | 0.6201 | 0.9990 | **1.0178** | 0.8979 | [544, 3310, 49, 5358] | 4 | 0.634 | 80.8 |
| causal_eps100 | 567885 | 2 | `{"causal_eps": 100.0}` | 0.5642 | 0.9454 | **1.0124** | 0.8636 | [122, 659, 4346, 4134] | 4 | 0.693 | 83.4 |
| cosine | 567885 | 2 | `{"lr_schedule": "cosine"}` | 0.0006 | 0.2873 | **0.5894** | 0.3786 | [3976, 1111, 815, 3359] | 4 | 0.865 | 77.8 |
| cosine3e-4 | 567885 | 2 | `{"lr": 0.0003, "lr_schedule": "cosine"}` | 0.0007 | 0.3272 | **0.8978** | 0.5517 | [3752, 1128, 1739, 2642] | 4 | 0.934 | 83.1 |
| cosine3e-4_s1 | 1 | 2 | `{"lr": 0.0003, "lr_schedule": "cosine"}` | 0.0010 | 0.2891 | **0.6186** | 0.3942 | [2421, 2944, 3460, 436] | 4 | 0.885 | 75.0 |
| cosine3e-4_s2 | 2 | 2 | `{"lr": 0.0003, "lr_schedule": "cosine"}` | 0.0009 | 0.4410 | **0.8814** | 0.5691 | [1457, 4208, 1473, 2123] | 4 | 0.923 | 89.3 |
| cosine3e-4_s3 | 3 | 2 | `{"lr": 0.0003, "lr_schedule": "cosine"}` | 0.0006 | 0.3018 | **0.8440** | 0.5175 | [3037, 4016, 395, 1813] | 4 | 0.852 | 74.7 |
| curriculum0.5 | 567885 | 2 | `{"curriculum_frac": 0.5}` | 0.0362 | 0.0384 | **0.0425** | 0.0391 | [8867, 0, 0, 394] | 2 | 0.127 | 74.9 |
| curriculum0.5_s1 | 1 | 2 | `{"curriculum_frac": 0.5}` | 0.0319 | 0.0429 | **0.2600** | 0.1533 | [759, 0, 8089, 413] | 3 | 0.333 | 74.1 |
| curriculum0.5_s2 | 2 | 2 | `{"curriculum_frac": 0.5}` | 0.0554 | 0.0538 | **0.0538** | 0.0544 | [9261, 0, 0, 0] | 1 | 0.000 | 71.3 |
| curriculum0.5_s3 | 3 | 2 | `{"curriculum_frac": 0.5}` | 0.0539 | 0.0525 | **0.0536** | 0.0534 | [0, 9261, 0, 0] | 1 | 0.000 | 69.6 |
| lr1e-4 | 567885 | 2 | `{"lr": 0.0001}` | 0.0052 | 0.1649 | **0.3064** | 0.2009 | [3001, 1886, 1937, 2437] | 4 | 0.987 | 79.3 |
| lr1e-4_s1 | 1 | 2 | `{"lr": 0.0001}` | 0.0045 | 0.2872 | **0.6144** | 0.3916 | [2198, 2883, 3362, 818] | 4 | 0.928 | 83.8 |
| lr1e-4_s2 | 2 | 2 | `{"lr": 0.0001}` | 0.0045 | 0.4694 | **0.8710** | 0.5713 | [1683, 4153, 1891, 1534] | 4 | 0.932 | 81.5 |
| lr1e-4_s3 | 3 | 2 | `{"lr": 0.0001}` | 0.0071 | 0.4202 | **0.8638** | 0.5547 | [1452, 3262, 1251, 3296] | 4 | 0.935 | 73.2 |
| lr1e-4_a0.1 | 567885 | 2 | `{"lr": 0.0001, "alpha": 0.1}` | 0.0072 | 0.0574 | **0.2663** | 0.1573 | [3229, 3206, 1427, 1399] | 4 | 0.944 | 79.5 |
| lr1e-4_a0.1_s1 | 1 | 2 | `{"lr": 0.0001, "alpha": 0.1}` | 0.0037 | 0.2763 | **0.6522** | 0.4090 | [2239, 3108, 3818, 96] | 4 | 0.810 | 82.1 |
| lr1e-4_a0.1_s2 | 2 | 2 | `{"lr": 0.0001, "alpha": 0.1}` | 0.0065 | 0.4397 | **0.8412** | 0.5481 | [864, 4268, 2206, 1923] | 4 | 0.899 | 72.6 |
| lr1e-4_a0.1_s3 | 3 | 2 | `{"lr": 0.0001, "alpha": 0.1}` | 0.0044 | 0.3879 | **0.6344** | 0.4293 | [2027, 2953, 582, 3699] | 4 | 0.893 | 78.4 |
| lr1e-4_a0.1_n1 | 567885 | 2 | `{"lr": 0.0001, "alpha": 0.1, "noise_std": 1.0}` | 0.0076 | 0.0092 | **0.0111** | 0.0094 | [2798, 1679, 3143, 1641] | 4 | 0.970 | 83.0 |
| lr1e-4_a0.1_n1_s1 | 1 | 2 | `{"lr": 0.0001, "alpha": 0.1, "noise_std": 1.0}` | 0.0071 | 0.0088 | **0.0128** | 0.0099 | [2324, 2967, 3957, 13] | 4 | 0.782 | 74.9 |
| lr1e-4_a0.1_n1_s2 | 2 | 2 | `{"lr": 0.0001, "alpha": 0.1, "noise_std": 1.0}` | 0.0077 | 0.0113 | **0.0126** | 0.0107 | [978, 4184, 2136, 1963] | 4 | 0.911 | 89.0 |
| lr1e-4_a0.1_n1_s3 | 3 | 2 | `{"lr": 0.0001, "alpha": 0.1, "noise_std": 1.0}` | 0.0076 | 0.0101 | **0.0096** | 0.0092 | [1649, 2540, 1374, 3698] | 4 | 0.946 | 80.3 |
| lr1e-4_a0.1_n1_cur0.5 | 567885 | 2 | `{"lr": 0.0001, "alpha": 0.1, "noise_std": 1.0, "curriculum_frac": 0.5}` | 0.0068 | 0.0071 | **0.0082** | 0.0074 | [3026, 2507, 2582, 1146] | 4 | 0.962 | 84.0 |
| lr1e-4_a0.1_n1_cur0.5_s1 | 1 | 2 | `{"lr": 0.0001, "alpha": 0.1, "noise_std": 1.0, "curriculum_frac": 0.5}` | 0.0090 | 0.0096 | **0.0097** | 0.0094 | [2234, 2969, 3994, 64] | 4 | 0.797 | 74.3 |
| lr1e-4_a0.1_n1_cur0.5_s2 | 2 | 2 | `{"lr": 0.0001, "alpha": 0.1, "noise_std": 1.0, "curriculum_frac": 0.5}` | 0.0052 | 0.0060 | **0.0077** | 0.0064 | [873, 4088, 3044, 1256] | 4 | 0.880 | 87.1 |
| lr1e-4_a0.1_n1_cur0.5_s3 | 3 | 2 | `{"lr": 0.0001, "alpha": 0.1, "noise_std": 1.0, "curriculum_frac": 0.5}` | 0.0049 | 0.0058 | **0.0071** | 0.0060 | [1538, 2481, 1274, 3968] | 4 | 0.928 | 76.4 |
| lr1e-4_causal1 | 567885 | 2 | `{"lr": 0.0001, "causal_eps": 1.0}` | 0.0068 | 0.1500 | **0.2894** | 0.1882 | [2958, 2027, 1770, 2506] | 4 | 0.986 | 83.5 |
| lr1e-4_causal1_s1 | 1 | 2 | `{"lr": 0.0001, "causal_eps": 1.0}` | 0.0044 | 0.2960 | **0.6085** | 0.3907 | [2258, 2909, 3506, 588] | 4 | 0.902 | 75.3 |
| lr1e-4_causal1_s2 | 2 | 2 | `{"lr": 0.0001, "causal_eps": 1.0}` | 0.0054 | 0.4725 | **0.8732** | 0.5733 | [1762, 4141, 1853, 1505] | 4 | 0.933 | 88.9 |
| lr1e-4_causal1_s3 | 3 | 2 | `{"lr": 0.0001, "causal_eps": 1.0}` | 0.0048 | 0.4243 | **0.8910** | 0.5698 | [1474, 2934, 1621, 3232] | 4 | 0.959 | 75.3 |
| lr1e-4_n1 | 567885 | 2 | `{"lr": 0.0001, "noise_std": 1.0}` | 0.0082 | 0.0101 | **0.0117** | 0.0101 | [2714, 846, 3647, 2054] | 4 | 0.923 | 77.9 |
| lr1e-4_n1_s1 | 1 | 2 | `{"lr": 0.0001, "noise_std": 1.0}` | 0.0082 | 0.0091 | **0.0104** | 0.0093 | [2338, 3050, 3870, 3] | 4 | 0.779 | 79.8 |
| lr1e-4_n1_s2 | 2 | 2 | `{"lr": 0.0001, "noise_std": 1.0}` | 0.0097 | 0.0106 | **0.0127** | 0.0110 | [2016, 4227, 562, 2456] | 4 | 0.874 | 73.7 |
| lr1e-4_n1_s3 | 3 | 2 | `{"lr": 0.0001, "noise_std": 1.0}` | 0.0059 | 0.0092 | **0.0099** | 0.0085 | [1648, 2345, 1741, 3527] | 4 | 0.964 | 73.5 |

(Usage = top-1 expert counts on the 21×21×21 evaluation grid, 9261 points, clean routing.
Entropy = normalised Shannon entropy of that histogram, 1 = even. Some JSON files store a
collapsed run's entropy as `-0.0`; fixed to `0.0` in the code after the runs.)

## Findings

1. **Done criterion met.** At the exact iteration-1 setting (8 threads, seed 567885, 6358
   steps) rel. L2 at t=1 went 0.2497 → **0.0077** (−97%) with lr 1e-4 + alpha 0.1 + noisy
   routing σ 1.0 + time curriculum 0.5. Over 4 seeds at 2 threads the same config gave
   0.0071–0.0098 (mean 0.0082) vs the defaults' 0.0268–0.6272 (mean 0.2211). Every seed is
   below 0.250.
2. **Noisy routing is what matters, together with lr 1e-4.** lr 1e-4 alone (mean 0.664) and
   lr 1e-4 + alpha 0.1 (0.599) are *worse* than the default at t=1. They fit t≈0 well (rel.
   L2 ~0.005) but error grows steadily with t (0.34 at t=0.5, 0.66 at t=1): the solution
   doesn't carry forward in time within the budget. Adding noise σ 1.0 to the same lr 1e-4
   run gives 0.0099–0.0127 at t=1 with flat error vs t. Alpha 0.1 on top of noise adds
   nothing (0.0115 vs 0.0112). The curriculum adds a small further gain (0.0082). At lr 1e-3, noise + alpha 0.1
   helped too (mean 0.069) but stayed spiky and still collapsed in 2 of 4 seeds.
   Why noise helps so much is not isolated here. One hypothesis: random routing blends
   neighbouring experts, which smooths the hard top-k seams in (x, y, t).
3. **Expert collapse fixed, but collapse was not the cause of the error.** Defaults: 1.75 of 4
   experts used on average (the iteration-1 run used 2: [0, 0, 1129, 8132]). All lr 1e-4 runs
   use all 4 (best_8thr [3015, 2507, 2579, 1160], entropy 0.963). But the collapsed default
   seeds 1 and 3 (1–2 experts) still reached 0.027 / 0.048, and the balanced lr 1e-4 runs without noise
   were the worst. Balance alone does not lower the error.
4. **Cosine schedule did not help in this budget.** From 1e-3: 0.589 (1 seed). From 3e-4:
   mean 0.810. Like lr 1e-4, these fit t=0 almost perfectly (~0.001) and fail at later t.
5. **Causal residual weighting did not help as implemented.** At lr 1e-3, ε 10 / 100 gave
   rel. L2 ≈ 1.0 at t=1 (the late bins get ~0 weight and lr-1e-3 spikes reset progress). With lr
   1e-4, ε 1 was the same as lr 1e-4 alone (mean 0.666). The simple linear time curriculum
   worked better: at lr 1e-3 it gave mean 0.103 with 3 of 4 seeds at 0.043–0.054. ε was not
   annealed or tuned further.
6. **Default runs are highly sensitive to thread count and seed** (0.027–0.627 at t=1).
   Single-run comparisons with the defaults are unreliable.

## Reproducibility / compute

- Seeds: 567885 (iteration-1 seed), 1, 2, 3. The seed sets init, sampling and router noise
  (global torch RNG; noise draws only when σ > 0, so the default RNG stream is unchanged).
- Fixed 6358 steps, 1024 residual points (+512 IC, +512 BC) per step, Adam.
- Wall clock: about 18.5 min total (12:01–12:20 ET; 44 runs at 2 threads, 4 in parallel, plus two 8-thread runs);
  sum of per-run wall time 3639 s (67–105 s per run).
- Defaults unchanged: `baseline_8thr` = iteration 1 bit-for-bit; tests check that the defaults
  equal explicitly-off options and that the default loss is the plain mean residual.
- Raw files: `runs/stage8b/<run>/{final.json,error_vs_t.json,train_log.jsonl,console.txt}`,
  `runs/stage8b/summary.{csv,json}`, `summary_console.txt`, `full_suite_console.txt`,
  `sandbox_console.txt`.
