# Artemis 567885 language benchmark, protocol v2 (development benchmark)

Code commit `96672bea9a38f7582d1deeba901f6cc59730cc75` · corpus manifest sha256 `ab94040953aeb890566c3f89fa11b54ed068ac12106aa41b4bcf93c2773c8397` · protocol sha256 `6ac83f9287fea26bc2b72780eed97ee5896a2e4077514e2719eaf94f2646f4e5`

Budgets are matched in optimizer steps (5,000) and byte positions processed, not in compute; wall time is reported next to every result. This corpus is a development benchmark: its test scores have been viewed, so confirmation belongs to the untouched confirmation set.

## Bits per byte (cross-entropy only; aux loss excluded)

| arm | seed | val bpb | test bpb | selected step | train bytes (selected / total) | train s | total s |
|---|---|---|---|---|---|---|---|
| dense | 42 | 2.973 | 3.038 | 5000 | 10,238,605 / 10,238,605 | 144 | 278 |
| dense | 1 | 2.915 | 2.981 | 5000 | 10,238,465 / 10,238,465 | 142 | 280 |
| dense | 2 | 3.025 | 3.093 | 5000 | 10,239,077 / 10,239,077 | 89 | 172 |
| moe-learned | 42 | 2.822 | 2.898 | 5000 | 10,238,605 / 10,238,605 | 285 | 450 |
| moe-learned | 1 | 2.730 | 2.804 | 5000 | 10,238,465 / 10,238,465 | 274 | 438 |
| moe-learned | 2 | 2.695 | 2.776 | 5000 | 10,239,077 / 10,239,077 | 178 | 284 |
| moe-sinusoidal | 42 | 2.653 | 2.733 | 5000 | 10,238,605 / 10,238,605 | 287 | 451 |
| moe-sinusoidal | 1 | 2.603 | 2.695 | 5000 | 10,238,465 / 10,238,465 | 277 | 441 |
| moe-sinusoidal | 2 | 2.662 | 2.760 | 5000 | 10,239,077 / 10,239,077 | 181 | 287 |
| moe-fourier | 42 | 2.701 | 2.782 | 5000 | 10,238,605 / 10,238,605 | 283 | 457 |
| moe-fourier | 1 | 2.668 | 2.750 | 5000 | 10,238,465 / 10,238,465 | 180 | 284 |
| moe-fourier | 2 | 2.641 | 2.730 | 5000 | 10,239,077 / 10,239,077 | 186 | 289 |

Unigram (train counts, add-one): val 4.529, test 4.545.

Active = shared parameters plus selected-expert parameters per byte (embedding table, position module, attention, norms, router gates and head all counted; 2 of 4 experts per MoE layer). An accounting convention, not measured FLOPs or memory traffic. Dense: active = total.

| arm | params total | shared + selected-expert params per byte | val bpb mean (sd) | test bpb mean (sd) | mean train s |
|---|---|---|---|---|---|
| dense | 37,888 | 37,888 | 2.971 (0.055) | 3.037 (0.056) | 125 |
| moe-learned | 63,296 | 46,528 | 2.749 (0.065) | 2.826 (0.064) | 246 |
| moe-sinusoidal | 59,744 | 42,976 | 2.640 (0.032) | 2.730 (0.032) | 248 |
| moe-fourier | 59,744 | 42,976 | 2.670 (0.030) | 2.754 (0.026) | 216 |

## Paired differences by seed (first arm minus second; negative = first arm lower bpb)

| comparison | split | seed 42 | seed 1 | seed 2 | mean | min | max |
|---|---|---|---|---|---|---|---|
| moe-fourier − moe-sinusoidal | val | +0.048 | +0.064 | -0.021 | +0.030 | -0.021 | +0.064 |
| moe-fourier − moe-sinusoidal | test | +0.048 | +0.055 | -0.029 | +0.024 | -0.029 | +0.055 |
| moe-fourier − moe-learned | val | -0.121 | -0.063 | -0.054 | -0.079 | -0.121 | -0.054 |
| moe-fourier − moe-learned | test | -0.116 | -0.054 | -0.046 | -0.072 | -0.116 | -0.046 |
| moe-sinusoidal − moe-learned | val | -0.168 | -0.127 | -0.033 | -0.110 | -0.168 | -0.033 |
| moe-sinusoidal − moe-learned | test | -0.164 | -0.109 | -0.017 | -0.097 | -0.164 | -0.017 |
| moe-learned − dense | val | -0.151 | -0.184 | -0.330 | -0.222 | -0.330 | -0.151 |
| moe-learned − dense | test | -0.140 | -0.176 | -0.317 | -0.211 | -0.317 | -0.140 |
| moe-sinusoidal − dense | val | -0.319 | -0.311 | -0.363 | -0.331 | -0.363 | -0.311 |
| moe-sinusoidal − dense | test | -0.304 | -0.285 | -0.333 | -0.308 | -0.333 | -0.285 |
| moe-fourier − dense | val | -0.272 | -0.247 | -0.384 | -0.301 | -0.384 | -0.247 |
| moe-fourier − dense | test | -0.256 | -0.231 | -0.363 | -0.283 | -0.363 | -0.231 |

Only moe-fourier − moe-sinusoidal is a parameter-matched comparison (same feature width, same projection, identical parameter counts). Comparisons with moe-learned (larger: 128-row position table) and with dense (different architecture and parameter count) are qualified accordingly.

Three seeds: preliminary evidence. The sign agreement across seeds and the range are reported, not a confidence interval. Development benchmark: test scores of this corpus have been viewed.

## Routing diagnostics (separate from bpb)

Load = fraction of top-2 assignments of real (non-padding) bytes per expert. Entropy normalized by log 4. Aux = Switch-style balance loss summed over both layers (1.0 per layer at perfect balance, so 2.0 total).

| arm | seed | layer | val load (selected ckpt) | val entropy | test entropy | train entropy, last 1000 steps | train aux, last 1000 steps |
|---|---|---|---|---|---|---|---|
| moe-learned | 42 | 0 | 0.24, 0.30, 0.24, 0.22 | 0.995 | 0.995 | 0.994 | 2.047 |
| | | 1 | 0.29, 0.24, 0.24, 0.22 | 0.997 | 0.996 | 0.998 | |
| moe-learned | 1 | 0 | 0.24, 0.24, 0.26, 0.26 | 0.999 | 0.999 | 0.999 | 2.027 |
| | | 1 | 0.28, 0.29, 0.23, 0.19 | 0.989 | 0.989 | 0.989 | |
| moe-learned | 2 | 0 | 0.18, 0.26, 0.23, 0.33 | 0.982 | 0.981 | 0.986 | 2.059 |
| | | 1 | 0.21, 0.27, 0.25, 0.27 | 0.996 | 0.996 | 0.995 | |
| moe-sinusoidal | 42 | 0 | 0.23, 0.29, 0.23, 0.24 | 0.996 | 0.997 | 0.998 | 2.007 |
| | | 1 | 0.29, 0.23, 0.25, 0.23 | 0.996 | 0.996 | 0.998 | |
| moe-sinusoidal | 1 | 0 | 0.21, 0.24, 0.28, 0.27 | 0.995 | 0.995 | 0.998 | 2.015 |
| | | 1 | 0.22, 0.25, 0.25, 0.28 | 0.997 | 0.997 | 0.998 | |
| moe-sinusoidal | 2 | 0 | 0.28, 0.21, 0.26, 0.26 | 0.996 | 0.996 | 0.996 | 1.981 |
| | | 1 | 0.24, 0.22, 0.24, 0.30 | 0.995 | 0.996 | 0.998 | |
| moe-fourier | 42 | 0 | 0.28, 0.20, 0.23, 0.29 | 0.991 | 0.990 | 0.992 | 2.015 |
| | | 1 | 0.27, 0.21, 0.26, 0.26 | 0.997 | 0.997 | 0.998 | |
| moe-fourier | 1 | 0 | 0.22, 0.24, 0.27, 0.27 | 0.998 | 0.997 | 0.998 | 2.002 |
| | | 1 | 0.23, 0.24, 0.24, 0.29 | 0.997 | 0.997 | 0.998 | |
| moe-fourier | 2 | 0 | 0.19, 0.28, 0.28, 0.25 | 0.992 | 0.991 | 0.995 | 2.041 |
| | | 1 | 0.29, 0.22, 0.22, 0.27 | 0.995 | 0.994 | 0.995 | |

Dropped assignments across all MoE runs: training 0, evaluation 0 (zero in both, as required for dropless routing).

## Run ledger

- 2026-10-23T07:12Z dense seed 42: completed (exit 0, 281.9 s)
- 2026-10-23T07:15Z moe-learned seed 42: completed (exit 0, 454.1 s)
- 2026-10-23T07:20Z moe-sinusoidal seed 42: completed (exit 0, 454.8 s)
- 2026-10-23T07:23Z moe-fourier seed 42: completed (exit 0, 461.2 s)
- 2026-10-23T07:25Z dense seed 1: completed (exit 0, 284.4 s)
- 2026-10-23T07:30Z moe-learned seed 1: completed (exit 0, 441.8 s)
- 2026-10-23T07:32Z moe-sinusoidal seed 1: completed (exit 0, 445.1 s)
- 2026-10-25T04:39Z dense seed 2: interrupted — process stopped during workspace inactivity after last ledger write 2026-10-23T07:32Z; 6 validation records, no metrics.json; outputs kept on disk
- 2026-10-25T04:39Z moe-fourier seed 1: interrupted — process stopped during workspace inactivity; 5 validation records, no metrics.json; outputs kept on disk
- 2026-10-25T04:39Z moe-learned seed 2: not_launched — batch driver stopped before launch
- 2026-10-25T04:39Z moe-sinusoidal seed 2: not_launched — batch driver stopped before launch
- 2026-10-25T04:39Z moe-fourier seed 2: not_launched — batch driver stopped before launch
- 2026-10-25T04:43Z dense seed 2: completed (exit 0, 183.7 s)
- 2026-10-25T04:45Z moe-fourier seed 1: completed (exit 0, 295.5 s)
- 2026-10-25T04:48Z moe-learned seed 2: completed (exit 0, 287.2 s)
- 2026-10-25T04:50Z moe-sinusoidal seed 2: completed (exit 0, 289.7 s)
- 2026-10-25T04:53Z moe-fourier seed 2: completed (exit 0, 291.7 s)

## Notes and caveats
- **Selected checkpoint is step 5,000 in all 12 runs**: validation bpb was still falling at the end of the
  budget, so these are budget-limited results, not converged ones.
- **Wall times are not comparable across sessions.** Seed-42 and seed-1 runs (except moe-fourier seed 1) ran
  on 2026-10-23; the five resumed runs ran on 2026-10-25 under lighter machine load (e.g. dense 89 s vs
  ~143 s of training). Within a session, MoE training took about 2× the dense time per step on CPU.
- **Resume was exact.** dense seed 2 and moe-fourier seed 1 were rerun from scratch at the frozen commit;
  every validation bpb they share with the interrupted partial runs (steps 0–5000 and 0–4000) is
  bit-identical. Interrupted outputs are kept in `runs_v2/_interrupted/`.
- **Position result (parameter-matched):** Fourier features did not improve on sinusoidal positions here:
  test bpb +0.048 and +0.055 worse on seeds 42 and 1, −0.029 better on seed 2 (mean +0.024). Mixed sign
  across three seeds is not evidence of a difference in either direction.
- Learned positions trailed both fixed encodings on every seed while having more parameters; the MoE arms
  beat dense on every seed at about 2× its CPU time per step. Both comparisons differ in architecture or
  parameter count and are not position effects.
- The confirmation set (45 books, 36 new authors) has not been scored.
