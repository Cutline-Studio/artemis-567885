# Neural scaling laws engine

**Status:** built (formulas and planner), with a measured tiny CPU sweep. No GPU run.

`engines/scaling/`.

## Formulas
Constants are as published. They were entered from the papers without re-fetching them this
session, so **verify** them before relying on them.
- Kaplan et al. 2020 (arXiv:2001.08361): L(N) = (8.8e13 / N)^0.076 and L(D) = (5.4e13 / D)^0.095,
  where N is non-embedding parameters and D is tokens.
- Hoffmann et al. 2022, Chinchilla (arXiv:2203.15556), approach 3: L(N, D) = 1.69 + 406.4/N^0.34
  + 410.7/D^0.28.
- C ≈ 6ND. The compute-optimal rule of thumb is D ≈ 20N, so N = √(C/120).
  `chinchilla_optimal(C)` solves the fitted form in closed form. At C = 5.76e23 it gives N in the
  tens of billions, the same order as the paper's 70B (tested).
- These fits come from other models, tokenizers and data. They do **not** transfer to Artemis's
  byte-level loss.

## Budget planner
`plan_for_budget(usd, peak_flops, mfu, usd_per_hour)` turns dollars into FLOPs, then into the
compute-optimal N and D. Peak FLOPs, MFU and the hourly price are inputs you supply. None are
assumed here. It plans only: it never rents anything.

## Artemis's own curve: measured 2026-10-25, CPU, 300 steps each
Data: 64,457 bytes (the repo's .md docs plus problem sets), 90/10 split, sequence 64, batch 8,
AdamW 3e-3.

| params | eval loss (nats/byte) |
|---|---|
| 7,472 | 2.961 |
| 34,976 | 2.580 |
| 123,200 | 2.501 |
| 504,672 | 2.665 |
| 885,888 | 2.888 |

Fitting a power law gives r² = 0.03, so **no scaling law shows up at this scale**. With a fixed
300 steps the larger models are undertrained, and 64 KB of data is far too little. A real curve
needs a compute-matched sweep on the full corpus, which means a GPU run and Latoya's approval.
Plot: `docs/scaling_sweep.png`.
