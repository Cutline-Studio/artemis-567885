# Stage 7 — alpha × capacity_factor sweep (byte-level Gutenberg task)

Measured on 2026-10-24, 17:59–18:38 ET (America/New_York), CPU only (no GPU on the box),
torch 2.14.1+cpu. Sweep wall time **2289 s (38.2 min)** for 20 runs. Raw data:
`runs/stage7/sweep.csv`, `runs/stage7/sweep.json`, and per run
`runs/stage7/<run>/{curve.csv,curve.jsonl,summary.json,samples.txt,console.txt}`.
Reproduce with `python3 tools/stage7_sweep.py` (aggregate only: `--report`).

## Causal safety: per-sequence capacity (`capacity_mode="sequence"`)

The existing capacity routing (`capacity_mode="global"`, still the default) ranks assignments
over the flattened `B*T` batch, so a future byte (or another sequence) can displace a past
byte — the leak documented in check 5. Every capacity-limited row below therefore uses the
new `capacity_mode="sequence"` (moe.py): each sequence gets its own per-expert capacity
`ceil(T / E * CF * top_k)`, and assignments are served first-come-first-served in
(position, slot) order, so keeping position t depends only on positions ≤ t of the same
sequence. Tests (`tests/test_stage7_capacity.py`) check drop counts and kept slots against a
brute-force reference, loop vs. vectorized parity (outputs, drops, gradients), bit-exact
future-independence and zero Jacobian ∂past/∂future in train and eval mode through attention +
2 MoE layers, independence from other sequences, and that global mode still leaks.
**None of the results below are leaky.** Cost of the causal ordering: when an expert
overflows, it is the later positions in the sequence that are dropped.

## Settings (identical for all 20 runs except alpha and CF)

- Data: the stage-6 corpus at `/workspace/artemis-corpus-bench/corpus` — train = 5 books
  (11,984,086 B); held-out = `val` split (Austen ×2, 1,230,205 B, author not in train).
- Model: `ArtemisModel`, d_model 128, 4 heads, d_ff 256, 3 layers × 4 experts, top-2,
  sinusoidal PE, causal mask, loop dispatch — 1,058,304 params (smaller than the stage-6
  default of 5.40M so the sweep fits the time budget).
- Training: seq_len 128 (windows of 129 bytes), batch 32, AdamW lr 2e-3 (weight decay 0.01),
  100 warmup steps then cosine to 0.1×, grad clip 1.0, **fixed 2000 steps** (8.19M training
  bytes), seed 567885, loss = next-byte CE + alpha × aux (aux summed over the 3 layers; uniform
  routing gives 3.0).
- Eval: the same 256 fixed, evenly spaced, non-overlapping held-out windows for every run
  (32,768 predicted bytes), eval mode (no routing noise), capacity applied as in training.
  Unigram baseline on those bytes: 4.5965 bpb; init: 8.29 bpb.
- Parallelism: 4 runs at a time, 2 torch threads each. The CF=1.5 wave ran ~3× slower in wall
  time (937–967 s vs. 287–356 s for the others; cause not determined, likely other load on the
  shared box). Runs are step-budgeted, so the numbers don't depend on timing.
- Metrics: held-out bpb / loss at step 2000; drop rate = dropped (token, slot) assignments /
  all assignments, summed over layers (train: mean over all 2000 steps; held-out: final eval);
  expert load entropy = Shannon entropy of each layer's expert-load histogram over the
  held-out windows divided by ln 4 (1.0 = uniform, 0 = one expert), "routed" = router's top-2
  choices before capacity, "kept" = assignments actually processed.
- Single seed — differences of ~0.01 bpb are within what a different seed could plausibly
  change; no seed repeats were run.

## Held-out bpb grid (lower is better)

| alpha \ CF | 0.8 | 1.0 | 1.25 | 1.5 | none |
|---|---|---|---|---|---|
| 0 | 2.1301 | 2.0241 | 1.9964 | 1.9921 | 1.9959 |
| 0.01 | 2.1390 | 2.0214 | 1.9910 | 1.9841 | 1.9866 |
| 0.1 | 2.1387 | 2.0204 | 1.9856 | 1.9855 | 1.9849 |
| 0.3 | 2.1507 | 2.0252 | 1.9897 | 1.9809 | 1.9906 |

Mean over alpha per CF: 0.8 → 2.1396, 1.0 → 2.0228, 1.25 → 1.9907, 1.5 → 1.9857, dropless → 1.9895.
Mean over CF per alpha: 0 → 2.0277, 0.01 → 2.0244, 0.1 → 2.0230, 0.3 → 2.0274.

## Full table

| alpha | CF | held-out bpb | held-out loss (nats) | train drop rate | held-out drop rate | load entropy routed (mean; L1/L2/L3) | load entropy kept | final aux (sum of 3 layers) | train time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8 | 2.1301 | 1.4765 | 19.88% | 19.68% | 0.9973 (0.9958 / 0.9975 / 0.9987) | 1.0000 | 3.0102 | 293 |
| 0 | 1.0 | 2.0241 | 1.4030 | 6.66% | 5.65% | 0.9947 (0.9873 / 0.9984 / 0.9985) | 0.9977 | 3.0449 | 336 |
| 0 | 1.25 | 1.9964 | 1.3838 | 1.41% | 0.77% | 0.9903 (0.9835 / 0.9992 / 0.9882) | 0.9913 | 3.0645 | 342 |
| 0 | 1.5 | 1.9921 | 1.3808 | 0.20% | 0.06% | 0.9856 (0.9733 / 0.9976 / 0.9859) | 0.9857 | 3.1570 | 953 |
| 0 | none (dropless) | 1.9959 | 1.3835 | 0.00% | 0.00% | 0.9703 (0.9484 / 0.9973 / 0.9653) | 0.9703 | 3.2857 | 352 |
| 0.01 | 0.8 | 2.1390 | 1.4827 | 19.64% | 19.57% | 0.9994 (0.9997 / 0.9985 / 0.9999) | 1.0000 | 3.0037 | 303 |
| 0.01 | 1.0 | 2.0214 | 1.4011 | 4.62% | 4.07% | 0.9990 (0.9978 / 0.9995 / 0.9997) | 0.9996 | 3.0026 | 344 |
| 0.01 | 1.25 | 1.9910 | 1.3800 | 0.23% | 0.14% | 0.9990 (0.9994 / 0.9980 / 0.9998) | 0.9991 | 3.0079 | 334 |
| 0.01 | 1.5 | 1.9841 | 1.3753 | 0.02% | 0.01% | 0.9986 (0.9995 / 0.9970 / 0.9994) | 0.9986 | 3.0165 | 962 |
| 0.01 | none (dropless) | 1.9866 | 1.3770 | 0.00% | 0.00% | 0.9987 (0.9982 / 0.9985 / 0.9995) | 0.9987 | 3.0199 | 344 |
| 0.1 | 0.8 | 2.1387 | 1.4824 | 19.53% | 19.55% | 0.9999 (0.9999 / 0.9999 / 1.0000) | 1.0000 | 2.9980 | 287 |
| 0.1 | 1.0 | 2.0204 | 1.4004 | 3.94% | 3.84% | 0.9998 (0.9999 / 0.9997 / 0.9999) | 0.9999 | 3.0013 | 341 |
| 0.1 | 1.25 | 1.9856 | 1.3763 | 0.09% | 0.09% | 0.9998 (0.9996 / 0.9997 / 1.0000) | 0.9998 | 3.0090 | 322 |
| 0.1 | 1.5 | 1.9855 | 1.3763 | 0.02% | 0.02% | 0.9998 (0.9998 / 0.9996 / 1.0000) | 0.9998 | 3.0202 | 963 |
| 0.1 | none (dropless) | 1.9849 | 1.3758 | 0.00% | 0.00% | 0.9999 (0.9999 / 0.9997 / 1.0000) | 0.9999 | 3.0113 | 343 |
| 0.3 | 0.8 | 2.1507 | 1.4908 | 19.50% | 19.54% | 0.9999 (0.9999 / 0.9998 / 1.0000) | 1.0000 | 3.0038 | 302 |
| 0.3 | 1.0 | 2.0252 | 1.4038 | 3.81% | 3.82% | 0.9999 (1.0000 / 0.9996 / 0.9999) | 1.0000 | 2.9966 | 334 |
| 0.3 | 1.25 | 1.9897 | 1.3792 | 0.08% | 0.11% | 0.9999 (0.9998 / 0.9999 / 1.0000) | 0.9999 | 3.0057 | 338 |
| 0.3 | 1.5 | **1.9809** | 1.3730 | 0.02% | 0.01% | 0.9999 (0.9999 / 0.9999 / 0.9999) | 0.9999 | 3.0019 | 937 |
| 0.3 | none (dropless) | 1.9906 | 1.3798 | 0.00% | 0.00% | 0.9999 (0.9998 / 1.0000 / 1.0000) | 0.9999 | 3.0038 | 338 |

## Findings

- **Best config: alpha 0.3, CF 1.5 — 1.9809 bpb** (loss 1.3730 nats, 0.02% train drops).
  Best dropless row: alpha 0.1, 1.9849 bpb. All CF ≥ 1.25 rows lie within 1.9809–1.9964,
  i.e. the top of the table is within ~0.016 bpb, about the size of single-seed noise.
- **Capacity factor dominates.** CF 0.8 cannot fit the load: with T = 129, E = 4, top-2 the
  per-sequence capacity is 52 slots/expert, 208 < 258 assignments, so ≥ 19.4% of assignments
  must drop; measured 19.5–19.9% (train) and the bpb is 0.13–0.16 worse than dropless at the same alpha. CF 1.0
  (capacity 65, no forced drops) still drops 3.8–6.7% in training because the load is not
  perfectly even, and costs 0.03–0.04 bpb. CF 1.25 drops 0.08–1.4% and CF 1.5 ≤ 0.2%; both match
  dropless (CF 1.5 was slightly better than dropless for 3 of 4 alphas, by ≤ 0.01 bpb).
- **Alpha has a small effect on bpb** (per-alpha means 2.023–2.028) but a clear effect on drops
  and balance: at CF 1.0, train drop rate falls 6.66% (alpha 0) → 4.62% (0.01) → 3.94% (0.1) →
  3.81% (0.3); at CF 1.25, 1.41% → 0.23% → 0.09% → 0.08%. alpha 0.3 is the worst alpha at
  CF 0.8 and 1.0 (2.1507, 2.0252) but the best at CF 1.5.
- **Expert load entropy**: no expert collapse at this scale even without the aux loss — routed
  entropy at alpha 0 is 0.9703 (dropless; layer 1 is lowest at 0.9484) to 0.9973; any alpha ≥
  0.01 brings every layer to ≥ 0.997, and alpha ≥ 0.1 to ≥ 0.9995. Under tight capacity the
  *kept* load is uniform by construction (1.0000 at CF 0.8).
- Recommended default from this sweep: alpha 0.01–0.1 with CF 1.25–1.5 in
  `capacity_mode="sequence"` (or dropless) — essentially no quality cost vs. dropless while
  bounding expert work. Caveats: one seed, 2000 steps of a 1.06M-param model; held-out bpb was
  still falling at step 2000 in every run (best == final).
