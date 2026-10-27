# Stage 8 — Switch top-1 gate weighting, router z-loss, expert-choice routing

Measured on 2026-10-24, 20:57–21:20 ET (America/New_York), CPU only (no GPU on the box),
torch 2.14.1+cpu. Sweep wall time **1399 s (23.3 min)** for 8 runs, 2 runs at a time × 2 torch
threads (4 threads total). Raw data: `runs/stage8/sweep.csv`, `runs/stage8/sweep.json`, and per
run `runs/stage8/<run>/{curve.csv,curve.jsonl,summary.json,samples.txt,console.txt}`.
Reproduce with `python3 tools/stage8_sweep.py` (aggregate only: `--report`).

## What was added (all options off by default)

| Option | Where | What it does |
|---|---|---|
| `gate_weighting="switch"` | `CapacitySparseMoELayer`, `MoETransformerBlock`, `DynamicMultiBlockTransformer` (incl. `add_layer`), `ArtemisModel`, `train_bytes.py --gate-weighting` | Each chosen expert is weighted by its **full** router probability `softmax(logits)[e]` (Switch Transformer) instead of a softmax over the K chosen logits (`"renorm"`, default). Fixes review #6: with top_k=1 the renorm weight is always 1.0, so the task loss sends no gradient to the gate. Loop and vectorized dispatch, any capacity mode. |
| router z-loss | every non-legacy layer stores `last_z_loss = mean_tokens(logsumexp(clean_logits)²)` (ST-MoE); `ArtemisModel(z_loss_coef=...)`, `model.last_z_loss` (sum over layers), `model.regularizer()`; `train_bytes.py --z-loss COEF` | Training loss becomes `CE + alpha·aux + coef·Σ_layers z`. Default coef 0: nothing is added to the loss (bit-identical training, checked). Computed on the clean (noise-free) logits. The legacy path also records it (before its noise draw, no RNG use) without changing outputs. |
| `routing="expert_choice"` | same classes, `train_bytes.py --routing expert_choice` | Each expert picks its top-C tokens by router probability within a group (whole flattened batch for `capacity_mode="global"`, one sequence for `"sequence"`), `C = ceil(n/E · CF · top_k)` (CF `None` → 1.0); a picked token is weighted by `softmax(logits)[e]`, so the gate gets task gradient. Tokens may get 0..E experts; `dropped_count` = tokens no expert picked. aux is still the token-choice formula and is exactly 1.0 (no gradient) because every expert gets exactly C tokens. |

`legacy=True` rejects the new weighting/routing options; `sandbox.py --mode legacy` still gives
Total Loss 1.5083.

### Expert choice is not causal — labelled leaky, not restricted

Whether position t is picked by an expert depends on the router scores of **every** token in
its group, including later positions. In a causal byte LM, a future byte can therefore change
past outputs (shown in `tests/test_stage8_routing.py::test_expert_choice_is_not_causal_and_flagged`).
Because the top-C pick is discrete, ∂past/∂future is still exactly 0, so a zero-Jacobian test
cannot certify causality here; the perturbation test does
(`test_expert_choice_leak_is_invisible_to_jacobian_check`). There is no causal-safe variant of
the per-group top-C pick without changing the algorithm (e.g. a learned per-expert threshold or
a prefix-only pick), so it is implemented as an option, `ArtemisModel.causal_safe` is `False`
for it, and `train_bytes.py` logs a warning and writes `leaky: true` to the summary. **Its row
below is LEAKY and is not a causal byte-LM result**: at eval, each 129-byte window is routed
using all of its bytes. (`sequence` grouping does stop leakage *across* sequences — tested.)
All other options stay causal: perturbation (allclose 1e-5) + exact zero Jacobian
∂past/∂future through 2 layers, train and eval, loop and vectorized.

## Settings (identical to the stage-7 sweep except top_k / routing options)

- Data: the stage-6 corpus at `/workspace/artemis-corpus-bench/corpus` — train = 5 books
  (11,984,086 B); held-out = `val` split (Austen ×2, author not in train).
- Model: `ArtemisModel`, d_model 128, 4 heads, d_ff 256, 3 layers × 4 experts, sinusoidal PE,
  causal mask, loop dispatch — 1,058,304 params for every row (top-1 rows run half the expert
  FLOPs per token).
- Training: seq_len 128, batch 32, AdamW lr 2e-3, 100 warmup + cosine, grad clip 1.0, fixed
  2000 steps, seed 567885, aux alpha 0.01, **dropless** token-choice routing (expert choice:
  CF 1.0, per-sequence groups → C = 65 of 129 tokens per expert).
- Eval: the same 256 held-out windows as stage 7 (32,768 predicted bytes), eval mode;
  unigram 4.5965 bpb.
- Reproducibility check: `top2_renorm` here reproduces stage-7 `a0.01_cfnone` exactly
  (1.98661 bpb, 1.37701 nats, same routed entropies, same sample text), so the stage-8 code
  doesn't change the default path.
- Single seed — differences of ~0.01 bpb are within what a different seed could plausibly
  change; no seed repeats were run. bpb is still falling at step 2000 in every run.
- `train_time_s` is wall time on a shared box; `top2_switch` and `top2_renorm_z1e-3` ran
  concurrently at ~520 s vs ~290 s for the same FLOPs elsewhere (other load, cause not
  determined). Runs are step-budgeted, so bpb doesn't depend on timing.

## Results (held-out bpb at step 2000, lower is better)

z columns: router z-loss summed over the 3 layers (train = mean over all steps; held-out =
final eval), logged for every run whether or not it is in the loss.

| run | top_k | weighting | routing | z coef | held-out bpb | Δ vs baseline | held-out nats | routed load entropy (mean; per layer) | train z | held-out z | train time s | leaky |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top2_renorm (baseline) | 2 | renorm | token choice | 0 | **1.9866** | — | 1.3770 | 0.9987; 0.9982 / 0.9985 / 0.9995 | 41.83 | 56.71 | 286 | no |
| top1_renorm (dead gate grad) | 1 | renorm | token choice | 0 | 2.1087 | +0.1221 | 1.4616 | 0.9994; 0.9996 / 0.9989 / 0.9998 | 7.53 | 7.72 | 223 | no |
| top1_switch | 1 | switch | token choice | 0 | 1.9936 | +0.0070 | 1.3818 | 0.9965; 0.9941 / 0.9978 / 0.9977 | 31.84 | 46.64 | 234 | no |
| top2_switch | 2 | switch | token choice | 0 | 1.9845 | −0.0021 | 1.3756 | 0.9983; 0.9970 / 0.9988 / 0.9990 | 40.74 | 55.96 | 527 | no |
| top2_renorm_z1e-3 | 2 | renorm | token choice | 1e-3 | **1.9767** | −0.0100 | 1.3701 | 0.9990; 0.9998 / 0.9982 / 0.9991 | 1.45 | 1.39 | 520 | no |
| top2_renorm_z1e-2 | 2 | renorm | token choice | 1e-2 | 1.9932 | +0.0065 | 1.3816 | 0.9993; 0.9992 / 0.9985 / 1.0000 | 0.31 | 0.23 | 304 | no |
| top1_switch_z1e-3 | 1 | switch | token choice | 1e-3 | 1.9977 | +0.0112 | 1.3847 | 0.9976; 0.9959 / 0.9974 / 0.9996 | 1.47 | 1.47 | 241 | no |
| ec_k2_cf1_seq_LEAKY | (2) | — | **expert choice** | 0 | 1.9849 | −0.0017 | 1.3758 | 1.0000 (by construction) | 40.96 | 55.09 | 269 | **YES** |

Held-out drop rate is 0 for every row (token choice is dropless; in the expert-choice run every
token was picked by ≥ 1 expert in training and at eval). Final aux (sum over 3 layers, uniform
= 3.0) is 2.997–3.023 for all rows.

Held-out bpb curve (steps 250 / 500 / 1000 / 1500 / 2000):

| run | 250 | 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|---|
| top2_renorm | 2.9998 | 2.4759 | 2.1550 | 2.0289 | 1.9866 |
| top1_renorm | 3.0815 | 2.6286 | 2.2998 | 2.1620 | 2.1087 |
| top1_switch | 3.0673 | 2.5075 | 2.1737 | 2.0436 | 1.9936 |
| top2_switch | 3.0330 | 2.4724 | 2.1694 | 2.0319 | 1.9845 |
| top2_renorm_z1e-3 | 2.9993 | 2.4727 | 2.1575 | 2.0234 | 1.9767 |
| top2_renorm_z1e-2 | 3.0312 | 2.4737 | 2.1772 | 2.0438 | 1.9932 |
| top1_switch_z1e-3 | 3.0714 | 2.5049 | 2.1735 | 2.0443 | 1.9977 |
| ec_k2_cf1_seq_LEAKY | 3.0158 | 2.4671 | 2.1491 | 2.0324 | 1.9849 |

## Review #6 gate gradient (`tools/stage8_gate_grad.py`, `runs/stage8/gate_grad_console.txt`)

Same setup as `tools/investigate_baseline.py` #6 (16-d, 4 experts, 32 tokens, task loss
`out.pow(2).mean()` only, no aux), fixed semantics:

| top_k | renorm | switch |
|---|---|---|
| 2 | 0.013825 | 0.014302 |
| 1 | **0.000000** | 0.015389 |

## Reading

- **Top-1 needs Switch weighting.** With renorm weights the top-1 gate is trained only by the
  aux loss (alpha 0.01), and the run ends 0.122 bpb behind top-2. Switch weighting closes that
  to +0.007 bpb behind top-2 while running half the expert FLOPs per token — the clearest
  effect in this sweep, well beyond the ~0.01 single-seed noise.
- **Switch at top-2** ≈ baseline (−0.002, within noise).
- **z-loss**: without it, the per-layer logsumexp² grows to ~14–19 on average (sum of 3 layers
  41.8 train / 56.7 held-out). Coefficient 1e-3 holds it at ~1.4 and gave the best row
  (1.9767, −0.010 vs baseline) — at the edge of the single-seed noise level, so not yet a
  confirmed gain; 1e-2 squeezes logits further (z 0.23) and is +0.007 worse. At top-1 Switch,
  z 1e-3 was +0.004 worse than without (noise level).
- **Expert choice** matched the baseline (1.9849, −0.002) with perfectly balanced load, but it
  is **leaky** (routing sees future bytes of the window), so this number is not a valid causal
  byte-LM result and must not be compared as one.

## Tests

`tests/test_stage8_routing.py` (51 tests): top-1 gate gets zero task gradient with renorm and
non-zero with switch (loop/vectorized, train/eval; and through `ArtemisModel`); switch weights =
full router probabilities (reference computation, top-2 not renormalised); default renorm
unchanged vs a reference; switch loop = vectorized (outputs, aux, drops, input/gate grads) for
dropless, global and sequence capacity, top-1/2; legacy rejects the options; bad values
rejected; z-loss = (ln E)² for a zero gate and = a pure-Python logsumexp² reference on clean
logits (token/expert choice, train/eval); z-loss sums over layers, `regularizer()` =
coef·z with gradient to every gate, and is a graph-free 0 by default; legacy block still gives
1.5083 and records z; expert choice picks each expert's top-C tokens (brute-force reference)
for global/sequence × CF None/0.5/1.5, drop count, experts-per-token, aux = 1.0, kept load;
expert-choice loop = vectorized (outputs, input/gate grads, train/eval); no cross-sequence
dependence with `sequence` groups; expert choice leaks (perturbation) and has a zero Jacobian
anyway; token-choice options stay causal (perturbation + exact zero Jacobian, both dispatches,
train/eval); `causal_safe` flags; `add_layer` keeps options; `train_bytes` flag defaults,
explicit defaults reproduce bit-identically, switch/z-loss/expert-choice runs and the leaky
label + z columns in curve/summary.
