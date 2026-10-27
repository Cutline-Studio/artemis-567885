# Artemis 567885 — correctness review (iteration 1, 2026-10-23)

Every finding below was verified by running code, not by reading alone.
Reproduce with: `PYTHONPATH=. python3 tools/investigate_baseline.py` (it forces `legacy=True`,
i.e. the original behaviour). Numbers quoted are the real output of that script on the box
(torch 2.14.1+cpu, Python 3.13.5, 8 CPU threads).

Baseline `python sandbox.py` (original files, unmodified):
```
Total Loss Value: 1.5083
Enforced Drop-out Vector Count: 6
Input Structural Gradient Norm (\nabla_x L): 0.1581
Router Gate Matrix Gradient Norm (\nabla_W_g L): 0.0546
```

| # | Finding | Verdict | Evidence (script output) | Stage-1 fix (`legacy=False`) |
|---|---|---|---|---|
| 1 | Aux loss uses the **noisy** logits for the router probabilities and counts only the **top-1** slot (Switch-style), while routing is top-2. | Bug-ish / inconsistent. With top-2 routing the second slot is real load that the regulariser never sees; the noise makes aux stochastic for a fixed input. | Same input, two noise seeds → aux 1.0076 vs 1.0055. | `pe` from clean logits; `me` = fraction of all top-k assignments (÷k), so uniform optimum stays exactly 1.0. Test: `test_fixed_aux_matches_formula_over_all_slots`, `test_fixed_aux_is_noise_free_when_routing_is_stable`. |
| 2 | Capacity overflow is cut **first-come by token order** (`nonzero` order), not by gate priority. | Bug. Late tokens in the flattened batch are always the ones dropped, regardless of confidence; top-2 picks can evict top-1 picks. | 4 tokens all routed to expert 0, capacity 1, gate scores [0.1, 0.2, 5.0, 6.0] → kept token 0 (lowest score), dropped the 5.0 and 6.0 tokens. | Rank by slot (all top-1 picks first), then by router probability descending. Test: `test_fixed_capacity_prefers_highest_gate_score`. |
| 3a | Dropped tokens detected via `output.sum(-1) == 0`. | Bug (fragile). A processed token whose weighted expert output sums to 0 is misclassified. | Experts forced to output [1,-1,2,-2] (sum 0), nothing dropped → every processed output was overwritten by the input copy. | Explicit `processed_slots` mask; no heuristic. Test: `test_fixed_no_sum_zero_false_positive`. |
| 3b | Fallback copies the **input** into the MoE output; the block then adds it to the residual. | Bug (double counting). A dropped token leaves the block as `x + LN(x)`-ish (MoE input copy), not `x`. | For a dropped token, `moe_out == moe_input` → True. | Dropped tokens get a zero MoE output; the residual already carries them (standard Switch/GShard behaviour). Test: `test_fixed_dropped_tokens_get_zero_not_input_copy`. |
| 4 | No causal mask in attention. | Design gap. Fine for the set/regression sandbox, wrong for autoregressive byte modelling (planned stage). | Perturbing the last position changed the output at position 0 → True. | Not changed in stage 1 (planned: `causal=True` option, stage 3). |
| 5 | No positional information. | Design gap. The block is permutation-equivariant, so it cannot use order. | `block(permute(x)) == permute(block(x))` → True. | Planned: sinusoidal/learned positions option (stage 3). |
| 6 | Gradient to the gate flows only through the top-k softmax weights and the aux loss. | Works for top-2; **dead for top-1**: softmax over one logit is constantly 1.0. | top-2 task-only gate grad 0.0140, aux-only 0.1740; **top_k=1 task-only gate grad = 0.000000**. | Documented; tests assert task-loss gradient reaches the gate with top-2. Planned option: weight top-1 by full softmax prob (Switch). |
| 7 | `dropped_count` sums dropped (token, slot) **assignments** across top-k slots, so it can exceed the token count. | Misleading metric. | 8 tokens, top-2, capacity 2/expert → dropped_count = 12 (> 8 tokens). | Return value kept as assignments for compatibility; `layer.last_stats` adds `dropped_tokens` (tokens with no slot processed) and `capacity`. |
| 8 | Per-expert Python loop on CPU. | Performance. Each expert costs a `nonzero`, gather, two matmuls and a scatter; overhead grows with E. | 4096 tokens, d=128, d_ff=512, forward: run 1 E=4 13.4 ms, E=16 12.4 ms, E=64 21.7 ms; run 2 E=4 10.2 ms, E=16 9.1 ms, E=64 38.7 ms (noisy box, but E=64 consistently slowest even though active FLOPs are constant). | Planned: sort-based vectorized dispatch + benchmark script (stage 4). |

Other notes
- `x.view(-1, d)` in the original fails on non-contiguous inputs; the fixed path uses `reshape`.
- In-place `output_workspace[idx] += ...` with duplicate indices is not an issue here (each token appears at most once per expert), but the fixed path uses `index_add` anyway.
- Noise std is hard-coded to `1/num_experts` and is applied to the routing weights too (standard noisy-top-k); left as-is.
- `capacity_factor` is multiplied by `top_k`, so with CF=0.8 and top-2 each expert holds 1.6× its uniform share of tokens.

Original behaviour remains reproducible: `MoETransformerBlock(..., legacy=True)` /
`python sandbox.py --mode legacy` gives the identical baseline numbers (enforced by
`tests/test_moe.py::test_legacy_sandbox_reproduces_baseline_numbers`).
