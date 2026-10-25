# Artemis 567885

A tokenization-free mathematical repository implementing a high-performance, CPU-optimized **Top-K Sparse Mixture of Experts (MoE) Layer** embedded inside a **Modular Transformer Block**.

## Repository Architecture
- `moe.py`: Mathematical gating logic router, auxiliary balance loss, and capacity constraints tracking engine. `legacy=True` reproduces the original behaviour; the default applies the fixes in `docs/REVIEW.md`.
- `transformer.py`: Multi-Head Attention layer and primary integrated transformer sub-layer sequence orchestrator.
- `sandbox.py`: Verification harness evaluating forward numerical pipelines, capacity dropouts, and backwards backpropagation gradients.
- `tests/`: pytest suite.
- `docs/REVIEW.md`: verified correctness findings. `PLAN.md`: staged roadmap. `PROGRESS.md`: iteration log.

## Setup (CPU)
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install pytest
```

## Execution Instruction
Execute the verification pipeline from the repository root directory:
```bash
python sandbox.py               # legacy + fixed modes
python sandbox.py --mode legacy # original numbers exactly
python -m pytest -q
```

## Expand engine (stage 2)
```bash
python expand.py init                       # writes config.json (single source of truth)
python supervisor.py --cycles 10 --timeout 300
```
- `model.py`: `DynamicMultiBlockTransformer` (num_layers from config; aux and dropped summed over layers; `add_layer` = exact-identity init; `add_expert` = copy + noise of the most-loaded expert, gate grows one row).
- `expand.py`: X → sin(X) regression on random vectors. Rules per cycle: dropped > 0 → capacity_factor += 0.05; aux > 1.2 → alpha × 1.5; loss stalls (< 1 % relative improvement) → add an expert, or a new layer if the previous expert addition did not help. Caps (`config.json` → `caps`): CF ≤ 2.5 (`capacity_factor_max`), experts ≤ 16 per layer (`num_experts_max`; a stall adds the expert to the deepest layer still below 16, layers already above it keep their trained experts but stop growing; if every layer is capped a new layer is added), alpha ≤ 1.0, layers ≤ 12 (`num_layers_max`). Checkpoint in `runs/checkpoint.pt`, every cycle in `runs/expand_log.jsonl`.
- `supervisor.py`: backup to `.backup_checkpoint/`, mutate config, `ast.parse`, subprocess with timeout, failure parsing (`runs/supervisor_log.jsonl`, `runs/runtime_errors.log`), shape auto-correct against the checkpoint with one validation re-run, rollback and a different mutation path otherwise.

## Physics sandbox
```bash
python physics_simulation.py --seconds 120   # results in runs/physics/final.json
```
