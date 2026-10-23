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
