# Artemis 567885

A tokenization-free mathematical repository implementing a high-performance, CPU-optimized **Top-K Sparse Mixture of Experts (MoE) Layer** embedded inside a **Modular Transformer Block**.

## Repository Architecture
- `moe.py`: Mathematical gating logic router, auxiliary balance loss, and capacity constraints tracking engine.
- `transformer.py`: Multi-Head Attention layer and primary integrated transformer sub-layer sequence orchestrator.
- `sandbox.py`: Verification harness evaluating forward numerical pipelines, capacity dropouts, and backwards backpropagation gradients.

## Execution Instruction
Execute the verification pipeline from the repository root directory:
```bash
python sandbox.py
```
