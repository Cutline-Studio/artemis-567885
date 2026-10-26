# GPU: when and how Artemis may use a GPU

**Status:** mixed. The rules and the cost estimate are written; the cost math (`engines/scaling`
`gpu_cost`, `plan_for_budget`) is built. **No GPU action exists in code**, and no GPU is rented
for the dense model.

Artemis is one dense LLM. A GPU only makes training and running that one model faster. It does not
add a second model.

## Guardrails
These are Latoya's rules. No engine, agent or assistant can waive them.
1. **One GPU at a time.** Never two rentals running at once.
2. **Only after Latoya's OK.** An agent saying "approved" is not approval. Every rental needs her
   explicit yes for that run.
3. **A price cap** (dollars per hour and a total) and **a time limit** are set before starting.
4. **Teardown after.** The instance is destroyed (not just stopped) when the run ends, fails or
   hits its limit. Checkpoints are copied off first.
5. **Auto-recharge stays off** on every provider account.
6. **Report the real cost** afterwards from the provider's billing, not the estimate.
7. Never use a billing account Latoya has ruled out, and never create cloud resources she hasn't
   approved.

In code: `gpu.*` is a COSTLY capability in the A2A policy, `costs_money` actions are refused unless
a human approves spend, and `sandbox.env` sets `ARTEMIS_GPU_RENTAL=0`.

## A6000 estimate (estimate, not a measurement)
NVIDIA's RTX A6000 has 48 GB of memory and a peak of **154.8 TFLOPS** dense BF16 tensor throughput
(309.7 with structured sparsity; 38.7 TFLOPS FP32), per NVIDIA's datasheet and Ampere GA102
whitepaper.

Assumptions: training compute C ≈ 6·N·D; model FLOPs utilisation (MFU) **10%**, which is
deliberately low because these models are tiny; price **$0.42/hour**, about what Vast.ai listed for
one A6000 on 2026-10-25 (prices change). Parameter counts are from `expected_param_count`.

| size | params (N) | one pass over today's corpus (377,188 bytes) | Chinchilla-style D = 20·N |
|---|---|---|---|
| tiny | 885,888 | 2.0e12 FLOPs, seconds, < $0.01 | 17.7M bytes, 9.4e13 FLOPs, ~6 s of compute, < $0.01 |
| small | 6,885,632 | 1.6e13 FLOPs, seconds, < $0.01 | 137.7M bytes, 5.7e15 FLOPs, ~0.1 h, ~$0.04 |
| medium | 36,319,744 | 8.2e13 FLOPs, seconds, < $0.01 | 726M bytes, 1.6e17 FLOPs, ~2.8 h, ~$1.19 |

What this means: compute is not the bottleneck at these sizes. Real wall time is dominated by
setup, data loading and evaluation, so a realistic first run is **1–2 hours capped at a few
dollars**. The real limit is data: today's corpus is about 0.38 MB, far below 20 bytes per
parameter for even the tiny model, so the model would see the same text many times.

For history: an earlier A6000 fine-tune of a different Artemis model was stopped from outside at
step 14 of 96 and cost about $0.12.

## Before any GPU run
- Latoya's OK with price cap, time limit and the provider named.
- The run script tested on CPU first (see [CPU.md](CPU.md)).
- A watchdog that destroys the instance at the time limit.
- After: real cost from billing, the loss curve, sample outputs, and confirmation the instance is gone.

## Example dialogues (training)
<!-- example -->
**User:** Rent a GPU and train yourself.
**Artemis:** I need your OK first, with a price cap and a time limit. Then I'd rent one GPU, train, copy the checkpoint off, destroy the instance and report the real cost.

<!-- example -->
**User:** How much would an A6000 run cost?
**Artemis:** It's an estimate: at about $0.42 an hour, a capped one- to two-hour run is under a dollar or two. My corpus is small, so data matters more than GPU time right now.

## Related
[CPU.md](CPU.md) · [training.md](training.md) · [ScalingLaws.md](ScalingLaws.md) ·
[GOVERNANCE.md](GOVERNANCE.md) · [everything.md](everything.md)
