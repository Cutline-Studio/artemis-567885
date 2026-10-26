# Accelerate: 20 ways Artemis learns faster

Code: [`accel/`](accel/) · tests: `tests/test_accel.py` · trainer: `python -m artemis_dense.accel.trainer`.
Every method has one flag in `accel/config.py` (`AccelConfig`). The registry is `accel.METHODS`.

Speedups below are **estimates** from public practice unless the row says *measured*. They
multiply only roughly, and none of them is a promise of "1000x". The one measured comparison on
this computer is in [Measured A/B](#measured-ab-cpu-same-wall-time).

Status: **built+tested** runs on CPU today and has a test; **approval-needed** has code that
refuses to run until Latoya says OK; **planned** is an interface only.

## The 20 methods

| # | method | flag | status | speedup (estimate) | code |
|---|---|---|---|---|---|
| 1 | Single GPU (A6000/A100) | `device=cuda`, `gpu_approved` | approval-needed, not run | 50-100x vs this CPU | `device.resolve_device` |
| 2 | bf16 mixed precision | `precision=bf16\|auto` | built+tested | ~2x on GPU; on this CPU the autotuner picks whichever is faster | `device.autocast` |
| 3 | Fast attention (PyTorch SDPA; FlashAttention kernel when present) | `attention=flash\|efficient\|math\|auto` | built+tested | 2-3x on long sequences (GPU) | `device.attention_ctx` |
| 4 | `torch.compile` | `compile=true` | built+tested (falls back to eager here) | 1.3-2x per step | `device.maybe_compile` |
| 5 | Multi-GPU DDP via `torchrun` | `ddp=true`, `ddp_approved` | approval-needed, not run | ~N x with N GPUs | `device.setup_ddp`, `device.torchrun_command` |
| 6 | BPE tokenizer (Latoya's `tokenizer.py`) | `tokenizer=bpe` | built+tested | 3-4x more text per step | `data.get_codec`, `data.train_bpe` |
| 7 | Sequence packing | `pack=true` | built+tested | ~1.5x on padded chat data | `data.pack_documents`, `data.pack_sft` |
| 8 | Sequence-length warmup | `seq_warmup=true` | built+tested | 1.1-1.3x (cheaper early steps) | `data.seq_len_at` |
| 9 | Start from open pretrained weights | `init_pretrained=<local dir>`, `pretrained_approved`, `license_ack` | approval-needed, not run | 100x+; **no longer from scratch** | `pretrained.load_local` |
| 10 | Distillation from teacher logits | `distill=true`, `teacher_path` | built+tested (local teacher) | ~10x with a strong teacher | `distill.kd_loss` |
| 10b | Distillation from teacher-generated text | `teacher_generate=true` | planned, approval-needed | depends on the teacher | `distill.teacher_generate` |
| 11 | Grow the model while it learns | `grow=true` | built+tested | 1.2-2x | `growth.grow` |
| 12 | LoRA adapters | `lora=true` | built+tested | fast skill updates of a trained model; not for from-scratch | `lora.apply_lora` |
| 13 | Quality filtering | `quality_filter=true` | built+tested | 2-5x per token on web text | `data.quality_filter` |
| 14 | Exact + MinHash near-duplicate removal | `dedup=true` | built+tested | data-dependent | `data.dedup` |
| 15 | Synthetic verified math/physics problems | `synthetic=true` | built+tested | data-dependent | `data.synthetic_problems` |
| 16 | Curriculum (easy to hard) | `curriculum=true` | built+tested | 1.1-1.5x | `data.curriculum_order`, `data.competence` |
| 17 | Scaling-law sizing | `scaling_size=true` | built+tested | avoids wasted compute | `schedule.choose_size` |
| 18 | LR warmup + cosine, LR range finder | `lr_find=true`, `warmup_frac` | built+tested | 1.2-2x | `schedule.lr_at`, `schedule.lr_range_test` |
| 19 | Mistake-first replay | `mistake_replay=true`, `mistakes_dir` | built+tested | data-dependent | `replay.MistakeReplay` |
| 20 | Self-play best-of-N (verified + heart score) | `self_play=true` | built+tested | needs a model that is already partly right | `selfplay.best_of_n` |

## How each one works

1. **Single GPU.** `resolve_device("cuda")` raises `ApprovalRequired` unless `gpu_approved=True`.
   Needs a price cap and a time limit from Latoya first; one GPU at a time, torn down after.
2. **bf16.** `torch.autocast(..., bfloat16)` around the forward pass. `precision=auto` times a
   few real steps in fp32 and bf16 and keeps the faster one.
3. **Fast attention.** The model already calls `F.scaled_dot_product_attention`; the flag pins the
   kernel (`flash`, `efficient`, `math`). `auto` times them like bf16.
4. **torch.compile.** Compiles at start; if the compiler can't build (on this box Inductor fails
   because `Python.h` is not installed), it falls back to eager and the status says so.
5. **DDP.** `torchrun_command(n, args)` prints the launch line; `setup_ddp` wraps the model and
   refuses without `ddp_approved`. Never run here (it opens a process group).
6. **BPE.** Uses `artemis_dense/tok.py` + Latoya's `tokenizer.py`. If no `tokenizer_path` is given
   the trainer trains a 4096-token BPE on the **train split only** inside the run directory.
7. **Packing.** Whole documents joined into full rows (no padding); for chat data `pack_sft` packs
   masked records (attention may see the previous record in the row; noted trade-off).
8. **Sequence-length warmup.** Linear ramp from `seq_min` to `context_len` over `seq_warmup_frac`.
9. **Pretrained init.** Loads a LOCAL Llama/Qwen2/Mistral-layout directory (config.json +
   safetensors/bin), maps names, and converts q/k rows from rotate-half to Artemis' interleaved RoPE.
   There is no download code. Needs `pretrained_approved` and `license_ack` in
   {`llama3`, `qwen2.5`, `mistral`}; see the license checklist in `pretrained.LICENSES` (summaries to
   re-check on each model card, not legal advice). The result is marked `from_scratch: false`.
10. **Distillation.** `(1-alpha) * CE + alpha * T^2 * KL(teacher || student)` against a local Artemis
    teacher checkpoint with the same vocabulary. Teacher-generated text from a big open model is
    planned and needs approval (weights or API access, and a license that allows distillation).
11. **Growth.** Start with `grow_start_layers` and add function-preserving layers (zeroed output
    projections) spread between `grow_at[0]` and `grow_at[-1]` of the run; optional FFN widening.
    The optimizer is rebuilt, keeping Adam state for surviving weights.
12. **LoRA.** Freezes the base, wraps `wq`/`wv` with rank-r adapters (B starts at zero), and
    `merge_lora` folds them back in.
13. **Quality filter.** Length, letter ratio, duplicate-line ratio, compressibility and word length.
    Runs on the train split only.
14. **Dedup.** Exact (normalised sha1) then MinHash-LSH (64 permutations, 16 bands) at a Jaccard
    threshold (default 0.8). Train split only.
15. **Synthetic problems.** New problems from the math and physics engines; only engine-verified
    answers are kept, and any question that appears in the held-out split is dropped.
16. **Curriculum.** Documents sorted by a length x character-entropy difficulty score; training
    samples from the easiest `competence(t)` share (sqrt pacing to 100% at `curriculum_full_at`).
17. **Scaling-law sizing.** Measures tokens/s for each ladder size, estimates D for the budget and
    picks the size with the lowest Chinchilla-fit L(N, D) from `engines/scaling` (the fit comes from
    big models, so at this scale it is a heuristic).
18. **LR.** Warmup + cosine to 10% of peak; `lr_find` runs a 30-step range test on a throwaway copy
    and uses the steepest-descent LR (capped at min-loss LR / 3).
19. **Mistake-first replay.** Recorded mistakes from `engines/mistakes.py` (context + verified fix)
    are replayed first, then the hardest (highest-loss) windows seen so far fill `replay_frac` of
    each batch.
20. **Self-play.** For engine-checked problems, sample N answers; keep one only if the engine says it
    is correct and the heart engine scores every value >= 0; among those keep the one the model
    finds most likely. An untrained model usually keeps nothing, and the log says so.

## Presets

- `baseline`: what `artemis_dense/train.py` stage 1 does today (small, bytes, context 512,
  batch 16, LR 1e-3, 100-step warmup + cosine).
- `fast`: every method that runs on CPU with no approval: 2, 3 (auto), 4, 6, 7, 8, 11, 13-20.
  Off on purpose: 10 (no trained local teacher exists yet), 12 (LoRA slows from-scratch learning;
  use it for fine-tuning), and 1, 5, 9 (approval-needed).
- `gpu_full`: adds the GPU and teacher methods; refuses to run until each approval flag is set.

```
python -m artemis_dense.accel.trainer --preset fast --minutes 15 --run my_run --threads 4
python -m artemis_dense.accel.trainer --preset fast --set lora=true --set teacher_path=\"ckpt.pt\" --set distill=true ...
```

The time budget includes everything a preset does (tokenizer training, filtering, probes, LR range
test, growth, self-play). Only validation passes are paused out. Validation uses the same held-out
split as `train.py` (never filtered) and reports bits-per-byte, which is comparable across byte
and BPE vocabularies.

## Measured A/B (CPU, same wall time)

RESULTS_PLACEHOLDER

## What needs Latoya's OK

- **GPU (methods 1, 5):** a price cap and a time limit per GPU; one GPU at a time; teardown after.
- **Pretrained weights (method 9):** which model, permission to download it, and the license read
  on its model card. It also means Artemis is no longer from scratch.
- **Teacher model (method 10b):** which teacher, download or API access, and a license that
  allows distillation.

## Related
[training.md](training.md) · [CPU.md](CPU.md) · [GPU.md](GPU.md) · [ScalingLaws.md](ScalingLaws.md) ·
[Mistakes.md](Mistakes.md) · [Heart.md](Heart.md) · [Mathematics.md](Mathematics.md) · [Physics.md](Physics.md)
