"""Accelerated-learning stack for the Artemis dense model (see ../Accelerate.md).

METHODS is the registry: one entry per method with its AccelConfig flag, status and module.
Status values: "built+tested" (runs on CPU now), "approval-needed" (code exists, refuses to run
without Latoya's OK), "planned" (interface only). Speedups are ESTIMATES unless measured.
"""
from .config import AccelConfig, PRESETS

METHODS = [
    (1, "Single GPU (A6000/A100)", "device=cuda + gpu_approved", "approval-needed", "50-100x vs this CPU (estimate)", "device.resolve_device"),
    (2, "bf16 mixed precision", "precision=bf16|auto", "built+tested", "~2x on GPU (estimate); slower on this CPU (measured)", "device.autocast"),
    (3, "Fast attention (SDPA / FlashAttention)", "attention=flash|efficient|math|auto", "built+tested", "2-3x on long sequences on GPU (estimate)", "device.attention_ctx"),
    (4, "torch.compile", "compile=true", "built+tested", "1.3-2x per step (estimate); falls back to eager on this box", "device.maybe_compile"),
    (5, "Multi-GPU DDP (torchrun)", "ddp=true + ddp_approved", "approval-needed", "~Nx with N GPUs (estimate)", "device.setup_ddp"),
    (6, "BPE tokenizer", "tokenizer=bpe", "built+tested", "3-4x more text per step (estimate)", "data.get_codec / data.train_bpe"),
    (7, "Sequence packing", "pack=true", "built+tested", "~1.5x for padded SFT data (estimate)", "data.pack_documents / data.pack_sft"),
    (8, "Sequence-length warmup", "seq_warmup=true", "built+tested", "cheaper early steps (estimate 1.1-1.3x)", "data.seq_len_at"),
    (9, "Init from open pretrained weights", "init_pretrained=<local dir> + pretrained_approved + license_ack", "approval-needed", "100x+ (estimate); breaks from-scratch purity", "pretrained.load_local"),
    (10, "Distillation (teacher logits)", "distill=true + teacher_path", "built+tested", "~10x with a strong teacher (estimate)", "distill.kd_loss"),
    (10.1, "Distillation (teacher-generated text)", "teacher_generate=true", "planned", "needs teacher weights/API + license", "distill.teacher_generate"),
    (11, "Model growth (depth/FFN width)", "grow=true", "built+tested", "1.2-2x (estimate)", "growth.grow"),
    (12, "LoRA adapters", "lora=true", "built+tested", "fast skill updates; not for from-scratch", "lora.apply_lora"),
    (13, "Quality filtering", "quality_filter=true", "built+tested", "2-5x per token on web data (estimate)", "data.quality_filter"),
    (14, "Exact + MinHash dedup", "dedup=true", "built+tested", "data-dependent (estimate)", "data.dedup"),
    (15, "Synthetic verified problems", "synthetic=true", "built+tested", "data-dependent (estimate)", "data.synthetic_problems"),
    (16, "Curriculum (easy->hard)", "curriculum=true", "built+tested", "1.1-1.5x (estimate)", "data.curriculum_order"),
    (17, "Scaling-law sizing", "scaling_size=true", "built+tested", "avoids wasted compute (estimate)", "schedule.choose_size"),
    (18, "LR warmup+cosine + range finder", "lr_find=true, warmup_frac", "built+tested", "1.2-2x (estimate)", "schedule.lr_range_test"),
    (19, "Mistake-first replay", "mistake_replay=true", "built+tested", "data-dependent (estimate)", "replay.MistakeReplay"),
    (20, "Self-play best-of-N", "self_play=true", "built+tested", "needs a model that is already partly right", "selfplay.best_of_n"),
]
