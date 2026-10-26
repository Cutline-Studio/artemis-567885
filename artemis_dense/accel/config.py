"""AccelConfig: one flag per accelerated-learning method (Accelerate.md lists all 20).

The DenseConfig defaults and artemis_dense/train.py are untouched; this is a separate
training-config section consumed by artemis_dense/accel/trainer.py.
Presets: "baseline" (mirrors train.py stage 1), "fast" (every method runnable on CPU today),
"gpu_full" (adds the approval-needed methods; refuses to run until each approval flag is set).
"""
from dataclasses import dataclass, asdict, field
import json


@dataclass
class AccelConfig:
    # 1 single GPU ----------------------------------------------------------------- approval-needed
    device: str = "cpu"                 # "cpu" | "auto" | "cuda"; cuda only when gpu_approved
    gpu_approved: bool = False
    # 2 mixed precision
    precision: str = "fp32"             # "fp32" | "bf16" | "auto" (probe both, keep faster)
    # 3 fast attention
    attention: str = "default"          # "default" (PyTorch picks) | "flash" | "efficient" | "math" | "auto"
    # 4 torch.compile
    compile: bool = False
    # 5 multi-GPU DDP --------------------------------------------------------------- approval-needed
    ddp: bool = False
    ddp_approved: bool = False
    ddp_nproc: int = 2
    # 6 BPE tokenizer
    tokenizer: str = "byte"             # "byte" | "bpe"
    tokenizer_path: str = ""
    # 7 sequence packing
    pack: bool = False
    # 8 sequence-length warmup
    seq_warmup: bool = False
    seq_min: int = 64
    seq_warmup_frac: float = 0.3
    # 9 init from open pretrained weights ------------------------------------------ approval-needed
    init_pretrained: str = ""           # LOCAL directory only; no download code exists
    pretrained_approved: bool = False
    license_ack: str = ""               # must name the license key from pretrained.LICENSES
    # 10 distillation
    distill: bool = False
    teacher_path: str = ""              # local Artemis checkpoint (teacher logits)
    kd_alpha: float = 0.5
    kd_temp: float = 2.0
    teacher_generate: bool = False      # planned/approval-needed: external teacher text generation
    # 11 model growth
    grow: bool = False
    grow_start_layers: int = 2          # start shallower, add function-preserving layers
    grow_at: list = field(default_factory=lambda: [0.25, 0.5])   # time fractions for each add_layer
    grow_ffn_to: int = 0                # 0 = keep d_ff; else widen once at the last grow point
    # 12 LoRA
    lora: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_targets: list = field(default_factory=lambda: ["wq", "wv"])
    # 13 quality filter / 14 dedup
    quality_filter: bool = False
    dedup: bool = False
    minhash_threshold: float = 0.8
    # 15 synthetic verified problems
    synthetic: bool = False
    synthetic_math: int = 300
    synthetic_physics: int = 100
    # 16 curriculum
    curriculum: bool = False
    curriculum_start: float = 0.3       # fraction of (easiest-first) data available at t=0
    curriculum_full_at: float = 0.6     # time fraction where all data is available
    # 17 scaling-law sizing
    scaling_size: bool = False
    tokens_per_param: float = 20.0
    # 18 LR schedule + range finder
    lr: float = 1e-3
    warmup_frac: float = 0.0            # 0 -> use warmup_steps
    warmup_steps: int = 100
    lr_find: bool = False
    # 19 mistake-first replay
    mistake_replay: bool = False
    replay_frac: float = 0.25
    replay_buffer: int = 256
    mistakes_dir: str = ""              # optional engines/mistakes.py journal dir
    # 20 self-play best-of-N
    self_play: bool = False
    self_play_at: list = field(default_factory=lambda: [0.7])
    self_play_n: int = 4
    self_play_prompts: int = 8
    # guard against over-fitting a small corpus (not one of the 20; added after the first A/B)
    keep_best: bool = False             # early stopping on a dev split carved from TRAIN (val untouched)
    best_every_s: float = 15.0
    # model / loop
    size: str = "small"
    context_len: int = 512
    batch: int = 16
    seed: int = 567885

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        json.dump(self.to_dict(), open(path, "w"), indent=2)

    @classmethod
    def preset(cls, name: str, **over) -> "AccelConfig":
        return cls(**{**PRESETS[name], **over})


PRESETS = {
    # exactly what artemis_dense/train.py stage 1 does today
    "baseline": dict(size="small", context_len=512, batch=16, lr=1e-3, warmup_steps=100),
    # everything runnable on this CPU box with no approvals
    "fast": dict(precision="auto", attention="auto", compile=True, tokenizer="bpe", pack=True,
                 seq_warmup=True, grow=True, quality_filter=True, dedup=True, synthetic=True,
                 curriculum=True, scaling_size=True, lr_find=True, warmup_frac=0.05,
                 mistake_replay=True, self_play=True, distill=False, lora=False, keep_best=True,
                 context_len=256, batch=16),
    # adds the approval-needed methods; the trainer refuses until each *_approved flag is True
    "gpu_full": dict(device="cuda", precision="bf16", attention="flash", compile=True, ddp=False,
                     tokenizer="bpe", pack=True, seq_warmup=True, distill=True, quality_filter=True,
                     dedup=True, synthetic=True, curriculum=True, scaling_size=True, lr_find=True,
                     warmup_frac=0.02, mistake_replay=True, self_play=True, context_len=1024, batch=32),
}
