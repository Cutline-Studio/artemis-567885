"""Model, KV-cache and GPU sizing for Artemis (pure Python, no torch).

Formulas follow Cisco Live BRKCOM-2115 "Building a full stack AI-ready data center" (2025):
  slide 11  raw model size        = parameters x bytes per parameter (FP32 4, FP16/BF16 2, INT8/FP8 1; INT4 0.5)
  slide 12  KV bytes per token    = 2 x precision x n_layers x d_model / GQA factor   (GQA factor = n_heads / n_kv_heads)
            inference GPU memory  = raw size + KV per token x context length          (x concurrent sessions, slide 14)
  slide 16  +20% overhead on inference memory; tensor / pipeline / hybrid parallelism
  slide 18  training memory       = FP16 weights + activations (~2.5 x weights) + Adam states in FP32 (params x 4 x 2)
                                  = 15 bytes per parameter, inside the deck's "raw size x 6-10" range

Every output is an ESTIMATE from these formulas, not a measurement. Units are decimal (1 GB = 1e9 bytes), as in the deck.
Known simplifications (kept on purpose so results match the deck's worked examples):
  * the slide-18 optimizer figure has no FP32 master weights or gradients; real mixed-precision Adam is ~16 B/param
    plus activations, so treat `training_gb` as a floor (see `training_gb_high`, the deck's 10x upper end);
  * activations really scale with batch x sequence, not with parameter count;
  * MoE models must keep ALL experts resident, so memory uses total parameters, not active ones.

CLI:  python tools/artemis_sizing.py                  markdown table for the Artemis stages below
      python tools/artemis_sizing.py --json             same rows as JSON
      python tools/artemis_sizing.py --params 8.03e9 --layers 32 --d-model 4096 --heads 32 --kv-heads 8 --context 131072
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

BYTES_PER_PARAM = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "fp8": 1.0, "int4": 0.5}
GPUS_GB = {"H100-80GB": 80.0, "H200-141GB": 141.0, "B200-192GB": 192.0}  # slide 14
INFERENCE_OVERHEAD = 1.20   # slide 16
GPUS_PER_NODE = 8           # one HGX node, NVLink/NVSwitch domain (slide 17)
HYBRID_PARAMS = 400e9       # slide 16: hybrid for "massive models like GPT-3 / Llama 400B"
GB = 1e9


def bytes_per_param(precision: str) -> float:
    try:
        return BYTES_PER_PARAM[precision.lower()]
    except KeyError:
        raise ValueError(f"unknown precision {precision!r}; choose one of {sorted(BYTES_PER_PARAM)}") from None


def raw_size_bytes(params: float, precision: str = "bf16") -> float:
    return params * bytes_per_param(precision)


def gqa_factor(n_heads: int, n_kv_heads: int | None) -> float:
    kv = n_kv_heads or n_heads
    if kv <= 0 or n_heads % kv:
        raise ValueError(f"n_heads ({n_heads}) must be a positive multiple of n_kv_heads ({kv})")
    return n_heads / kv


def kv_bytes_per_token(n_layers: int, d_model: int, n_heads: int, n_kv_heads: int | None = None,
                       kv_precision: str = "bf16") -> float:
    """Slide 12: 2 (K and V) x precision x layers x d_model / GQA factor."""
    return 2 * bytes_per_param(kv_precision) * n_layers * d_model / gqa_factor(n_heads, n_kv_heads)


def inference_bytes(params: float, n_layers: int, d_model: int, n_heads: int, n_kv_heads: int | None, context: int,
                    users: int = 1, precision: str = "bf16", kv_precision: str | None = None,
                    overhead: float = INFERENCE_OVERHEAD) -> float:
    kv = kv_bytes_per_token(n_layers, d_model, n_heads, n_kv_heads, kv_precision or precision)
    return (raw_size_bytes(params, precision) + kv * context * users) * overhead


def training_bytes(params: float, activation_mult: float = 2.5) -> float:
    """Slide 18 breakdown: FP16 weights (2 B) + activations (activation_mult x weights) + Adam FP32 (4 B x 2)."""
    weights = raw_size_bytes(params, "fp16")
    return weights + activation_mult * weights + params * 4 * 2


def gpus_needed(total_bytes: float, gpu_gb: float) -> int:
    return max(1, math.ceil(total_bytes / (gpu_gb * GB) - 1e-9))


def max_concurrent_sessions(params: float, n_layers: int, d_model: int, n_heads: int, n_kv_heads: int | None,
                            context: int, gpu_gb: float, n_gpus: int = 1, precision: str = "bf16",
                            kv_precision: str | None = None, overhead: float = INFERENCE_OVERHEAD) -> int:
    """How many full-context sessions fit next to one copy of the weights (slide 14: each session has its own KV cache)."""
    free = n_gpus * gpu_gb * GB / overhead - raw_size_bytes(params, precision)
    per = kv_bytes_per_token(n_layers, d_model, n_heads, n_kv_heads, kv_precision or precision) * context
    return max(0, int(free // per))


def parallelism(params: float, n_gpus: int, moe: bool = False) -> str:
    """Slide 16 rules: no split if one GPU holds it, tensor inside one NVLink node, pipeline across nodes, hybrid at 400B+."""
    if n_gpus <= 1:
        out = "none (replicas / data parallel)"
    elif params >= HYBRID_PARAMS:
        out = "hybrid (tensor in node + pipeline across nodes)"
    elif n_gpus <= GPUS_PER_NODE:
        out = "tensor (one NVLink node)"
    else:
        out = "pipeline across nodes (+tensor in node)"
    return out + (" + expert parallel" if moe and n_gpus > 1 else "")


@dataclass(frozen=True)
class Shape:
    name: str
    params: float          # total parameters (all experts for MoE)
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    context: int
    active_params: float | None = None
    source: str = ""

    @property
    def moe(self) -> bool:
        return self.active_params is not None and self.active_params < self.params


def size(shape: Shape, users: int = 1) -> dict:
    s = shape
    kv = kv_bytes_per_token(s.n_layers, s.d_model, s.n_heads, s.n_kv_heads)
    kv_mha = kv_bytes_per_token(s.n_layers, s.d_model, s.n_heads, s.n_heads)
    inf = inference_bytes(s.params, s.n_layers, s.d_model, s.n_heads, s.n_kv_heads, s.context, users)
    train = training_bytes(s.params)
    row = {
        "name": s.name, "params_b": s.params / 1e9, "active_b": (s.active_params or s.params) / 1e9,
        "gqa": gqa_factor(s.n_heads, s.n_kv_heads), "context": s.context, "users": users,
        "weights_gb": {p: raw_size_bytes(s.params, p) / GB for p in ("bf16", "fp8", "int4")},
        "kv_kb_per_token": kv / 1e3, "kv_kb_per_token_no_gqa": kv_mha / 1e3,
        "kv_gb_at_context": kv * s.context * users / GB,
        "inference_gb_bf16": inf / GB,
        "inference_gpus_bf16": {g: gpus_needed(inf, gb) for g, gb in GPUS_GB.items()},
        "training_gb": train / GB, "training_gb_high": 10 * raw_size_bytes(s.params, "bf16") / GB,
        "training_gpus": {g: gpus_needed(train, gb) for g, gb in GPUS_GB.items()},
        "source": s.source,
    }
    row["parallelism_inference"] = parallelism(s.params, row["inference_gpus_bf16"]["H100-80GB"], s.moe)
    row["parallelism_training"] = parallelism(s.params, row["training_gpus"]["H100-80GB"], s.moe)
    return row


# Stages. Artemis rows use Cutline-Industries gateway/configs/models.yaml (param counts from ModelConfig.param_count, quoted there).
# Llama / Qwen rows use the published configs. The 100B / 1T / 6T rows are the planning page's sizes; their
# layer/width/head shapes are ASSUMPTIONS made here only to size the KV cache (no such model exists).
STAGES = (
    Shape("artemis 100m (models.yaml)", 153e6, 12, 768, 12, 4, 2048, 111e6, "configs/models.yaml"),
    Shape("artemis 1b (models.yaml)", 1.31e9, 20, 2048, 16, 8, 4096, 1.06e9, "configs/models.yaml"),
    Shape("artemis 3b (models.yaml)", 4.17e9, 28, 3072, 24, 8, 4096, 3.38e9, "configs/models.yaml"),
    Shape("artemis 8b (models.yaml)", 10.2e9, 36, 4096, 32, 8, 8192, 8.4e9, "configs/models.yaml"),
    Shape("Qwen2.5-7B-Instruct (teacher/base)", 7.62e9, 28, 3584, 28, 4, 32768, None, "Qwen2.5 config.json"),
    Shape("Llama 3.1 8B (teacher)", 8.03e9, 32, 4096, 32, 8, 131072, None, "BRKCOM-2115 slides 11-12, 18"),
    Shape("Llama 3.1 405B (deck example)", 405e9, 126, 16384, 128, 8, 131072, None, "BRKCOM-2115 slide 16"),
    Shape("100B MoE plan (shape assumed)", 100e9, 48, 5120, 40, 8, 131072, 8e9, "planning page, Oct 27"),
    Shape("1T MoE plan (Kimi-K2-like dims, GQA assumed)", 1e12, 61, 7168, 64, 8, 131072, 32e9, "planning page, Oct 27"),
    Shape("6T MoE option (shape assumed)", 6e12, 96, 12288, 96, 8, 131072, 180e9, "planning page, Oct 27"),
)


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else (f"{x:,.1f}" if x >= 10 else f"{x:,.3g}")


def markdown(rows: list[dict]) -> str:
    g = list(GPUS_GB)
    head = ("| model | params (B) | BF16 / FP8 / INT4 weights (GB) | KV KB/token GQA (no GQA) | ctx | "
            "inference GB BF16 +20% | inference GPUs " + " / ".join(g) + " | training GB (deck) | training GPUs "
            + " / ".join(g) + " | parallelism inference ; training |")
    lines = [head, "|" + "---|" * 10]
    for r in rows:
        w = r["weights_gb"]
        lines.append(
            f"| {r['name']} | {_fmt(r['params_b'])}" + (f" ({_fmt(r['active_b'])} active)" if r["active_b"] != r["params_b"] else "")
            + f" | {_fmt(w['bf16'])} / {_fmt(w['fp8'])} / {_fmt(w['int4'])}"
            + f" | {_fmt(r['kv_kb_per_token'])} ({_fmt(r['kv_kb_per_token_no_gqa'])}) GQA {r['gqa']:g}"
            + f" | {r['context']:,} | {_fmt(r['inference_gb_bf16'])}"
            + " | " + " / ".join(str(r["inference_gpus_bf16"][k]) for k in g)
            + f" | {_fmt(r['training_gb'])} | " + " / ".join(str(r["training_gpus"][k]) for k in g)
            + f" | {r['parallelism_inference']} ; {r['parallelism_training']} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Artemis model / KV-cache / GPU sizing (estimates, BRKCOM-2115 formulas)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--params", type=float)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--d-model", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--users", type=int, default=1)
    a = ap.parse_args(argv)
    if a.params:
        if not (a.layers and a.d_model and a.heads):
            ap.error("--params needs --layers, --d-model and --heads")
        rows = [size(Shape("custom", a.params, a.layers, a.d_model, a.heads, a.kv_heads or a.heads, a.context), a.users)]
    else:
        rows = [size(s, a.users) for s in STAGES]
    print(json.dumps(rows, indent=2) if a.json else markdown(rows))
    return rows


if __name__ == "__main__":
    main()
