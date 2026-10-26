"""Method 9: initialise Artemis from open pretrained weights (Llama / Qwen / Mistral family).

APPROVAL-NEEDED and NOT RUN. There is deliberately no download code: the loader only reads a
LOCAL directory (config.json + *.safetensors / *.bin) that a human has already fetched after
accepting the model's license. Using it means Artemis is no longer trained from scratch.

The license notes below are a checklist to verify on each model card before any download;
they are summaries, not legal advice, and licenses change (re-check the card each time).
"""
import json, os
from typing import Dict

import torch

from ..config import DenseConfig

LICENSES = {
    "llama3": {"family": "Meta Llama 3.x", "license": "Llama 3.x Community License (custom, not OSI)",
               "check": ["gated: accept on the model page", "700M monthly-active-user clause",
                         "Acceptable Use Policy applies", "'Built with Llama' attribution + name prefix for derivatives"]},
    "qwen2.5": {"family": "Qwen2.5", "license": "Apache-2.0 for most sizes; some sizes use the Qwen license",
                "check": ["read the LICENSE file of the exact size you pick", "Qwen license has a MAU threshold"]},
    "mistral": {"family": "Mistral 7B / Mixtral", "license": "Apache-2.0 for the open-weight base releases",
                "check": ["newer Mistral models may use research/non-production licenses (e.g. MNPL)",
                          "confirm per model card"]},
}


class ApprovalRequired(RuntimeError):
    pass


def check_approval(approved: bool, license_ack: str):
    if not approved:
        raise ApprovalRequired("pretrained init needs Latoya's OK (download + license) and breaks from-scratch purity")
    if license_ack not in LICENSES:
        raise ApprovalRequired(f"license_ack must be one of {sorted(LICENSES)} after reading the model card")


def config_from_hf(hf: dict, context_len: int = 2048) -> DenseConfig:
    import dataclasses
    d, h = hf["hidden_size"], hf["num_attention_heads"]
    extra = {"tokenizer": "bpe", "tokenizer_path": "<hf tokenizer>"} \
        if "tokenizer" in {f.name for f in dataclasses.fields(DenseConfig)} else {}
    return DenseConfig(vocab_size=hf["vocab_size"], **extra,
                       context_len=min(context_len, hf.get("max_position_embeddings", context_len)),
                       n_layers=hf["num_hidden_layers"], d_model=d, n_heads=h,
                       n_kv_heads=hf.get("num_key_value_heads", h), d_ff=hf["intermediate_size"],
                       rope_theta=float(hf.get("rope_theta", 10000.0)), norm_eps=hf.get("rms_norm_eps", 1e-6),
                       tie_embeddings=bool(hf.get("tie_word_embeddings", False)),
                       caps={"num_layers_max": 10 ** 6, "d_ff_max": 10 ** 9})


def _hf_to_interleaved(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """HF Llama stores q/k rows in rotate-half order; Artemis' RoPE rotates (even, odd) pairs."""
    out, inp = w.shape
    return w.view(n_heads, 2, out // n_heads // 2, inp).transpose(1, 2).reshape(out, inp)


def interleaved_to_hf(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    out, inp = w.shape
    return w.view(n_heads, out // n_heads // 2, 2, inp).transpose(1, 2).reshape(out, inp)


def map_llama_state(sd: Dict[str, torch.Tensor], cfg: DenseConfig) -> Dict[str, torch.Tensor]:
    """HF Llama/Qwen2/Mistral key names -> ArtemisDense state_dict (biases, e.g. Qwen qkv, unsupported)."""
    m = {"embed.weight": sd["model.embed_tokens.weight"], "norm.weight": sd["model.norm.weight"]}
    m["lm_head.weight"] = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        m[f"blocks.{i}.attn.wq.weight"] = _hf_to_interleaved(sd[p + "self_attn.q_proj.weight"], cfg.n_heads)
        m[f"blocks.{i}.attn.wk.weight"] = _hf_to_interleaved(sd[p + "self_attn.k_proj.weight"], cfg.n_kv_heads)
        m[f"blocks.{i}.attn.wv.weight"] = sd[p + "self_attn.v_proj.weight"]
        m[f"blocks.{i}.attn.wo.weight"] = sd[p + "self_attn.o_proj.weight"]
        m[f"blocks.{i}.ffn.w_gate.weight"] = sd[p + "mlp.gate_proj.weight"]
        m[f"blocks.{i}.ffn.w_up.weight"] = sd[p + "mlp.up_proj.weight"]
        m[f"blocks.{i}.ffn.w_down.weight"] = sd[p + "mlp.down_proj.weight"]
        m[f"blocks.{i}.attn_norm.weight"] = sd[p + "input_layernorm.weight"]
        m[f"blocks.{i}.ffn_norm.weight"] = sd[p + "post_attention_layernorm.weight"]
    return m


def load_local(path: str, approved: bool = False, license_ack: str = "", context_len: int = 2048):
    """Build an ArtemisDense from an already-downloaded local HF directory. No network."""
    check_approval(approved, license_ack)
    from ..model import ArtemisDense
    hf = json.load(open(os.path.join(path, "config.json")))
    cfg = config_from_hf(hf, context_len)
    sd = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd.update(load_file(os.path.join(path, f)))
        elif f.endswith(".bin"):
            sd.update(torch.load(os.path.join(path, f), map_location="cpu", weights_only=True))
    model = ArtemisDense(cfg)
    model.load_state_dict(map_llama_state(sd, cfg), strict=False)
    model.trained_info = {"init": "pretrained", "source": os.path.basename(path.rstrip("/")),
                          "license": license_ack, "from_scratch": False}
    return model
