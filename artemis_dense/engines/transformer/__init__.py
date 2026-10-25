"""Transformer engine: the single entry point to the ONE dense model (no MoE / router / experts).

Every other engine gets the model through here (build / load / save / generate / embed / score).
  build(size="small"|"tiny"|"medium"|DenseConfig, **overrides) -> ArtemisDense
  validate(cfg) -> list of problems ([] = ok)
  save(model, path) / load(path) -> ArtemisDense
  generate(model, prompt_bytes, max_new_tokens, temperature, top_k, top_p, seed, use_cache)
  embed(model, tokens) -> [B, d_model] mean-pooled final hidden states (used by RAG)
  logprob(model, tokens) -> [B] mean next-byte log-prob (used by self_train, heart)
  count_params(model) / flops_per_token(cfg) (forward ~2N + attention term) / train_flops(cfg, D) = 6ND
  attention_maps(model, tokens) -> per-layer [B, H, T, T]; activations(model, tokens) -> per-layer [B, T, d]
  model_card(model) -> dict
"""
import json, os
from typing import Optional, Union

import torch
import torch.nn.functional as F

from ...config import DenseConfig, SCALES, expected_param_count
from ...model import ArtemisDense, apply_rope


def validate(cfg: DenseConfig) -> list:
    p = []
    if cfg.vocab_size != 256:
        p.append("vocab_size != 256: the byte-level data pipeline assumes 256")
    if cfg.n_layers < 1 or cfg.d_model < 8:
        p.append("model too small")
    if cfg.n_layers > cfg.caps.get("num_layers_max", cfg.n_layers):
        p.append("n_layers above num_layers_max cap")
    if cfg.d_ff > cfg.caps.get("d_ff_max", cfg.d_ff):
        p.append("d_ff above d_ff_max cap")
    if cfg.context_len * cfg.rope_scaling > 65536:
        p.append("effective context above 65536")
    return p


def build(size: Union[str, DenseConfig] = "small", seed: Optional[int] = None, **overrides) -> ArtemisDense:
    cfg = DenseConfig.preset(size, **overrides) if isinstance(size, str) else size
    problems = validate(cfg)
    if problems:
        raise ValueError("; ".join(problems))
    if seed is not None:
        torch.manual_seed(seed)
    return ArtemisDense(cfg)


def save(model: ArtemisDense, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"cfg": model.cfg.to_dict(), "model": model.state_dict()}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load(path: str) -> ArtemisDense:
    ck = torch.load(path, weights_only=False)
    m = ArtemisDense(DenseConfig.from_dict(ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m


def count_params(model: ArtemisDense) -> int:
    return model.num_params()


def flops_per_token(cfg: DenseConfig, seq_len: Optional[int] = None) -> int:
    """Forward FLOPs per token ~ 2*N_nonembedding + 2*n_layers*seq_len*d_model (Kaplan et al. 2020)."""
    n = expected_param_count(cfg) - cfg.vocab_size * cfg.d_model
    T = seq_len or cfg.context_len
    return 2 * n + 2 * cfg.n_layers * T * cfg.d_model


def train_flops(n_params: int, n_tokens: int) -> float:
    return 6.0 * n_params * n_tokens


@torch.no_grad()
def generate(model: ArtemisDense, prompt: Union[bytes, torch.Tensor], max_new_tokens: int = 64,
             temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, seed: Optional[int] = None,
             use_cache: bool = True) -> bytes:
    model.eval()
    x = torch.tensor([list(prompt)]) if isinstance(prompt, (bytes, bytearray)) else prompt
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    cache = model.new_cache() if use_cache else None
    out, inp = x, x
    for _ in range(max_new_tokens):
        if out.shape[1] >= model.cfg.context_len:
            break
        logits, _ = model(inp if use_cache else out, cache=cache)
        last = logits[:, -1].float()
        if temperature <= 0:
            nxt = last.argmax(-1, keepdim=True)
        else:
            last = last / temperature
            if top_k:
                kth = torch.topk(last, min(top_k, last.size(-1))).values[:, -1:]
                last = last.masked_fill(last < kth, float("-inf"))
            if top_p < 1.0:
                srt, idx = torch.sort(last, descending=True)
                cum = F.softmax(srt, -1).cumsum(-1)
                drop = cum - F.softmax(srt, -1) > top_p
                srt = srt.masked_fill(drop, float("-inf"))
                last = torch.full_like(last, float("-inf")).scatter(-1, idx, srt)
            nxt = torch.multinomial(F.softmax(last, -1), 1, generator=g)
        out = torch.cat([out, nxt], 1)
        inp = nxt
    return bytes(out[0, x.shape[1]:].tolist())


@torch.no_grad()
def embed(model: ArtemisDense, tokens: torch.Tensor) -> torch.Tensor:
    model.eval()
    h = model.trunk(model.embed(tokens))
    return F.normalize(h.mean(1), dim=-1)


@torch.no_grad()
def logprob(model: ArtemisDense, tokens: torch.Tensor) -> torch.Tensor:
    return model.sequence_logprob(tokens)


@torch.no_grad()
def activations(model: ArtemisDense, tokens: torch.Tensor) -> list:
    acts, h = [], model.embed(tokens)
    for b in model.blocks:
        h = b(h, model.rope_cos, model.rope_sin)
        acts.append(h.clone())
    return acts


@torch.no_grad()
def attention_maps(model: ArtemisDense, tokens: torch.Tensor) -> list:
    """Recompute each layer's softmax attention explicitly (same weights as the fused kernel)."""
    maps, h = [], model.embed(tokens)
    T = tokens.shape[1]
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    for b in model.blocks:
        a = b.attn
        x = b.attn_norm(h)
        B = x.shape[0]
        q = a.wq(x).view(B, T, a.n_heads, a.hd).transpose(1, 2)
        k = a.wk(x).view(B, T, a.n_kv, a.hd).transpose(1, 2)
        q, k = apply_rope(q, model.rope_cos, model.rope_sin), apply_rope(k, model.rope_cos, model.rope_sin)
        k = k.repeat_interleave(a.n_heads // a.n_kv, 1)
        s = (q @ k.transpose(-1, -2)) / a.hd ** 0.5
        maps.append(F.softmax(s.masked_fill(~mask, float("-inf")), -1))
        h = b(h, model.rope_cos, model.rope_sin)
    return maps


def model_card(model: ArtemisDense) -> dict:
    c = model.cfg
    return {"name": "Artemis 567885 dense", "architecture": "decoder-only dense transformer (no MoE)",
            "n_layers": c.n_layers, "d_model": c.d_model, "n_heads": c.n_heads, "n_kv_heads": c.n_kv_heads,
            "d_ff": c.d_ff, "vocab": c.vocab_size, "context_len": c.context_len, "rope_scaling": c.rope_scaling,
            "params": model.num_params(), "flops_per_token_fwd": flops_per_token(c),
            "tool_head": c.tool_head, "value_head": c.value_head, "encoders": list(model.encoders.keys()),
            "trained": "no (no real training run has been done)"}
