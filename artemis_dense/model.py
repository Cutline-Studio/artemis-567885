"""Artemis 567885 dense: one decoder-only dense transformer, no MoE / router / experts.

Pre-norm RMSNorm -> causal grouped-query attention with RoPE (+ linear RoPE scaling) -> residual
            RMSNorm -> SwiGLU dense FFN                                           -> residual
Tied byte embeddings / LM head. Optional heads: tool head (which engine to call) and value head
(self-score). Optional multimodal prefix encoders (see multimodal.py) project into the same stream.

Interfaces
  forward(tokens[B,T], targets=None, cache=None, prefix=None) -> (logits[B,T,V], loss|None)
  forward_vectors(x[B,T,d_model]) -> (y, aux=0., dropped=0)   # old MoE model's 3-tuple
  generate(tokens, max_new_tokens, use_cache=True)            # greedy / sampled, with KV cache
  sequence_logprob(tokens) / score(tokens)                     # self-scoring for self_train
  tool_logits(tokens) / tool_call(tokens)                      # tool head -> engines.tools router
Growth (function-preserving): add_layer(), widen_ffn(new_d_ff)
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DenseConfig, TOOLS


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(T: int, head_dim: int, theta: float, scaling: float = 1.0, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(T, device=device).float() / scaling          # linear position interpolation
    ang = torch.outer(pos, inv)                                     # [T, hd/2]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin, start: int = 0):
    # x: [B, H, T, hd]; rotate (even, odd) pairs at absolute positions start..start+T-1
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[start:start + T].to(x.dtype), sin[start:start + T].to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class KVCache:
    """Per-layer key/value tensors for incremental decoding (keys stored after RoPE, before GQA repeat)."""

    def __init__(self, n_layers: int):
        self.k: List[Optional[torch.Tensor]] = [None] * n_layers
        self.v: List[Optional[torch.Tensor]] = [None] * n_layers

    @property
    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, i, k, v):
        if self.k[i] is None:
            self.k[i], self.v[i] = k, v
        else:
            self.k[i] = torch.cat([self.k[i], k], 2)
            self.v[i] = torch.cat([self.v[i], v], 2)
        return self.k[i], self.v[i]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.hd, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, start: int = 0, cache: Optional[KVCache] = None, layer: int = 0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin, start), apply_rope(k, cos, sin, start)
        if cache is not None:
            k, v = cache.update(layer, k, v)
        if self.n_kv != self.n_heads:                       # GQA: each kv head serves n_heads/n_kv q heads
            rep = self.n_heads // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        S = k.shape[2]
        drop = self.dropout if self.training else 0.0
        if S == T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:   # cached decode: query i sits at absolute position S-T+i and may see keys <= that
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    """One dense FFN: w_down( silu(w_gate x) * w_up x ). Every token uses every weight."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DenseBlock(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin, start=0, cache=None, layer=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, start, cache, layer)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ArtemisDense(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DenseBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.tool_head = nn.Linear(cfg.d_model, len(TOOLS), bias=False) if cfg.tool_head else None
        self.value_head = nn.Linear(cfg.d_model, 1) if cfg.value_head else None
        self.encoders = nn.ModuleDict()          # pluggable multimodal encoders (not counted in base params)
        cos, sin = rope_cache(cfg.context_len, cfg.head_dim, cfg.rope_theta, cfg.rope_scaling)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:   # GPT-2 style scaled residual projections
            for w in (b.attn.wo.weight, b.ffn.w_down.weight):
                nn.init.normal_(w, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))
        if self.value_head is not None:
            nn.init.zeros_(self.value_head.bias)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    # ---- bookkeeping -------------------------------------------------------
    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    def num_params(self, include_encoders: bool = False) -> int:
        enc = set(id(p) for p in self.encoders.parameters())
        return sum(p.numel() for p in self.parameters() if include_encoders or id(p) not in enc)

    def new_cache(self) -> KVCache:
        return KVCache(self.num_layers)

    # ---- forward -----------------------------------------------------------
    def trunk(self, h, cache: Optional[KVCache] = None):
        start = cache.length if cache is not None else 0
        if start + h.shape[1] > self.cfg.context_len:
            raise ValueError(f"sequence length {start + h.shape[1]} > context_len {self.cfg.context_len}")
        for i, b in enumerate(self.blocks):
            h = b(h, self.rope_cos, self.rope_sin, start, cache, i)
        return self.norm(h)

    def embed_inputs(self, tokens: torch.Tensor, prefix: Optional[torch.Tensor] = None):
        h = self.embed(tokens)
        return h if prefix is None else torch.cat([prefix.to(h.dtype), h], 1)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None,
                cache: Optional[KVCache] = None, prefix: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.trunk(self.embed_inputs(tokens, prefix), cache)
        if prefix is not None:
            h = h[:, prefix.shape[1]:]                 # logits only for the text positions
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def forward_vectors(self, x: torch.Tensor):
        """Continuous-vector path matching DynamicMultiBlockTransformer.forward's signature.
        Dense => no load-balancing aux loss and no dropped tokens."""
        return self.trunk(x), x.new_zeros(()), 0

    def forward_multimodal(self, tokens, targets=None, **inputs):
        """inputs: name=tensor for each attached encoder, e.g. image=[B,C,H,W], audio=[B,n_mels,frames]."""
        parts = [self.encoders[name](x) for name, x in inputs.items() if x is not None]
        prefix = torch.cat(parts, 1) if parts else None
        return self.forward(tokens, targets, prefix=prefix)

    def attach_encoder(self, name: str, module: nn.Module):
        if getattr(module, "d_model", self.cfg.d_model) != self.cfg.d_model:
            raise ValueError("encoder d_model must match the model")
        self.encoders[name] = module.to(self.embed.weight.device)

    # ---- generation with KV cache ------------------------------------------
    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
                 use_cache: bool = True, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        out = tokens
        cache = self.new_cache() if use_cache else None
        nxt_in = tokens
        for _ in range(max_new_tokens):
            logits, _ = self(nxt_in if use_cache else out, cache=cache)
            last = logits[:, -1]
            if temperature <= 0:
                nxt = last.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(last / temperature, -1), 1, generator=generator)
            out = torch.cat([out, nxt], 1)
            nxt_in = nxt
        return out

    # ---- self-scoring --------------------------------------------------------
    @torch.no_grad()
    def sequence_logprob(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mean next-byte log-prob per sequence [B]. Higher = the model finds it easier/more likely."""
        logits, _ = self(tokens[:, :-1])
        lp = F.log_softmax(logits, -1).gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        return lp.mean(-1)

    def score(self, tokens: torch.Tensor) -> torch.Tensor:
        """Value-head self-score per sequence [B] (from the last position). Requires value_head=True."""
        if self.value_head is None:
            raise RuntimeError("value_head is disabled in this config")
        return self.value_head(self.trunk(self.embed(tokens))[:, -1]).squeeze(-1)

    # ---- tool head -------------------------------------------------------------
    def tool_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.tool_head is None:
            raise RuntimeError("tool_head is disabled in this config")
        return self.tool_head(self.trunk(self.embed(tokens))[:, -1])

    @torch.no_grad()
    def tool_call(self, tokens: torch.Tensor) -> List[str]:
        return [TOOLS[i] for i in self.tool_logits(tokens).argmax(-1).tolist()]

    # ---- function-preserving growth ---------------------------------------
    @torch.no_grad()
    def add_layer(self) -> int:
        ref = self.embed.weight
        blk = DenseBlock(self.cfg).to(ref.device, ref.dtype)
        for m in blk.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.cfg.init_std)
        blk.attn.wo.weight.zero_()
        blk.ffn.w_down.weight.zero_()
        self.blocks.append(blk)
        self.cfg.n_layers = len(self.blocks)
        return len(self.blocks) - 1

    @torch.no_grad()
    def widen_ffn(self, new_d_ff: int, generator: Optional[torch.Generator] = None):
        old = self.cfg.d_ff
        if new_d_ff <= old:
            raise ValueError(f"new_d_ff {new_d_ff} must exceed current d_ff {old}")
        extra = new_d_ff - old
        for b in self.blocks:
            f = b.ffn
            dev, dt = f.w_gate.weight.device, f.w_gate.weight.dtype
            new = SwiGLU(self.cfg.d_model, new_d_ff).to(dev, dt)
            for name in ("w_gate", "w_up"):
                w_old = getattr(f, name).weight
                w_new = torch.randn(extra, w_old.shape[1], generator=generator, dtype=dt) * self.cfg.init_std
                getattr(new, name).weight.copy_(torch.cat([w_old, w_new.to(dev)], 0))
            new.w_down.weight.copy_(torch.cat([f.w_down.weight, f.w_down.weight.new_zeros(self.cfg.d_model, extra)], 1))
            b.ffn = new
        self.cfg.d_ff = new_d_ff
