"""Positional encodings for the continuous-vector MoE transformer (PLAN stage 3).

Without a positional signal, unmasked self-attention + per-token experts is
permutation-equivariant: reordering the sequence just reorders the output.
These modules add a position-dependent vector to every token so order matters.

  pos_encoding="none"        -> identity (original behaviour, no parameters)
  pos_encoding="sinusoidal"  -> fixed Vaswani et al. table, no parameters
  pos_encoding="learned"     -> nn.Embedding(max_len, d_model), init N(0, 0.02)
"""
import math
import torch
import torch.nn as nn

POS_ENCODINGS = ("none", "sinusoidal", "learned")


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """PE[p, 2i] = sin(p / 10000^(2i/d)), PE[p, 2i+1] = cos(p / 10000^(2i/d)). Works for odd d."""
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    i2 = torch.arange(0, d_model, 2, dtype=torch.float32)
    inv = torch.exp(-math.log(10000.0) * i2 / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv[: d_model // 2])
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.max_len = max_len
        # non-persistent: not saved in state_dict, so old checkpoints still load
        self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(-2)
        if L > self.max_len:
            raise ValueError(f"sequence length {L} exceeds max_len {self.max_len}")
        return x + self.pe[:L].to(x.dtype)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(-2)
        if L > self.max_len:
            raise ValueError(f"sequence length {L} exceeds max_len {self.max_len}")
        return x + self.emb.weight[:L].to(x.dtype)


def build_positional(kind: str, d_model: int, max_len: int) -> nn.Module:
    if kind not in POS_ENCODINGS:
        raise ValueError(f"pos_encoding must be one of {POS_ENCODINGS}, got {kind!r}")
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len)
    if kind == "learned":
        return LearnedPositionalEncoding(d_model, max_len)
    return nn.Identity()
