"""Positional encoding options for the byte-level language experiment.

All options map integer positions 0..T-1 to vectors of size d_model, added to byte embeddings.

  none        zeros (no position information; attention is still causal, which leaks some order)
  learned     one trainable vector per position; undefined beyond max_len (raises)
  sinusoidal  fixed Transformer encoding of the integer position (Vaswani et al. frequencies),
              F features, then a trainable Linear(F, d_model)
  fourier     multiscale Fourier features of the continuous coordinate t = position / ref_len:
              gamma(t) = [sin(2^k pi t), cos(2^k pi t)] for k < F/2 (stage 9a),
              then the same trainable Linear(F, d_model)

`sinusoidal` and `fourier` have identical feature width F and identical projection, so their
parameter counts match exactly and only the feature map differs. That pair isolates the Fourier
features' contribution; `learned` and `none` are reference arms.

Neither feature map guarantees useful extrapolation past the training length; evaluating beyond
it needs its own protocol (see docs/LANGUAGE_BENCHMARK.md).
"""
import math

import torch
import torch.nn as nn

KINDS = ("none", "learned", "sinusoidal", "fourier")


def sinusoidal_features(positions: torch.Tensor, num_features: int) -> torch.Tensor:
    if num_features % 2:
        raise ValueError("num_features must be even")
    half = num_features // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                                         device=positions.device) / half)
    ang = positions.float().unsqueeze(-1) * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def fourier_features(t: torch.Tensor, num_features: int) -> torch.Tensor:
    if num_features % 2:
        raise ValueError("num_features must be even")
    k = torch.arange(num_features // 2, dtype=torch.float32, device=t.device)
    ang = t.float().unsqueeze(-1) * (2.0 ** k) * math.pi
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def default_num_features(ref_len: int) -> int:
    """2 * (log2(ref_len) + 1): the octave count at which 2^k pi t still separates adjacent
    positions on t = n / ref_len; higher octaves alias to a constant."""
    return 2 * (int(math.log2(ref_len)) + 1)


class PositionEncoding(nn.Module):
    def __init__(self, kind: str, d_model: int, max_len: int, ref_len: int = None,
                 num_features: int = None):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind, self.d_model, self.max_len = kind, d_model, max_len
        self.ref_len = ref_len or max_len
        self.num_features = num_features or default_num_features(self.ref_len)
        if kind == "learned":
            self.table = nn.Embedding(max_len, d_model)
        elif kind in ("sinusoidal", "fourier"):
            self.proj = nn.Linear(self.num_features, d_model)

    def features(self, positions: torch.Tensor) -> torch.Tensor:
        if self.kind == "sinusoidal":
            return sinusoidal_features(positions, self.num_features)
        return fourier_features(positions.float() / self.ref_len, self.num_features)

    def forward(self, seq_len: int, device=None) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        if self.kind == "none":
            return torch.zeros(seq_len, self.d_model, device=device)
        if self.kind == "learned":
            if seq_len > self.max_len:
                raise ValueError(f"learned positions are defined only up to {self.max_len}, got {seq_len}")
            return self.table(positions)
        return self.proj(self.features(positions))
