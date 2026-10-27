"""Multiscale Fourier feature encoding (PLAN stage 9a).

Standalone module: nothing else in the repo uses it yet (stage 9c will feed the
continuous document position t through it). Existing modules are unchanged.

For an input x of shape (..., in_dim) and L frequency bands,

    gamma(x) = [ x (optional),
                 sin(2^k * pi * x_j)  for k < L, j < in_dim,
                 cos(2^k * pi * x_j)  for k < L, j < in_dim,
                 sin(2 pi * x @ B), cos(2 pi * x @ B)   (optional random Gaussian features) ]

Layout of the multiscale part: band-major, i.e. the sin block is
[sin(2^0 pi x_0..x_{d-1}), sin(2^1 pi x_0..), ...]; the cos block follows the same order.

Random Gaussian Fourier features (Tancik et al. 2020): B has shape (in_dim, m) with
entries ~ N(0, gaussian_scale^2), drawn once at construction from a torch.Generator
seeded with `seed` (so the same seed gives the same B), stored as a buffer (saved in
state_dict, not trained).

Output dim = (in_dim if include_input) + 2 * L * in_dim + 2 * m.
"""
import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 1, num_frequencies: int = 8, include_input: bool = False,
                 gaussian_features: int = 0, gaussian_scale: float = 1.0, seed: int = 0):
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if num_frequencies < 0 or gaussian_features < 0:
            raise ValueError("num_frequencies and gaussian_features must be >= 0")
        if num_frequencies == 0 and gaussian_features == 0 and not include_input:
            raise ValueError("encoding would be empty")
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.gaussian_features = gaussian_features
        self.gaussian_scale = gaussian_scale
        self.seed = seed
        if gaussian_features > 0:
            g = torch.Generator().manual_seed(seed)
            B = torch.randn(in_dim, gaussian_features, generator=g) * gaussian_scale
            self.register_buffer("B", B)
        else:
            self.B = None

    def frequencies(self, dtype=torch.float32, device=None) -> torch.Tensor:
        """2^k * pi for k = 0..L-1, built in float64 then cast to `dtype` (so float64 inputs
        get full-precision frequencies; nothing is stored, nothing is trained)."""
        f = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float64)) * math.pi
        return f.to(dtype=dtype, device=device)

    @property
    def out_dim(self) -> int:
        return (self.in_dim if self.include_input else 0) + 2 * self.num_frequencies * self.in_dim \
            + 2 * self.gaussian_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_dim:
            raise ValueError(f"expected last dim {self.in_dim}, got {x.size(-1)}")
        parts = [x] if self.include_input else []
        if self.num_frequencies > 0:
            # (..., L, in_dim) -> (..., L*in_dim), band-major
            proj = (x.unsqueeze(-2) * self.frequencies(x.dtype, x.device).unsqueeze(-1)).flatten(-2)
            parts += [torch.sin(proj), torch.cos(proj)]
        if self.B is not None:
            gp = 2.0 * math.pi * (x @ self.B.to(x.dtype))
            parts += [torch.sin(gp), torch.cos(gp)]
        return torch.cat(parts, dim=-1)

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, num_frequencies={self.num_frequencies}, "
                f"include_input={self.include_input}, gaussian_features={self.gaussian_features}, "
                f"gaussian_scale={self.gaussian_scale}, out_dim={self.out_dim}")
