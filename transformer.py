import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe import CapacitySparseMoELayer

class MultiHeadAttention(nn.Module):
    """causal=False (default) is the original, unmasked attention.
    causal=True masks keys at positions > query position (repo-checks, check 5)."""

    def __init__(self, d_model: int, n_heads: int, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, d_model = x.shape
        q = self.q_linear(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.causal:
            future = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(b, seq_len, d_model)
        return self.out_linear(context)

class MoETransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_experts: int, top_k: int = 2, capacity_factor: float = 1.0,
                 legacy: bool = False, causal: bool = False, dispatch: str = "loop"):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ln1 = nn.LayerNorm(d_model)
        self.moe = CapacitySparseMoELayer(d_model, d_ff, num_experts, top_k, capacity_factor, legacy=legacy,
                                          dispatch=dispatch)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        moe_out, aux_loss, dropped_count = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux_loss, dropped_count
