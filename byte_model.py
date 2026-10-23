"""Tokenization-free byte-level causal MoE language model (PLAN stages 5 and 9c).

bytes -> 256-entry byte embedding + position option -> DynamicMultiBlockTransformer
(causal attention, causal-safe MoE capacity) -> LayerNorm -> 256-way next-byte logits.

forward(idx) returns (logits, aux_loss, dropped_assignments). The aux load-balancing loss is
for training only; bits per byte must be computed from the cross-entropy alone.
"""
import torch
import torch.nn as nn

from model import DynamicMultiBlockTransformer
from positions import PositionEncoding


class ArtemisByteLM(nn.Module):
    def __init__(self, position: str, d_model: int = 32, n_heads: int = 2, d_ff: int = 64,
                 num_experts: int = 4, num_layers: int = 2, top_k: int = 2,
                 capacity_factor: float = 1.25, capacity_mode: str = "causal",
                 context: int = 128, max_len: int = None):
        super().__init__()
        if capacity_mode == "batch":
            # Batch-level capacity lets later (and other documents') tokens evict earlier ones,
            # which leaks future information into next-byte predictions.
            raise ValueError("ArtemisByteLM needs capacity_mode 'causal' or 'none', not 'batch'")
        self.context = context
        self.byte_embedding = nn.Embedding(256, d_model)
        self.position = PositionEncoding(position, d_model, max_len=max_len or context, ref_len=context)
        self.body = DynamicMultiBlockTransformer(
            d_model, n_heads, d_ff, num_experts, num_layers, top_k=top_k,
            capacity_factor=capacity_factor, causal=True, capacity_mode=capacity_mode)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)

    def forward(self, idx: torch.Tensor):
        x = self.byte_embedding(idx) + self.position(idx.shape[1], idx.device)
        h, aux, dropped = self.body(x)
        return self.head(self.norm(h)), aux, dropped
