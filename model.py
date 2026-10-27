"""Stacked MoE transformer with vertical (layers) and horizontal (experts) growth."""
from typing import List, Sequence, Union
import torch
import torch.nn as nn
from transformer import MoETransformerBlock


class DynamicMultiBlockTransformer(nn.Module):
    """N MoETransformerBlocks on continuous vectors (no tokenizer).

    forward(x) -> (y, aux_loss_sum, dropped_count_sum); y.shape == x.shape.
    Each block returns three values (x, aux, dropped); both aux and dropped are
    summed across layers. top_k / capacity_factor / legacy are passed to every block.
    `num_experts` may be an int (same for all layers) or a per-layer list.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_experts: Union[int, Sequence[int]],
                 num_layers: int = 1, top_k: int = 2, capacity_factor: float = 1.0, legacy: bool = False,
                 causal: bool = False, dispatch: str = "loop"):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if isinstance(num_experts, int):
            num_experts = [num_experts] * num_layers
        if len(num_experts) != num_layers:
            raise ValueError(f"num_experts list has {len(num_experts)} entries for {num_layers} layers")
        self.d_model, self.n_heads, self.d_ff = d_model, n_heads, d_ff
        self.top_k, self.legacy, self.causal, self.dispatch = top_k, legacy, causal, dispatch
        self._capacity_factor = capacity_factor
        self.blocks = nn.ModuleList([
            MoETransformerBlock(d_model, n_heads, d_ff, e, top_k, capacity_factor, legacy=legacy,
                                causal=causal, dispatch=dispatch)
            for e in num_experts])

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    @property
    def num_experts(self) -> List[int]:
        return [b.moe.num_experts for b in self.blocks]

    def set_capacity_factor(self, cf: float):
        self._capacity_factor = cf
        for b in self.blocks:
            b.moe.capacity_factor = cf

    def forward(self, x: torch.Tensor):
        aux_total = x.new_zeros(())
        dropped_total = 0
        for block in self.blocks:
            x, aux, dropped = block(x)
            aux_total = aux_total + aux
            dropped_total += int(dropped)
        return x, aux_total, dropped_total

    @torch.no_grad()
    def add_layer(self, num_experts: int) -> int:
        """Append a block initialised to the exact identity map.

        The attention output projection and every expert's output projection
        are zero-initialised, so the new block contributes 0 + 0 on top of its
        residual stream and the trained model's function is preserved. Returns
        the new layer index.
        """
        ref = next(self.parameters())
        blk = MoETransformerBlock(self.d_model, self.n_heads, self.d_ff, num_experts, self.top_k,
                                  self._capacity_factor, legacy=self.legacy, causal=self.causal,
                                  dispatch=self.dispatch).to(ref.device, ref.dtype)
        blk.attn.out_linear.weight.zero_(); blk.attn.out_linear.bias.zero_()
        for e in blk.moe.experts:
            e.w2.weight.zero_(); e.w2.bias.zero_()
        self.blocks.append(blk)
        return len(self.blocks) - 1

    def add_expert(self, layer: int = -1, source_idx: int = None, noise_std: float = 1e-2,
                   generator: torch.Generator = None, init: str = "copy") -> int:
        return self.blocks[layer].moe.add_expert(source_idx, noise_std, generator, init=init)
