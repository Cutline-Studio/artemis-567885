"""Stacked MoE transformer with vertical (layers) and horizontal (experts) growth,
plus ArtemisModel, the byte-level LM built on it (stage 5)."""
from typing import List, Sequence, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import MoETransformerBlock
from positional import build_positional


class DynamicMultiBlockTransformer(nn.Module):
    """N MoETransformerBlocks on continuous vectors (no tokenizer).

    forward(x) -> (y, aux_loss_sum, dropped_count_sum); y.shape == x.shape.
    Each block returns three values (x, aux, dropped); both aux and dropped are
    summed across layers. top_k / capacity_factor / legacy are passed to every block.
    `num_experts` may be an int (same for all layers) or a per-layer list.
    pos_encoding in {"none", "sinusoidal", "learned"} is added to x once, before
    the first block (positional.py); "none" keeps the original behaviour.
    Stage 8: gate_weighting ("renorm" | "switch") and routing ("token_choice" |
    "expert_choice", not causal) are passed to every block (moe.py). After each
    forward, self.last_z_loss is the router z-loss summed over layers (graph-
    attached; only used if the caller adds it to its loss).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_experts: Union[int, Sequence[int]],
                 num_layers: int = 1, top_k: int = 2, capacity_factor: float = 1.0, legacy: bool = False,
                 causal: bool = False, dispatch: str = "loop", pos_encoding: str = "none",
                 max_len: int = 512, capacity_mode: str = "global", gate_weighting: str = "renorm",
                 routing: str = "token_choice"):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if isinstance(num_experts, int):
            num_experts = [num_experts] * num_layers
        if len(num_experts) != num_layers:
            raise ValueError(f"num_experts list has {len(num_experts)} entries for {num_layers} layers")
        self.d_model, self.n_heads, self.d_ff = d_model, n_heads, d_ff
        self.top_k, self.legacy, self.causal, self.dispatch = top_k, legacy, causal, dispatch
        self.capacity_mode = capacity_mode
        self.gate_weighting, self.routing = gate_weighting, routing
        self.last_z_loss = None
        self._capacity_factor = capacity_factor
        self.pos_encoding, self.max_len = pos_encoding, max_len
        # "none" is nn.Identity (no parameters): default behaviour and checkpoints unchanged
        self.pos = build_positional(pos_encoding, d_model, max_len)
        self.blocks = nn.ModuleList([
            MoETransformerBlock(d_model, n_heads, d_ff, e, top_k, capacity_factor, legacy=legacy,
                                causal=causal, dispatch=dispatch, capacity_mode=capacity_mode,
                                gate_weighting=gate_weighting, routing=routing)
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
        z_total = x.new_zeros(())
        x = self.pos(x)
        for block in self.blocks:
            x, aux, dropped = block(x)
            aux_total = aux_total + aux
            dropped_total += int(dropped)
            if block.moe.last_z_loss is not None:
                z_total = z_total + block.moe.last_z_loss
        self.last_z_loss = z_total
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
                                  dispatch=self.dispatch,
                                  capacity_mode=getattr(self, "capacity_mode", "global"),
                                  gate_weighting=getattr(self, "gate_weighting", "renorm"),
                                  routing=getattr(self, "routing", "token_choice")).to(ref.device, ref.dtype)
        blk.attn.out_linear.weight.zero_(); blk.attn.out_linear.bias.zero_()
        for e in blk.moe.experts:
            e.w2.weight.zero_(); e.w2.bias.zero_()
        self.blocks.append(blk)
        return len(self.blocks) - 1

    def add_expert(self, layer: int = -1, source_idx: int = None, noise_std: float = 1e-2,
                   generator: torch.Generator = None) -> int:
        return self.blocks[layer].moe.add_expert(source_idx, noise_std, generator)


class ArtemisModel(nn.Module):
    """Byte-level MoE language model (PLAN stage 5).

    bytes [B, T] (LongTensor, values 0..255) -> nn.Embedding(256, d_model)
    -> DynamicMultiBlockTransformer (N stacked MoETransformerBlocks, the same
    causal / pos_encoding / dispatch / legacy options) -> LayerNorm -> Linear(d_model, 256).

    forward(ids) -> (logits [B, T, 256], aux_loss_sum, dropped_count_sum), with aux and
    dropped summed across layers exactly as in DynamicMultiBlockTransformer.

    Defaults are for next-byte prediction: causal=True and capacity_factor=None
    (dropless). Capacity-limited routing ranks tokens over the whole flattened
    batch, so a future byte can displace a past byte and leak information
    (checks/test_check5_causal.py); pass a float capacity_factor only when that is
    acceptable, or use capacity_mode="sequence" (stage 7: per-sequence capacity
    served in position order, causal-safe). legacy=True is forwarded to every block (original MoE semantics)
    and needs a float capacity_factor, since the original code has no dropless mode.

    Stage 8: gate_weighting="switch" weights each chosen expert by its full router
    probability (gate gets task gradient at top_k=1; default "renorm" unchanged);
    z_loss_coef (default 0.0) is stored on the model and `regularizer()` returns
    z_loss_coef * router z-loss summed over layers from the last forward (exactly
    0 and graph-free when z_loss_coef == 0, so default training is unchanged);
    routing="expert_choice" is NOT causal (an expert's top-C token pick looks at
    future positions), so it leaks future bytes and its results must be labelled
    leaky.
    """

    VOCAB = 256

    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128,
                 num_experts: Union[int, Sequence[int]] = 4, num_layers: int = 2, top_k: int = 2,
                 capacity_factor: float = None, legacy: bool = False, causal: bool = True,
                 dispatch: str = "loop", pos_encoding: str = "sinusoidal", max_len: int = 512,
                 capacity_mode: str = "global", gate_weighting: str = "renorm",
                 routing: str = "token_choice", z_loss_coef: float = 0.0):
        super().__init__()
        if legacy and capacity_factor is None:
            raise ValueError("legacy=True needs a float capacity_factor (original code has no dropless mode)")
        self.embed = nn.Embedding(self.VOCAB, d_model)
        self.body = DynamicMultiBlockTransformer(d_model, n_heads, d_ff, num_experts, num_layers, top_k,
                                                 capacity_factor, legacy=legacy, causal=causal,
                                                 dispatch=dispatch, pos_encoding=pos_encoding,
                                                 max_len=max_len, capacity_mode=capacity_mode,
                                                 gate_weighting=gate_weighting, routing=routing)
        self.z_loss_coef = float(z_loss_coef)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.VOCAB)

    @property
    def blocks(self):
        return self.body.blocks

    @property
    def num_layers(self) -> int:
        return self.body.num_layers

    @property
    def num_experts(self) -> List[int]:
        return self.body.num_experts

    def set_capacity_factor(self, cf: float):
        self.body.set_capacity_factor(cf)

    @property
    def causal_safe(self) -> bool:
        """False when routing can see future positions (expert choice, or global capacity with a finite CF)."""
        b = self.body
        if not b.causal or b.routing == "expert_choice":
            return False
        return b._capacity_factor is None or b.capacity_mode == "sequence"

    @property
    def last_z_loss(self):
        """Router z-loss summed over layers from the last forward (unweighted)."""
        return self.body.last_z_loss

    def regularizer(self) -> torch.Tensor:
        """z_loss_coef * last_z_loss; a plain 0.0 tensor (no graph) when z_loss_coef == 0."""
        if not self.z_loss_coef:
            return torch.zeros(())
        return self.z_loss_coef * self.body.last_z_loss

    def forward(self, ids: torch.Tensor):
        if ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
            raise TypeError(f"ids must be an integer tensor of byte values, got {ids.dtype}")
        if ids.dim() != 2:
            raise ValueError(f"ids must have shape [B, T], got {tuple(ids.shape)}")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) > self.VOCAB - 1):
            raise ValueError(f"byte ids must be in 0..{self.VOCAB - 1}, got min {int(ids.min())} max {int(ids.max())}")
        x = self.embed(ids.long())
        x, aux, dropped = self.body(x)
        return self.head(self.ln_f(x)), aux, dropped


def next_byte_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy (nats) of predicting ids[:, t+1] from logits[:, t]."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1).long())


def bits_per_byte(loss_nats) -> float:
    """Convert a mean cross-entropy in nats to bits per byte (loss / ln 2)."""
    return float(loss_nats) / math.log(2.0)
