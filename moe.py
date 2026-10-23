import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))

class CapacitySparseMoELayer(nn.Module):
    """Top-K sparse MoE with a per-expert capacity ceiling.

    legacy=True reproduces the original Artemis 567885 behaviour bit-for-bit
    (see docs/REVIEW.md). legacy=False (default) applies the stage-1 fixes:
      * aux loss: router probs from the *clean* logits, expert load counted over
        all top-k slots (fraction of assignments), so the uniform optimum stays 1.0
      * capacity: assignments ranked by slot (all top-1 picks first) then by gate
        weight (descending) instead of by token order
      * dropped tokens get zero MoE output (the block residual carries them);
        no 'sum == 0' heuristic and no input copy that would be double counted
      * last_stats exposes dropped assignments AND fully dropped tokens separately
    The return signature (output, aux_loss, dropped_count) is unchanged;
    dropped_count is the number of dropped (token, slot) assignments in both modes.
    """

    CAPACITY_MODES = ("batch", "none", "causal")

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2,
                 capacity_factor: float = 1.0, legacy: bool = False, capacity_mode: str = "batch"):
        """capacity_mode (fixed path only):
          * "batch"  (default, unchanged): one capacity over the whole flattened batch, ranked by
            slot then router probability. Not causal: a later token can evict an earlier one.
          * "none":   no capacity limit, nothing is dropped.
          * "causal": per sequence, positions are admitted in order; the token at position t is
            accepted by expert e only if fewer than ceil((t+1)/E * capacity_factor * top_k)
            earlier tokens of the same sequence were accepted by e. Decisions at t depend only on
            positions <= t of the same sequence, so routing preserves causality in training and
            evaluation. Requires input shaped [batch, seq, d_model].
        """
        if capacity_mode not in self.CAPACITY_MODES:
            raise ValueError(f"capacity_mode must be one of {self.CAPACITY_MODES}, got {capacity_mode!r}")
        if legacy and capacity_mode != "batch":
            raise ValueError("legacy=True only supports capacity_mode='batch'")
        super().__init__()
        self.capacity_mode = capacity_mode
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.legacy = legacy
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.last_stats = {}

    def expert_capacity(self, num_tokens: int) -> int:
        return int(math.ceil((num_tokens / self.num_experts) * self.capacity_factor * self.top_k))

    def forward(self, x: torch.Tensor):
        if self.legacy:
            return self._forward_legacy(x)
        return self._forward_fixed(x)

    # ---- original implementation, kept verbatim for reproducibility ----
    def _forward_legacy(self, x: torch.Tensor):
        orig_shape = x.shape
        flat_tokens = x.view(-1, orig_shape[-1])
        num_tokens = flat_tokens.size(0)

        logits = self.gate(flat_tokens)
        if self.training:
            logits = logits + torch.randn_like(logits) * (1.0 / self.num_experts)

        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_softmax_weights = F.softmax(topk_weights, dim=-1)

        gate_prob = F.softmax(logits, dim=-1)
        expert_mask = F.one_hot(topk_indices[:, 0], num_classes=self.num_experts).float()
        me = torch.mean(expert_mask, dim=0)
        pe = torch.mean(gate_prob, dim=0)
        aux_loss = self.num_experts * torch.sum(me * pe)

        expert_capacity = int(math.ceil((num_tokens / self.num_experts) * self.capacity_factor * self.top_k))
        output_workspace = torch.zeros_like(flat_tokens)
        dropped_count = 0

        for expert_idx in range(self.num_experts):
            token_positions, k_positions = (topk_indices == expert_idx).nonzero(as_tuple=True)
            if token_positions.numel() == 0:
                continue
            if token_positions.numel() > expert_capacity:
                dropped_count += (token_positions.numel() - expert_capacity)
                token_positions = token_positions[:expert_capacity]
                k_positions = k_positions[:expert_capacity]

            if token_positions.numel() > 0:
                routed_inputs = flat_tokens[token_positions]
                expert_outputs = self.experts[expert_idx](routed_inputs)
                gating_weight = topk_softmax_weights[token_positions, k_positions].unsqueeze(-1)
                output_workspace[token_positions] += expert_outputs * gating_weight

        unprocessed_mask = (output_workspace.sum(dim=-1) == 0)
        output_workspace[unprocessed_mask] = flat_tokens[unprocessed_mask]
        self.last_stats = {"capacity": expert_capacity, "dropped_assignments": int(dropped_count),
                           "dropped_tokens": int(unprocessed_mask.sum().item()), "num_tokens": num_tokens}
        return output_workspace.view(orig_shape), aux_loss, dropped_count

    # ---- fixed implementation ----
    def _forward_fixed(self, x: torch.Tensor):
        orig_shape = x.shape
        flat_tokens = x.reshape(-1, orig_shape[-1])
        num_tokens = flat_tokens.size(0)

        clean_logits = self.gate(flat_tokens)
        logits = clean_logits
        if self.training:
            logits = clean_logits + torch.randn_like(clean_logits) * (1.0 / self.num_experts)

        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_softmax_weights = F.softmax(topk_logits, dim=-1)

        # Aux load-balancing loss over all top-k assignments (uniform optimum = 1.0).
        gate_prob = F.softmax(clean_logits, dim=-1)
        assign = F.one_hot(topk_indices, num_classes=self.num_experts).float().sum(dim=1)  # [N, E]
        me = assign.mean(dim=0) / self.top_k
        pe = gate_prob.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(me * pe)

        output_workspace = torch.zeros_like(flat_tokens)
        processed_slots = torch.zeros(num_tokens, self.top_k, dtype=torch.bool, device=x.device)
        if self.capacity_mode != "batch":
            return self._dispatch_masked(x, flat_tokens, topk_indices, topk_softmax_weights,
                                         aux_loss, output_workspace, processed_slots)

        expert_capacity = self.expert_capacity(num_tokens)
        dropped_count = 0
        # priority score = full router probability of the routing logits (detached);
        # the renormalised top-k weights tie at 1.0 when top_k == 1, so they can't rank.
        priority = F.softmax(logits.detach(), dim=-1)

        for expert_idx in range(self.num_experts):
            token_positions, k_positions = (topk_indices == expert_idx).nonzero(as_tuple=True)
            n = token_positions.numel()
            if n == 0:
                continue
            if n > expert_capacity:
                # priority: lower slot first (top-1 before top-2), then larger router probability
                w = priority[token_positions, expert_idx]
                order = torch.argsort(-w, stable=True)
                order = order[torch.argsort(k_positions[order], stable=True)]
                keep = order[:expert_capacity]
                dropped_count += n - expert_capacity
                token_positions = token_positions[keep]
                k_positions = k_positions[keep]
            if token_positions.numel() > 0:
                expert_outputs = self.experts[expert_idx](flat_tokens[token_positions])
                gating_weight = topk_softmax_weights[token_positions, k_positions].unsqueeze(-1)
                output_workspace = output_workspace.index_add(0, token_positions, expert_outputs * gating_weight)
                processed_slots[token_positions, k_positions] = True

        dropped_tokens = int((~processed_slots.any(dim=-1)).sum().item())
        self.last_stats = {"capacity": expert_capacity, "dropped_assignments": int(dropped_count),
                           "dropped_tokens": dropped_tokens, "num_tokens": num_tokens}
        return output_workspace.view(orig_shape), aux_loss, int(dropped_count)

    def _causal_slot_mask(self, x: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """[N, top_k] bool: which (token, slot) assignments are admitted under causal capacity."""
        if x.dim() != 3:
            raise ValueError("capacity_mode='causal' needs input shaped [batch, seq, d_model]")
        b, t, _ = x.shape
        idx = topk_indices.view(b, t, self.top_k)
        wants = F.one_hot(idx, self.num_experts).sum(dim=2).bool()        # [B, T, E]
        accepted = torch.zeros_like(wants)
        counts = torch.zeros(b, self.num_experts, dtype=torch.long, device=x.device)
        scale = self.capacity_factor * self.top_k / self.num_experts
        for pos in range(t):
            cap = int(math.ceil((pos + 1) * scale))
            ok = wants[:, pos] & (counts < cap)
            accepted[:, pos] = ok
            counts += ok.long()
        return torch.gather(accepted, 2, idx).view(b * t, self.top_k)

    def _dispatch_masked(self, x, flat_tokens, topk_indices, topk_weights, aux_loss,
                         output_workspace, processed_slots):
        n = flat_tokens.size(0)
        if self.capacity_mode == "none":
            keep = torch.ones(n, self.top_k, dtype=torch.bool, device=x.device)
        else:
            keep = self._causal_slot_mask(x, topk_indices)
        for expert_idx in range(self.num_experts):
            token_positions, k_positions = ((topk_indices == expert_idx) & keep).nonzero(as_tuple=True)
            if token_positions.numel() == 0:
                continue
            expert_outputs = self.experts[expert_idx](flat_tokens[token_positions])
            gating_weight = topk_weights[token_positions, k_positions].unsqueeze(-1)
            output_workspace = output_workspace.index_add(0, token_positions, expert_outputs * gating_weight)
            processed_slots[token_positions, k_positions] = True
        dropped_count = int((~keep).sum().item())
        dropped_tokens = int((~processed_slots.any(dim=-1)).sum().item())
        self.last_stats = {"capacity": None if self.capacity_mode == "none" else "causal-prefix",
                           "dropped_assignments": dropped_count, "dropped_tokens": dropped_tokens,
                           "num_tokens": n}
        return output_workspace.view(x.shape), aux_loss, dropped_count

    @torch.no_grad()
    def add_expert(self, source_idx: int = None, noise_std: float = 1e-2, generator: torch.Generator = None) -> int:
        """Grow the layer by one expert, preserving all trained weights.

        The new expert is a copy of `source_idx` (default: expert 0) plus Gaussian
        noise; the gate gains one row copied from the source row plus noise.
        Returns the index of the new expert.
        """
        if source_idx is None:
            source_idx = 0
        src = self.experts[source_idx]
        new = Expert(self.d_model, self.d_ff).to(src.w1.weight.device, src.w1.weight.dtype)
        new.load_state_dict(src.state_dict())
        for p in new.parameters():
            p.add_(torch.randn(p.shape, generator=generator, dtype=p.dtype) * noise_std)
        self.experts.append(new)

        old_w = self.gate.weight.data
        row = old_w[source_idx:source_idx + 1].clone()
        row.add_(torch.randn(row.shape, generator=generator, dtype=row.dtype) * noise_std)
        new_gate = nn.Linear(self.d_model, self.num_experts + 1, bias=False).to(old_w.device, old_w.dtype)
        new_gate.weight.data.copy_(torch.cat([old_w, row], dim=0))
        self.gate = new_gate
        self.num_experts += 1
        return self.num_experts - 1
