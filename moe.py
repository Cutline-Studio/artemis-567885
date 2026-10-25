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

def expert_load_entropy(counts, normalize: bool = True) -> float:
    """Shannon entropy of an expert load histogram (counts per expert, any scale).

    Natural log; normalize=True divides by ln(E) so a perfectly uniform load is
    1.0 and all load on one expert is 0.0. Zero counts contribute 0 (0 log 0 = 0).
    E == 1 or an all-zero histogram returns 0.0.
    """
    c = torch.as_tensor(counts, dtype=torch.float64).flatten()
    if c.numel() == 0 or float(c.sum()) <= 0:
        return 0.0
    p = c / c.sum()
    nz = p > 0
    h = float(-(p[nz] * torch.log(p[nz])).sum())
    if normalize:
        return h / math.log(c.numel()) if c.numel() > 1 else 0.0
    return h


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

    dispatch="loop" (default) is the per-expert Python loop above.
    dispatch="vectorized" (repo-checks, check 3) computes the SAME routing,
    capacity ranking, drops and outputs with one sort and two batched matmuls
    over a padded [E, capacity, d] buffer. Equivalence to the loop (outputs,
    assignments, drops, gradients) is asserted in checks/test_check3_moe_dispatch.py.
    Only valid with legacy=False.

    capacity_factor=None means dropless routing (capacity = number of tokens).
    Capacity-limited routing ranks tokens across the whole flattened batch, so
    with a causal mask a future token can displace a past token (see
    checks/test_check5_causal.py); use capacity_factor=None for causal models,
    or capacity_mode="sequence".

    capacity_mode="global" (default) is the behaviour described above.
    capacity_mode="sequence" (stage 7) is a causal-safe capacity: x must be
    [B, T, d] (a 2-D x is one sequence); every sequence gets its own per-expert
    capacity ceil(T / E * capacity_factor * top_k), and assignments are served
    first-come-first-served in position order (position ascending, then slot),
    so whether the assignment at position t is kept depends only on the routing
    of positions <= t of the same sequence. With causal attention, future bytes
    and other sequences in the batch therefore cannot change past outputs.
    Cost of the causal ordering: late positions are the ones that get dropped
    when an expert overflows. Only valid with legacy=False; works with both
    dispatch paths (same keep mask).

    Stage 8 options (all off by default, so default and legacy outputs are unchanged):
      * gate_weighting="renorm" (default): the chosen experts are weighted by a
        softmax over the K chosen logits. With top_k=1 that weight is always
        1.0, so the task loss sends NO gradient to the gate (review #6).
        gate_weighting="switch": weight = full router probability
        softmax(logits)[e] of each chosen expert (Switch Transformer), not
        renormalised over the K picks; the gate gets task gradient at top_k=1.
      * router z-loss: every non-legacy forward stores
        self.last_z_loss = mean_tokens(logsumexp(clean_logits)^2) (ST-MoE),
        graph-attached; it only affects training when a caller adds
        coef * last_z_loss to its loss (ArtemisModel z_loss_coef / train_bytes
        --z-loss). The return signature is unchanged.
      * routing="expert_choice": each expert picks its top-C tokens by router
        probability (Zhou et al. 2022), C = ceil(n / E * CF * top_k) with CF =
        capacity_factor (None -> 1.0) and n = tokens per routing group (the whole
        flattened batch for capacity_mode="global", one sequence for "sequence").
        A token is weighted by softmax(logits)[e] for every expert that picked it
        and may get 0..E experts. NOT CAUSAL: whether position t is picked depends
        on the scores of every other token in its group, including future
        positions, so it leaks future bytes in a causal LM (results must be
        labelled leaky). dropped_count = tokens that no expert picked (tokens, not
        (token, slot) assignments); aux is still the token-choice formula (it is
        exactly 1.0, no gradient, because every expert gets exactly C tokens).
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2,
                 capacity_factor: float = 1.0, legacy: bool = False, dispatch: str = "loop",
                 capacity_mode: str = "global", gate_weighting: str = "renorm",
                 routing: str = "token_choice"):
        super().__init__()
        if gate_weighting not in ("renorm", "switch"):
            raise ValueError(f"gate_weighting must be 'renorm' or 'switch', got {gate_weighting!r}")
        if routing not in ("token_choice", "expert_choice"):
            raise ValueError(f"routing must be 'token_choice' or 'expert_choice', got {routing!r}")
        if legacy and (gate_weighting != "renorm" or routing != "token_choice"):
            raise ValueError("gate_weighting/routing options implement the fixed (legacy=False) semantics only")
        self.gate_weighting, self.routing = gate_weighting, routing
        self.last_z_loss = None
        if capacity_mode not in ("global", "sequence"):
            raise ValueError(f"capacity_mode must be 'global' or 'sequence', got {capacity_mode!r}")
        if capacity_mode == "sequence" and legacy:
            raise ValueError("capacity_mode='sequence' implements the fixed (legacy=False) semantics only")
        self.capacity_mode = capacity_mode
        if dispatch not in ("loop", "vectorized"):
            raise ValueError(f"dispatch must be 'loop' or 'vectorized', got {dispatch!r}")
        if dispatch == "vectorized" and legacy:
            raise ValueError("dispatch='vectorized' implements the fixed (legacy=False) semantics only")
        self.dispatch = dispatch
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
        if self.capacity_factor is None:  # dropless
            return int(num_tokens)
        return int(math.ceil((num_tokens / self.num_experts) * self.capacity_factor * self.top_k))

    def _sequence_keep(self, topk_indices: torch.Tensor, shape) -> torch.Tensor:
        """capacity_mode='sequence': keep mask [N, K] for flat assignments.

        Per sequence, assignments to each expert are ranked by (position, slot)
        and the first `capacity` are kept: an inclusive cumulative count over
        the position-major/slot-minor order, so position t only sees t' <= t.
        """
        K, E = self.top_k, self.num_experts
        if len(shape) == 3:
            b, t = shape[0], shape[1]
        elif len(shape) == 2:
            b, t = 1, shape[0]
        else:
            raise ValueError(f"capacity_mode='sequence' needs x of shape [B, T, d] or [T, d], got {tuple(shape)}")
        cap = self.expert_capacity(t)
        onehot = F.one_hot(topk_indices.reshape(b, t * K), num_classes=E)       # [B, T*K, E]
        rank = (onehot.cumsum(dim=1) * onehot).sum(dim=-1) - 1                  # 0-based rank in its expert
        return (rank < cap).reshape(b * t, K), cap

    def _gate_weights(self, logits: torch.Tensor, topk_logits: torch.Tensor, topk_indices: torch.Tensor):
        """[N, K] combine weights for the chosen experts (see gate_weighting)."""
        if getattr(self, "gate_weighting", "renorm") == "switch":
            return F.softmax(logits, dim=-1).gather(1, topk_indices)
        return F.softmax(topk_logits, dim=-1)

    @staticmethod
    def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
        """ST-MoE router z-loss: mean over tokens of logsumexp(logits)^2."""
        return torch.logsumexp(logits, dim=-1).pow(2).mean()

    def forward(self, x: torch.Tensor):
        if self.legacy:
            return self._forward_legacy(x)
        if getattr(self, "routing", "token_choice") == "expert_choice":
            return self._forward_expert_choice(x)
        if self.dispatch == "vectorized":
            return self._forward_vectorized(x)
        return self._forward_fixed(x)

    # ---- original implementation, kept verbatim for reproducibility ----
    def _forward_legacy(self, x: torch.Tensor):
        orig_shape = x.shape
        flat_tokens = x.view(-1, orig_shape[-1])
        num_tokens = flat_tokens.size(0)

        logits = self.gate(flat_tokens)
        self.last_z_loss = self.router_z_loss(logits)  # stage 8 diagnostic only; no RNG, outputs unchanged
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
        topk_softmax_weights = self._gate_weights(logits, topk_logits, topk_indices)
        self.last_z_loss = self.router_z_loss(clean_logits)

        # Aux load-balancing loss over all top-k assignments (uniform optimum = 1.0).
        gate_prob = F.softmax(clean_logits, dim=-1)
        assign = F.one_hot(topk_indices, num_classes=self.num_experts).float().sum(dim=1)  # [N, E]
        me = assign.mean(dim=0) / self.top_k
        pe = gate_prob.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(me * pe)

        if self.capacity_mode == "sequence":
            seq_keep, expert_capacity = self._sequence_keep(topk_indices, orig_shape)
        else:
            seq_keep, expert_capacity = None, self.expert_capacity(num_tokens)
        output_workspace = torch.zeros_like(flat_tokens)
        processed_slots = torch.zeros(num_tokens, self.top_k, dtype=torch.bool, device=x.device)
        dropped_count = 0
        # priority score = full router probability of the routing logits (detached);
        # the renormalised top-k weights tie at 1.0 when top_k == 1, so they can't rank.
        priority = F.softmax(logits.detach(), dim=-1)

        for expert_idx in range(self.num_experts):
            token_positions, k_positions = (topk_indices == expert_idx).nonzero(as_tuple=True)
            n = token_positions.numel()
            if n == 0:
                continue
            if seq_keep is not None:
                keep = seq_keep[token_positions, k_positions]
                dropped_count += int(n - int(keep.sum()))
                token_positions = token_positions[keep]
                k_positions = k_positions[keep]
            elif n > expert_capacity:
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
                           "dropped_tokens": dropped_tokens, "num_tokens": num_tokens,
                           "topk_indices": topk_indices.detach(), "kept_slots": processed_slots}
        return output_workspace.view(orig_shape), aux_loss, int(dropped_count)

    # ---- vectorized dispatch, same semantics as _forward_fixed ----
    def _forward_vectorized(self, x: torch.Tensor):
        orig_shape = x.shape
        flat_tokens = x.reshape(-1, orig_shape[-1])
        num_tokens, d = flat_tokens.shape
        E, K = self.num_experts, self.top_k

        clean_logits = self.gate(flat_tokens)
        logits = clean_logits
        if self.training:
            logits = clean_logits + torch.randn_like(clean_logits) * (1.0 / self.num_experts)
        topk_logits, topk_indices = torch.topk(logits, K, dim=-1)
        topk_softmax_weights = self._gate_weights(logits, topk_logits, topk_indices)
        self.last_z_loss = self.router_z_loss(clean_logits)

        gate_prob = F.softmax(clean_logits, dim=-1)
        assign = F.one_hot(topk_indices, num_classes=E).float().sum(dim=1)
        me = assign.mean(dim=0) / K
        pe = gate_prob.mean(dim=0)
        aux_loss = E * torch.sum(me * pe)

        if self.capacity_mode == "sequence":
            return self._forward_vectorized_sequence(flat_tokens, orig_shape, topk_indices,
                                                     topk_softmax_weights, aux_loss)
        cap = self.expert_capacity(num_tokens)
        priority = F.softmax(logits.detach(), dim=-1)
        # flat assignment list in the loop's nonzero() order: token-major, then slot
        tok = torch.arange(num_tokens, device=x.device).repeat_interleave(K)
        slot = torch.arange(K, device=x.device).repeat(num_tokens)
        exp_ = topk_indices.reshape(-1)
        prio = priority[tok, exp_]
        # rank inside each expert by (slot asc, priority desc, original order asc),
        # exactly the loop's two stable argsorts, then group by expert (stable).
        order = torch.argsort(-prio, stable=True)
        order = order[torch.argsort(slot[order], stable=True)]
        order = order[torch.argsort(exp_[order], stable=True)]
        e_sorted = exp_[order]
        counts = torch.bincount(e_sorted, minlength=E)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(order.numel(), device=x.device) - starts[e_sorted]
        keep = rank < cap
        dropped_count = int((~keep).sum().item())
        k_order, k_e, k_rank = order[keep], e_sorted[keep], rank[keep]
        k_tok, k_slot = tok[k_order], slot[k_order]

        buf_len = int(min(cap, int(counts.max().item()) if counts.numel() else 0))
        buf = flat_tokens.new_zeros(E, max(buf_len, 1), d)
        buf = buf.index_put((k_e, k_rank), flat_tokens[k_tok])
        W1 = torch.stack([e.w1.weight for e in self.experts])      # [E, d_ff, d]
        b1 = torch.stack([e.w1.bias for e in self.experts])        # [E, d_ff]
        W2 = torch.stack([e.w2.weight for e in self.experts])      # [E, d, d_ff]
        b2 = torch.stack([e.w2.bias for e in self.experts])        # [E, d]
        h = self.experts[0].act(torch.baddbmm(b1.unsqueeze(1), buf, W1.transpose(1, 2)))
        out = torch.baddbmm(b2.unsqueeze(1), h, W2.transpose(1, 2))  # [E, L, d]
        contrib = out[k_e, k_rank] * topk_softmax_weights[k_tok, k_slot].unsqueeze(-1)
        output_workspace = torch.zeros_like(flat_tokens).index_add(0, k_tok, contrib)

        processed_slots = torch.zeros(num_tokens, K, dtype=torch.bool, device=x.device)
        processed_slots[k_tok, k_slot] = True
        dropped_tokens = int((~processed_slots.any(dim=-1)).sum().item())
        self.last_stats = {"capacity": cap, "dropped_assignments": dropped_count,
                           "dropped_tokens": dropped_tokens, "num_tokens": num_tokens,
                           "topk_indices": topk_indices.detach(), "kept_slots": processed_slots}
        return output_workspace.view(orig_shape), aux_loss, dropped_count

    def _expert_batch(self, buf: torch.Tensor) -> torch.Tensor:
        W1 = torch.stack([e.w1.weight for e in self.experts])
        b1 = torch.stack([e.w1.bias for e in self.experts])
        W2 = torch.stack([e.w2.weight for e in self.experts])
        b2 = torch.stack([e.w2.bias for e in self.experts])
        h = self.experts[0].act(torch.baddbmm(b1.unsqueeze(1), buf, W1.transpose(1, 2)))
        return torch.baddbmm(b2.unsqueeze(1), h, W2.transpose(1, 2))

    def _forward_vectorized_sequence(self, flat_tokens, orig_shape, topk_indices, topk_softmax_weights, aux_loss):
        num_tokens, d = flat_tokens.shape
        E, K = self.num_experts, self.top_k
        keep2d, cap = self._sequence_keep(topk_indices, orig_shape)
        keep = keep2d.reshape(-1)
        tok = torch.arange(num_tokens, device=flat_tokens.device).repeat_interleave(K)
        slot = torch.arange(K, device=flat_tokens.device).repeat(num_tokens)
        exp_ = topk_indices.reshape(-1)
        dropped_count = int((~keep).sum().item())
        k_tok, k_slot, k_e = tok[keep], slot[keep], exp_[keep]
        order = torch.argsort(k_e, stable=True)
        k_tok, k_slot, k_e = k_tok[order], k_slot[order], k_e[order]
        counts = torch.bincount(k_e, minlength=E)
        starts = torch.cumsum(counts, 0) - counts
        k_rank = torch.arange(k_e.numel(), device=flat_tokens.device) - starts[k_e]
        buf_len = int(counts.max().item()) if k_e.numel() else 1
        buf = flat_tokens.new_zeros(E, max(buf_len, 1), d).index_put((k_e, k_rank), flat_tokens[k_tok])
        out = self._expert_batch(buf)
        contrib = out[k_e, k_rank] * topk_softmax_weights[k_tok, k_slot].unsqueeze(-1)
        output_workspace = torch.zeros_like(flat_tokens).index_add(0, k_tok, contrib)
        processed_slots = keep2d.clone()
        dropped_tokens = int((~processed_slots.any(dim=-1)).sum().item())
        self.last_stats = {"capacity": cap, "dropped_assignments": dropped_count,
                           "dropped_tokens": dropped_tokens, "num_tokens": num_tokens,
                           "topk_indices": topk_indices.detach(), "kept_slots": processed_slots}
        return output_workspace.view(orig_shape), aux_loss, dropped_count

    # ---- stage 8: expert-choice routing (NOT causal, see class docstring) ----
    def expert_choice_capacity(self, group_tokens: int) -> int:
        cf = 1.0 if self.capacity_factor is None else self.capacity_factor
        return max(1, min(int(group_tokens), int(math.ceil(group_tokens / self.num_experts * cf * self.top_k))))

    def _forward_expert_choice(self, x: torch.Tensor):
        orig_shape = x.shape
        flat_tokens = x.reshape(-1, orig_shape[-1])
        num_tokens, d = flat_tokens.shape
        E = self.num_experts
        if self.capacity_mode == "sequence":
            if x.dim() == 3:
                groups, n = orig_shape[0], orig_shape[1]
            elif x.dim() == 2:
                groups, n = 1, orig_shape[0]
            else:
                raise ValueError(f"capacity_mode='sequence' needs x of shape [B, T, d] or [T, d], got {tuple(orig_shape)}")
        else:
            groups, n = 1, num_tokens
        clean_logits = self.gate(flat_tokens)
        logits = clean_logits
        if self.training:
            logits = clean_logits + torch.randn_like(clean_logits) * (1.0 / E)
        probs = F.softmax(logits, dim=-1)                                     # [N, E]
        self.last_z_loss = self.router_z_loss(clean_logits)
        C = self.expert_choice_capacity(n)
        # each expert picks its top-C tokens inside each group (ties -> lower index, torch.topk)
        sel = torch.topk(probs.detach().reshape(groups, n, E), C, dim=1).indices  # [G, C, E]
        offs = (torch.arange(groups, device=x.device) * n).view(groups, 1, 1)
        tok = (sel + offs).permute(2, 0, 1).reshape(E, groups * C)               # [E, G*C] flat token ids
        w = probs[tok, torch.arange(E, device=x.device)[:, None]]                                       # [E, G*C] probs[tok, e]
        if self.dispatch == "vectorized":
            out = self._expert_batch(flat_tokens[tok])                           # [E, G*C, d]
            contrib = (out * w.unsqueeze(-1)).reshape(-1, d)
            output_workspace = torch.zeros_like(flat_tokens).index_add(0, tok.reshape(-1), contrib)
        else:
            output_workspace = torch.zeros_like(flat_tokens)
            for e in range(E):
                y = self.experts[e](flat_tokens[tok[e]]) * w[e].unsqueeze(-1)
                output_workspace = output_workspace.index_add(0, tok[e], y)
        # aux in the token-choice form, for comparability (== 1.0 here: every expert gets G*C tokens)
        n_per_expert = torch.full((E,), float(groups * C), device=x.device)
        me = n_per_expert / n_per_expert.sum()
        aux_loss = E * torch.sum(me * F.softmax(clean_logits, dim=-1).mean(dim=0))
        experts_per_token = torch.bincount(tok.reshape(-1), minlength=num_tokens)
        dropped_tokens = int((experts_per_token == 0).sum().item())
        topk_indices = torch.topk(logits.detach(), self.top_k, dim=-1).indices   # what token choice would pick
        self.last_stats = {"capacity": C, "dropped_assignments": dropped_tokens, "dropped_tokens": dropped_tokens,
                           "num_tokens": num_tokens, "assignments": num_tokens, "routing": "expert_choice",
                           "topk_indices": topk_indices,
                           "kept_slots": torch.ones_like(topk_indices, dtype=torch.bool),
                           "routed_load": n_per_expert.detach(), "kept_load": n_per_expert.detach(),
                           "expert_choice_tokens": tok.detach(), "experts_per_token": experts_per_token}
        return output_workspace.view(orig_shape), aux_loss, dropped_tokens

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
