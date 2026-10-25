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

class _BatchedExpertFFN(torch.autograd.Function):
    """Batched expert FFN over a padded [E, L, d] buffer for dispatch="vectorized".

    Forward: the same two baddbmm calls as before (each output row depends only
    on its own input row, so padding rows are inert and the rows equal the
    loop's nn.Linear rows). Backward: input/activation gradients are batched
    too (row-wise), but each expert's weight and bias gradients are reductions
    over that expert's tokens, so they are computed per expert over exactly the
    first n_e (real) rows, with the same mm/sum calls nn.Linear's autograd uses.
    The backward is built from differentiable ops (activations are rebuilt
    from the inputs when create_graph=True), so double backward still works.
    The caller lays rows out in the loop's processing order, so the float32
    reduction order (and therefore every bit of the gradient) matches
    dispatch="loop". A padded batched reduction would sum over L >= n_e rows in
    a different order/blocking and drift by ~eps * sum|terms| (measured ~3e-5
    at N=4096, d=128, d_ff=512 before this change).
    """

    @staticmethod
    def forward(ctx, buf, n_rows, W1, b1, W2, b2):
        pre = torch.baddbmm(b1.unsqueeze(1), buf, W1.transpose(1, 2))
        h = F.gelu(pre)
        out = torch.baddbmm(b2.unsqueeze(1), h, W2.transpose(1, 2))
        ctx.save_for_backward(buf, pre, h, W1, b1, W2)
        ctx.n_rows = n_rows
        return out

    @staticmethod
    def backward(ctx, g_out):
        buf, pre, h, W1, b1, W2 = ctx.saved_tensors
        if torch.is_grad_enabled():
            # create_graph=True (double backward): rebuild the activations from the
            # graph-carrying inputs so second-order terms through pre/h are kept.
            pre = torch.baddbmm(b1.unsqueeze(1), buf, W1.transpose(1, 2))
            h = F.gelu(pre)
        g_out = g_out.contiguous()
        g_h = torch.bmm(g_out, W2)
        g_pre = torch.ops.aten.gelu_backward(g_h, pre)
        g_buf = torch.bmm(g_pre, W1)
        gW1, gb1, gW2, gb2 = [], [], [], []
        for e, n in enumerate(ctx.n_rows):
            if n == 0:  # idle expert: zero grads, as the padded batched path gave
                gW1.append(W1.new_zeros(W1.shape[1:])); gb1.append(W1.new_zeros(W1.shape[1]))
                gW2.append(W2.new_zeros(W2.shape[1:])); gb2.append(W2.new_zeros(W2.shape[1]))
                continue
            go, gp = g_out[e, :n], g_pre[e, :n]
            gW2.append(go.t().mm(h[e, :n])); gb2.append(go.sum(0))
            gW1.append(gp.t().mm(buf[e, :n])); gb1.append(gp.sum(0))
        return g_buf, None, torch.stack(gW1), torch.stack(gb1), torch.stack(gW2), torch.stack(gb2)


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
    checks/test_check5_causal.py); use capacity_factor=None for causal models.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2,
                 capacity_factor: float = 1.0, legacy: bool = False, dispatch: str = "loop"):
        super().__init__()
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

    def forward(self, x: torch.Tensor):
        if self.legacy:
            return self._forward_legacy(x)
        if self.dispatch == "vectorized":
            return self._forward_vectorized(x)
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

        expert_capacity = self.expert_capacity(num_tokens)
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
        topk_softmax_weights = F.softmax(topk_logits, dim=-1)

        gate_prob = F.softmax(clean_logits, dim=-1)
        assign = F.one_hot(topk_indices, num_classes=E).float().sum(dim=1)
        me = assign.mean(dim=0) / K
        pe = gate_prob.mean(dim=0)
        aux_loss = E * torch.sum(me * pe)

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
        # buffer row = position in the loop's processing order for that expert:
        # nonzero() (token-major, slot) order when nothing overflows, the
        # priority order (= rank) when the expert overflows its capacity.
        f_order = torch.argsort(exp_, stable=True)
        f_pos = torch.empty_like(f_order)
        f_pos[f_order] = torch.arange(f_order.numel(), device=x.device) - starts[exp_[f_order]]
        k_row = torch.where(counts[k_e] > cap, k_rank, f_pos[k_order])
        n_rows = torch.clamp(counts, max=cap).tolist()

        buf_len = int(min(cap, int(counts.max().item()) if counts.numel() else 0))
        buf = flat_tokens.new_zeros(E, max(buf_len, 1), d)
        buf = buf.index_put((k_e, k_row), flat_tokens[k_tok])
        W1 = torch.stack([e.w1.weight for e in self.experts])      # [E, d_ff, d]
        b1 = torch.stack([e.w1.bias for e in self.experts])        # [E, d_ff]
        W2 = torch.stack([e.w2.weight for e in self.experts])      # [E, d, d_ff]
        b2 = torch.stack([e.w2.bias for e in self.experts])        # [E, d]
        out = _BatchedExpertFFN.apply(buf, n_rows, W1, b1, W2, b2)  # [E, L, d]
        contrib = out[k_e, k_row] * topk_softmax_weights[k_tok, k_slot].unsqueeze(-1)
        output_workspace = torch.zeros_like(flat_tokens).index_add(0, k_tok, contrib)

        processed_slots = torch.zeros(num_tokens, K, dtype=torch.bool, device=x.device)
        processed_slots[k_tok, k_slot] = True
        dropped_tokens = int((~processed_slots.any(dim=-1)).sum().item())
        self.last_stats = {"capacity": cap, "dropped_assignments": dropped_count,
                           "dropped_tokens": dropped_tokens, "num_tokens": num_tokens,
                           "topk_indices": topk_indices.detach(), "kept_slots": processed_slots}
        return output_workspace.view(orig_shape), aux_loss, dropped_count

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
