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
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2, capacity_factor: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
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
        return output_workspace.view(orig_shape), aux_loss, dropped_count
