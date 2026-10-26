"""Method 12: LoRA adapters - fine-tune small low-rank add-ons instead of the whole model.

apply_lora(model, r, alpha, targets) freezes every base weight and wraps the named nn.Linear
layers (default attention wq/wv) with W x + (alpha/r) * B(A x), B initialised to zero so the
wrapped model starts identical. merge_lora() folds the adapters back into plain weights.
Best for fast skill updates of an already-trained model, not for from-scratch pretraining.
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def apply_lora(model, r: int = 8, alpha: float = 16.0, targets=("wq", "wv")) -> int:
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(mod, name, LoRALinear(child, r, alpha)); n += 1
    return n


def lora_parameters(model):
    return [p for m in model.modules() if isinstance(m, LoRALinear) for p in (m.A, m.B)]


@torch.no_grad()
def merge_lora(model) -> int:
    n = 0
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight += (child.B @ child.A) * child.scale
                child.base.weight.requires_grad_(True)
                setattr(mod, name, child.base); n += 1
    for p in model.parameters():
        p.requires_grad_(True)
    return n
