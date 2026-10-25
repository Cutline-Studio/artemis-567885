"""Stage 8: review #6 measurement — task-loss-only gradient norm reaching the gate,
renorm vs switch weighting, top_k 1 and 2 (same setup as tools/investigate_baseline.py #6,
fixed semantics, legacy=False)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from moe import CapacitySparseMoELayer

torch.set_num_threads(1)
for k in (2, 1):
    for gw in ("renorm", "switch"):
        torch.manual_seed(0)
        m = CapacitySparseMoELayer(16, 32, 4, k, 10.0, gate_weighting=gw); m.train()
        x = torch.randn(32, 16, requires_grad=True)
        torch.manual_seed(1)
        out, aux, _ = m(x); out.pow(2).mean().backward()
        g = m.gate.weight.grad
        print(f"top_k={k} gate_weighting={gw:6s}: task-only gate grad norm = {0.0 if g is None else g.norm().item():.6f}")
