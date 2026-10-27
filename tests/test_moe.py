import math
import subprocess, sys, os
import pytest
import torch
import torch.nn.functional as F
from moe import CapacitySparseMoELayer
from transformer import MoETransformerBlock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def routed_to_expert0(n_tokens=4, d=4, scores=(0.1, 0.2, 5.0, 6.0), legacy=False, cf=0.5):
    m = CapacitySparseMoELayer(d, 8, 2, top_k=1, capacity_factor=cf, legacy=legacy); m.eval()
    with torch.no_grad():
        m.gate.weight.zero_(); m.gate.weight[0, 0] = 1.0
    x = torch.zeros(n_tokens, d); x[:, 0] = torch.tensor(scores)
    return m, x

@pytest.mark.parametrize("legacy", [True, False])
def test_shapes(legacy):
    torch.manual_seed(0)
    blk = MoETransformerBlock(16, 2, 32, 4, top_k=2, capacity_factor=1.0, legacy=legacy)
    x = torch.randn(3, 5, 16)
    y, aux, dropped = blk(x)
    assert y.shape == x.shape
    assert aux.dim() == 0
    assert isinstance(dropped, int)

@pytest.mark.parametrize("legacy", [True, False])
def test_gradients_reach_input_and_gate(legacy):
    torch.manual_seed(0)
    blk = MoETransformerBlock(16, 2, 32, 4, top_k=2, capacity_factor=1.0, legacy=legacy); blk.train()
    x = torch.randn(2, 8, 16, requires_grad=True)
    y, aux, _ = blk(x)
    (F.mse_loss(y, torch.zeros_like(y)) + 0.1 * aux).backward()
    assert x.grad is not None and x.grad.norm() > 0
    assert blk.moe.gate.weight.grad is not None and blk.moe.gate.weight.grad.norm() > 0

def test_task_loss_alone_reaches_gate_with_top2():
    torch.manual_seed(0)
    m = CapacitySparseMoELayer(16, 32, 4, top_k=2, capacity_factor=10.0); m.train()
    out, _, _ = m(torch.randn(32, 16))
    out.pow(2).mean().backward()
    assert m.gate.weight.grad.norm() > 0

@pytest.mark.parametrize("legacy", [True, False])
def test_capacity_drop_count(legacy):
    m, x = routed_to_expert0(legacy=legacy)
    cap = m.expert_capacity(4)
    assert cap == math.ceil(4 / 2 * 0.5 * 1) == 1
    _, _, dropped = m(x)
    assert dropped == 4 - cap
    assert m.last_stats["dropped_assignments"] == 3

def test_dropped_count_is_assignments_with_topk():
    torch.manual_seed(0)
    m = CapacitySparseMoELayer(4, 8, 2, top_k=2, capacity_factor=0.25); m.eval()
    _, _, dropped = m(torch.randn(8, 4))
    cap = m.expert_capacity(8)
    # every token picks both experts -> 8 assignments per expert
    assert dropped == 2 * (8 - cap)
    assert m.last_stats["dropped_tokens"] <= 8

def test_fixed_capacity_prefers_highest_gate_score():
    m, x = routed_to_expert0(legacy=False)
    out, _, _ = m(x)
    nonzero = out.abs().sum(-1) > 0
    assert nonzero.tolist() == [False, False, False, True]  # token with score 6.0 kept

def test_legacy_capacity_is_token_order():
    m, x = routed_to_expert0(legacy=True)
    out, _, _ = m(x)
    kept = (out - x).abs().sum(-1) > 0
    assert kept.tolist() == [True, False, False, False]

def test_fixed_dropped_tokens_get_zero_not_input_copy():
    m, x = routed_to_expert0(legacy=False)
    out, _, _ = m(x)
    assert torch.equal(out[:3], torch.zeros_like(out[:3]))
    assert m.last_stats["dropped_tokens"] == 3

def test_fixed_no_sum_zero_false_positive():
    torch.manual_seed(0)
    m = CapacitySparseMoELayer(4, 8, 2, top_k=1, capacity_factor=10.0); m.eval()
    with torch.no_grad():
        for e in m.experts:
            e.w2.weight.zero_(); e.w2.bias.copy_(torch.tensor([1., -1., 2., -2.]))
    x = torch.randn(3, 4)
    out, _, _ = m(x)
    assert torch.allclose(out, torch.tensor([1., -1., 2., -2.]).expand(3, 4))
    assert m.last_stats["dropped_tokens"] == 0

def test_fixed_aux_matches_formula_over_all_slots():
    torch.manual_seed(0)
    m = CapacitySparseMoELayer(16, 32, 4, top_k=2, capacity_factor=10.0); m.eval()
    x = torch.randn(64, 16)
    _, aux, _ = m(x)
    logits = m.gate(x)
    idx = logits.topk(2, dim=-1).indices
    me = F.one_hot(idx, 4).float().sum(1).mean(0) / 2
    pe = F.softmax(logits, -1).mean(0)
    assert torch.allclose(aux, 4 * (me * pe).sum())
    with torch.no_grad(): m.gate.weight.zero_()
    _, a_uni, _ = m(x)
    assert abs(a_uni.item() - 1.0) < 1e-6  # uniform router -> optimum 1.0

def test_fixed_aux_is_noise_free_when_routing_is_stable():
    m = CapacitySparseMoELayer(4, 8, 4, top_k=2, capacity_factor=10.0); m.train()
    with torch.no_grad():
        m.gate.weight.zero_(); m.gate.weight[0, 0] = 50.0; m.gate.weight[1, 0] = 40.0
    x = torch.zeros(16, 4); x[:, 0] = 1.0  # huge margins: noise (std 0.25) can't flip top-2
    torch.manual_seed(1); _, a1, _ = m(x)
    torch.manual_seed(2); _, a2, _ = m(x)
    assert a1.item() == a2.item()
    m.legacy = True
    torch.manual_seed(1); _, l1, _ = m(x)
    torch.manual_seed(2); _, l2, _ = m(x)
    assert l1.item() != l2.item()  # legacy uses noisy probabilities

@pytest.mark.parametrize("legacy", [True, False])
def test_determinism_with_seed(legacy):
    def run():
        torch.manual_seed(123)
        blk = MoETransformerBlock(16, 2, 32, 4, top_k=2, capacity_factor=0.8, legacy=legacy); blk.train()
        x = torch.randn(2, 8, 16, requires_grad=True)
        y, aux, d = blk(x)
        (y.pow(2).mean() + 0.1 * aux).backward()
        return y.detach(), aux.item(), d, x.grad.clone()
    a, b = run(), run()
    assert torch.equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2] and torch.equal(a[3], b[3])

def test_legacy_sandbox_reproduces_baseline_numbers():
    out = subprocess.run([sys.executable, "sandbox.py", "--mode", "legacy"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert "Total Loss Value: 1.5083" in out
    assert "Enforced Drop-out Vector Count: 6" in out
    assert "(\\nabla_x L): 0.1581" in out
    assert "(\\nabla_W_g L): 0.0546" in out

def test_sandbox_runs_both_modes():
    r = subprocess.run([sys.executable, "sandbox.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("SANDBOX EXECUTION METRICS") == 2
