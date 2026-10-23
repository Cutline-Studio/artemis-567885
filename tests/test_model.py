import torch
import torch.nn.functional as F
from model import DynamicMultiBlockTransformer
from transformer import MoETransformerBlock


def make(layers=2, experts=4):
    torch.manual_seed(0)
    return DynamicMultiBlockTransformer(16, 2, 32, experts, layers, top_k=2, capacity_factor=1.0)


def test_output_shape_equals_input():
    m = make(3)
    x = torch.randn(2, 5, 16)
    y, aux, dropped = m(x)
    assert y.shape == x.shape and aux.dim() == 0 and isinstance(dropped, int)


def test_aux_and_dropped_are_summed_across_layers():
    m = make(3); m.eval()
    x = torch.randn(2, 8, 16)
    y, aux, dropped = m(x)
    h, auxes, drops = x, [], []
    for b in m.blocks:
        h, a, d = b(h)
        auxes.append(a); drops.append(d)
    assert torch.allclose(y, h)
    assert torch.allclose(aux, torch.stack(auxes).sum())
    assert dropped == sum(drops)


def test_grads_reach_all_layers_and_gates():
    m = make(3); m.train()
    x = torch.randn(2, 8, 16, requires_grad=True)
    y, aux, _ = m(x)
    (F.mse_loss(y, torch.sin(x)) + 0.1 * aux).backward()
    assert x.grad.norm() > 0
    for b in m.blocks:
        assert b.moe.gate.weight.grad.norm() > 0
        assert b.attn.q_linear.weight.grad.norm() > 0


def test_add_layer_preserves_outputs_and_new_layer_is_trainable():
    m = make(2); m.eval()
    x = torch.randn(2, 8, 16)
    y0, _, _ = m(x)
    m.add_layer(4)
    assert m.num_layers == 3 and m.num_experts == [4, 4, 4]
    y1, _, _ = m(x)
    assert torch.allclose(y0, y1, atol=1e-6)
    m.train(); xg = x.clone().requires_grad_(True)
    y, aux, _ = m(xg)
    (F.mse_loss(y, torch.sin(xg)) + 0.1 * aux).backward()
    new = m.blocks[-1]
    assert new.attn.out_linear.weight.grad.norm() > 0
    assert any(e.w2.weight.grad is not None and e.w2.weight.grad.norm() > 0 for e in new.moe.experts)


def test_add_expert_preserves_trained_weights_and_grows_gate():
    m = make(1, 4); m.eval()
    old_gate = m.blocks[0].moe.gate.weight.detach().clone()
    old_e2 = {k: v.clone() for k, v in m.blocks[0].moe.experts[2].state_dict().items()}
    new = m.add_expert(0, source_idx=2, noise_std=1e-2, generator=torch.Generator().manual_seed(0))
    moe = m.blocks[0].moe
    assert new == 4 and moe.num_experts == 5 and moe.gate.weight.shape == (5, 16)
    assert torch.equal(moe.gate.weight[:4], old_gate)
    assert torch.allclose(moe.gate.weight[4], old_gate[2], atol=0.1) and not torch.equal(moe.gate.weight[4], old_gate[2])
    for k, v in moe.experts[4].state_dict().items():
        assert torch.allclose(v, old_e2[k], atol=0.1) and not torch.equal(v, old_e2[k])
    for k, v in moe.experts[2].state_dict().items():
        assert torch.equal(v, old_e2[k])
    y, _, _ = m(torch.randn(2, 8, 16))
    assert y.shape == (2, 8, 16)


def test_mismatched_heads_rejected():
    import pytest
    with pytest.raises(ValueError):
        DynamicMultiBlockTransformer(15, 2, 32, 4, 1)
