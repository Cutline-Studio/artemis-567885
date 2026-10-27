"""Reproducible checks backing docs/REVIEW.md (runs against the ORIGINAL legacy behaviour)."""
import math, time, torch, torch.nn.functional as F
from moe import CapacitySparseMoELayer
from transformer import MoETransformerBlock, MultiHeadAttention

def legacy(layer):
    if hasattr(layer, "legacy"): layer.legacy = True
    return layer

torch.manual_seed(0)
print("== 1. aux loss uses noisy logits + only top-1 slot ==")
m = legacy(CapacitySparseMoELayer(16, 32, 4, top_k=2, capacity_factor=10.0)); m.train()
x = torch.randn(64, 16)
torch.manual_seed(1); _, a1, _ = m(x)
torch.manual_seed(2); _, a2, _ = m(x)
print(f"same input, different noise seeds -> aux {a1.item():.4f} vs {a2.item():.4f} (differs={a1.item()!=a2.item()})")
# top-2 slot never influences aux: force expert routing where top-2 picks are concentrated
m.eval()
with torch.no_grad():
    m.gate.weight.zero_(); m.gate.weight[0,0]=5; m.gate.weight[1,0]=4  # tokens with x0>0 -> top1=0, top2=1
xp = torch.randn(64,16); xp[:,0]=xp[:,0].abs()+0.5
_, a, _ = m(xp)
print(f"all tokens send BOTH slots to experts 0/1 (2 of 4 experts idle); aux={a.item():.4f} (uniform optimum=1.0)")

print("== 2. capacity drops are first-come by token order ==")
m = legacy(CapacitySparseMoELayer(4, 8, 2, top_k=1, capacity_factor=0.5)); m.eval()
with torch.no_grad(): m.gate.weight.zero_(); m.gate.weight[0,0]=1.0
xs = torch.zeros(4,4); xs[:,0] = torch.tensor([0.1,0.2,5.0,6.0])  # all route to expert 0, capacity=1
cap = math.ceil(4/2*0.5*1)
out,_,d = m(xs)
kept = (out - xs).abs().sum(-1) > 0  # legacy copies input for dropped tokens
print(f"capacity={cap}, dropped={d}; token gate scores {xs[:,0].tolist()}; kept tokens (not equal to input copy): {kept.tolist()}  -> token 0 (lowest score) kept, high-score tokens 2,3 dropped")

print("== 3. 'sum==0' fallback is fragile and double counts with residual ==")
m = legacy(CapacitySparseMoELayer(4, 8, 2, top_k=1, capacity_factor=0.5)); m.eval()
with torch.no_grad(): m.gate.weight.zero_(); m.gate.weight[0,0]=1.0
xs = torch.zeros(4,4); xs[:,0]=torch.tensor([1.,2.,3.,4.]); xs[:,1]=1.0
out,_,d = m(xs)
print(f"dropped={d}; for dropped token 3 moe_out == input? {torch.equal(out[3], xs[3])} -> block residual gives x + x = 2x")
blk = legacy_blk = MoETransformerBlock(4,1,8,2,top_k=1,capacity_factor=0.5); blk.moe.legacy=True; blk.eval()
# sum==0 false positive: a processed token whose expert output sums to exactly 0 gets overwritten
m2 = legacy(CapacitySparseMoELayer(4, 8, 2, top_k=1, capacity_factor=10.0)); m2.eval()
with torch.no_grad():
    for e in m2.experts: e.w2.weight.zero_(); e.w2.bias.copy_(torch.tensor([1.,-1.,2.,-2.]))  # output sums to 0
xr = torch.randn(3,4); o,_,d = m2(xr)
print(f"processed tokens with zero-sum expert output (dropped={d}) wrongly replaced by input copy: {torch.equal(o, xr)}")

print("== 4. no causal mask ==")
torch.manual_seed(0); att = MultiHeadAttention(16,2); att.eval()
x = torch.randn(1,8,16); y1 = att(x); x2 = x.clone(); x2[0,-1] += 1.0; y2 = att(x2)
print(f"changing the LAST position changes output at position 0: {not torch.allclose(y1[0,0], y2[0,0])}")

print("== 5. no positional information (permutation equivariant) ==")
torch.manual_seed(0); blk = MoETransformerBlock(16,2,32,4,top_k=2,capacity_factor=10.0); blk.moe.legacy=True; blk.eval()
x = torch.randn(1,8,16); perm = torch.randperm(8)
y,_,_ = blk(x); yp,_,_ = blk(x[:,perm])
print(f"block(permuted x) == permuted block(x): {torch.allclose(yp, y[:,perm], atol=1e-5)}")

print("== 6. gradient reaches gate only via top-k softmax weights + aux ==")
torch.manual_seed(0); m = legacy(CapacitySparseMoELayer(16,32,4,2,10.0)); m.train()
x = torch.randn(32,16, requires_grad=True); out,aux,_ = m(x); out.pow(2).mean().backward()
g_task = m.gate.weight.grad.norm().item(); m.zero_grad()
out,aux,_ = m(x); aux.backward(); g_aux = m.gate.weight.grad.norm().item()
print(f"gate grad from task loss only: {g_task:.4f}; from aux only: {g_aux:.4f}")
m.top_k = 1; m.zero_grad(); out,_,_ = m(x); out.pow(2).mean().backward()
print(f"top_k=1: softmax over a single logit is constant 1.0 -> gate grad from task loss = {m.gate.weight.grad.norm().item():.6f}")

print("== 7. dropped_count counts (token, slot) assignments, not tokens ==")
m = legacy(CapacitySparseMoELayer(4,8,2,top_k=2,capacity_factor=0.25)); m.eval()
x = torch.randn(8,4); out,_,d = m(x)
cap = math.ceil(8/2*0.25*2)
print(f"8 tokens, top_k=2 -> 16 assignments, capacity/expert={cap}; dropped_count={d} (max possible 16, > token count 8 possible)")

print("== 8. per-expert Python loop timing on CPU ==")
torch.set_num_threads(max(1, torch.get_num_threads()))
for E in (4, 16, 64):
    m = legacy(CapacitySparseMoELayer(128,512,E,2,1.25)); m.eval(); x = torch.randn(4096,128)
    with torch.no_grad():
        for _ in range(2): m(x)
        t=time.perf_counter()
        for _ in range(5): m(x)
    print(f"E={E:3d}: {(time.perf_counter()-t)/5*1000:.1f} ms / forward (4096 tokens, d=128, d_ff=512)")
