import torch
import torch.nn.functional as F
from transformer import MoETransformerBlock

def run_sandbox():
    torch.manual_seed(42)
    D_MODEL = 16
    N_HEADS = 2
    D_FF = 64
    NUM_EXPERTS = 4
    TOP_K = 2
    BATCH_SIZE = 2
    SEQ_LEN = 8
    CAPACITY_FACTOR = 0.8

    print("--- Initializing Artemis 567885 Sparse Sandbox ---")
    moe_block = MoETransformerBlock(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, num_experts=NUM_EXPERTS, top_k=TOP_K, capacity_factor=CAPACITY_FACTOR)
    input_vectors = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True)
    
    moe_block.train()
    outputs, balancing_loss, dropped_count = moe_block(input_vectors)
    target_loss = F.mse_loss(outputs, torch.zeros_like(outputs))
    total_loss = target_loss + (0.1 * balancing_loss)
    total_loss.backward()

    print("\n=== SANDBOX EXECUTION METRICS ===")
    print(f"Total Loss Value: {total_loss.item():.4f}")
    print(f"Enforced Drop-out Vector Count: {dropped_count}")
    print(f"Input Structural Gradient Norm (\\nabla_x L): {input_vectors.grad.norm().item():.4f}")
    print(f"Router Gate Matrix Gradient Norm (\\nabla_W_g L): {moe_block.moe.gate.weight.grad.norm().item():.4f}")

if __name__ == "__main__":
    run_sandbox()
