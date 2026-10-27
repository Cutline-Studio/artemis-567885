# ByteModel copied verbatim from the user's pasted script (this class was fully visible).
import torch
import torch.nn as nn

class ByteModel(nn.Module):
    """Two-layer causal byte transformer: 37,888 parameters at context=128."""

    def __init__(self, context=128):
        super().__init__()
        self.byte = nn.Embedding(256, 32)
        self.position = nn.Embedding(context, 32)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=32,
                nhead=2,
                dim_feedforward=64,
                dropout=0.0,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 256)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.byte(tokens) + self.position(positions)

        future_mask = torch.ones(
            length, length, dtype=torch.bool, device=tokens.device
        ).triu(1)

        for block in self.blocks:
            x = block(x, src_mask=future_mask)

        return self.head(self.norm(x))

torch.manual_seed(0)
m = ByteModel(128)
total = sum(p.numel() for p in m.parameters())
print("param_count", total)
for n, mod in [("byte", m.byte), ("position", m.position), ("block0", m.blocks[0]), ("norm", m.norm), ("head", m.head)]:
    print(" ", n, sum(p.numel() for p in mod.parameters()))
print("nhead=1 vs 2 vs 4 counts:", [sum(p.numel() for p in nn.TransformerEncoderLayer(32, h, 64, batch_first=True).parameters()) for h in (1, 2, 4)])

for mode in ("eval", "train"):
    getattr(m, mode)()
    L, t = 64, 40
    x = torch.randint(0, 256, (4, L))
    x2 = x.clone(); x2[:, t] = (x2[:, t] + 1) % 256
    with torch.no_grad():
        a, b = m(x), m(x2)
    print(mode, "forward: max|delta| positions <t:", (a[:, :t] - b[:, :t]).abs().max().item(),
          " at/after t:", (a[:, t:] - b[:, t:]).abs().max().item())

    # gradient through the model's own path: hook embedding sum via wrapper
    emb = (m.byte(x) + m.position(torch.arange(L))).detach().requires_grad_(True)
    mask = torch.ones(L, L, dtype=torch.bool).triu(1)
    h = emb
    for blk in m.blocks:
        h = blk(h, src_mask=mask)
    logits = m.head(m.norm(h))
    logits[:, t - 1].sum().backward()
    g = emb.grad
    print(mode, "grad of logits[:, t-1] wrt emb[:, >=t]: max|g| =", g[:, t:].abs().max().item(),
          " exactly zero:", bool((g[:, t:] == 0).all()), "; wrt emb[:, <t] max|g| =", g[:, :t].abs().max().item())
