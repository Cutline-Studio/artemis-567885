"""Config for the Artemis 567885 dense model ("one big brain").

Plain dataclass, JSON round-trippable, in the same spirit as the expand engine's
config.json (flat keys, ints/floats only).
"""
from dataclasses import dataclass, asdict, field
import json


@dataclass
class DenseConfig:
    vocab_size: int = 256          # raw bytes, same input space as bytelm/ (nn.Embedding(256, ...))
    context_len: int = 256         # max sequence length (RoPE cache size)
    n_layers: int = 8
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 8            # == n_heads -> standard MHA; < n_heads -> grouped-query attention
    d_ff: int = 768                # SwiGLU hidden width
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0      # linear position interpolation: positions are divided by this
                                   # factor, so a model trained at L can run at L * rope_scaling
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02
    tool_head: bool = False        # small classifier over TOOLS (adds d_model*len(TOOLS) params)
    value_head: bool = False       # scalar self-score head (adds d_model + 1 params)
    # growth caps used by engines/self_train.py
    caps: dict = field(default_factory=lambda: {"num_layers_max": 12, "d_ff_max": 1536})

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.rope_scaling < 1.0:
            raise ValueError("rope_scaling must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DenseConfig":
        return cls(**d)

    @classmethod
    def preset(cls, name: str, **overrides) -> "DenseConfig":
        return cls(**{**SCALES[name], **overrides})

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DenseConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# tool ids for the optional tool head; index 0 = "no tool"
TOOLS = ("none", "code_exec.run", "a2a.send_task", "skills.lookup", "skills.acquire", "env.read",
         "env.write", "env.execute", "skills.define", "self_author.create_doc", "math.compute",
         "physics.compute")

# Scale ladder ("small" == the default DenseConfig()). Param counts are asserted in tests.
SCALES = {
    "tiny":   dict(n_layers=4,  d_model=128, n_heads=4, n_kv_heads=4, d_ff=384,  context_len=256),
    "small":  dict(n_layers=8,  d_model=256, n_heads=8, n_kv_heads=8, d_ff=768,  context_len=256),
    "medium": dict(n_layers=12, d_model=512, n_heads=8, n_kv_heads=2, d_ff=1536, context_len=1024,
                   caps={"num_layers_max": 16, "d_ff_max": 2048}),
}


def expected_param_count(cfg: DenseConfig) -> int:
    """Closed-form parameter count (no biases anywhere; tied embeddings)."""
    d, f, L, V, hd = cfg.d_model, cfg.d_ff, cfg.n_layers, cfg.vocab_size, cfg.head_dim
    attn = d * d + 2 * d * (cfg.n_kv_heads * hd) + d * d   # wq, wk, wv, wo
    ffn = 3 * d * f                                       # w_gate, w_up, w_down
    norms = 2 * d                                         # attn_norm, ffn_norm
    emb = V * d
    head = 0 if cfg.tie_embeddings else V * d
    extra = (d * len(TOOLS) if cfg.tool_head else 0) + (d + 1 if cfg.value_head else 0)
    return L * (attn + ffn + norms) + emb + head + d + extra   # + final norm
