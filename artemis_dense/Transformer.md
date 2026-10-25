# Transformer engine

**Status:** built.

`engines/transformer/` is the single entry point to the ONE dense model (no MoE, no router, no
experts). self_train, watch, rag, scaling and conversation all get the model through it.

| function | what |
|---|---|
| `build(size or DenseConfig, seed, **overrides)` | builds after `validate()` (byte vocab 256, caps, effective context ≤ 65,536) |
| `save(model, path)` / `load(path)` | atomic checkpoint: `{"cfg", "model"}` |
| `generate(model, prompt_bytes, max_new_tokens, temperature, top_k, top_p, seed, use_cache)` | greedy or sampled, with the KV cache. Cached and uncached outputs are identical (tested). |
| `embed(model, tokens)` | mean-pooled final hidden state, L2-normalised (used by RAG) |
| `logprob(model, tokens)` | mean next-byte log-prob (self-scoring) |
| `activations`, `attention_maps` | per-layer hidden states and softmax attention (causal, rows sum to 1; tested) |
| `count_params`, `flops_per_token(cfg)` (≈ 2·N_non-emb + 2·L·T·d), `train_flops(N, D) = 6ND` | |
| `model_card(model)` | config, params, FLOPs, heads, encoders, `trained: no` |

The architecture itself lives in `model.py` and `config.py`: pre-norm RMSNorm, GQA with RoPE
(+ linear scaling), SwiGLU, tied embeddings, optional tool and value heads, and image/audio
encoders.
