---
name: artemis-transformer
description: Use the Artemis transformer engine — the single entry point to the ONE dense model.
---

# Transformer engine

When to use: whenever Artemis needs to build/validate a model at a scale or save/load checkpoints.

Steps:
1. build/validate a model at a scale
2. save/load checkpoints
3. generate with KV cache and sampling
4. embed text (for RAG)
5. log-prob scoring
6. attention maps and activations
7. param/FLOP counts and a model card

Safety: local only; never loads a second model.

Full doc: [Transformer.md](../../Transformer.md)
