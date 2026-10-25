# Dictionary

**Status:** mixed. Artemis's term glossary is written here. The full English dictionary is built
locally and served by `skills.define`.

## English dictionary (data)
- Built by `python -m artemis_dense.data.build_dictionary --wordnet <dir> --gcide <dir>` into
  `data/dictionary/dictionary.jsonl`. It is git-ignored (70 MB), and its counts and source
  checksums are in `data/dictionary/manifest.json`.
- Sources: WordNet 3.1 (Princeton, WordNet licence, BSD-style) and GCIDE 0.54 (GNU, GPL-3.0-or-
  later, based on the public-domain Webster's 1913). Wiktionary was skipped because of disk space.
- 292,125 records (WordNet 3.1: 155,444; GCIDE 0.54: 136,681), 206,674 unique headwords, 31.1 MB of
  definition text. One record per (word, part of speech, source).
- Lookup: `skills.define(word)` / tool `skills.define` / RAG "define X". The corpus includes a
  deterministic 10% sample, capped at 3× the doc bytes.

## Artemis terms, A–Z
- **A2A**: agent-to-agent engine. Signed envelopes, capability tokens, policy gate, loopback only.
- **approve_resume**: the human call that clears the constitution's strict mode.
- **Chat template**: `<|system|> <|user|> <|assistant|> <|tool|> … <|end|>` (Conversation.md).
- **Citation**: `path#section` from the RAG engine, or `dictionary:<source>:<word>`.
- **Clarifying question**: what Artemis asks instead of guessing when a request is ambiguous.
- **Compute-optimal**: the N and D split of a FLOP budget (C ≈ 6ND, D ≈ 20N; ScalingLaws.md).
- **Constitution**: Latoya's custom principles (CustomConstitutionalAI.md). Loose by default,
  strict on escalation.
- **Critical violation**: production path, network, payment, secret or self-modify. It switches
  strict mode on.
- **Dense model**: one transformer where every parameter is active for every token (no MoE).
- **DimensionError**: the physics engine's unit-mismatch error.
- **Function-preserving growth**: adding a layer or widening the FFN with zero-initialised
  outputs, so the model's output doesn't change.
- **Heart**: the engine that applies the ranked values (Heart.md).
- **KV cache**: cached keys and values for fast generation.
- **Key facts**: user facts kept in the system block across context trimming.
- **Kill switch**: a `KILL` file in a run dir or any parent. It stops every engine.
- **Promotion**: moving sandbox output into the repo, the skills store or a promoted checkpoint.
  It always goes through the gate.
- **Promotion gate**: `promotion.PromotionGate`. It runs the constitution before promotion.
- **RAG**: retrieval-augmented generation (RAG.md).
- **RoPE**: rotary position embeddings, with linear scaling for longer context.
- **Sandbox**: `sandbox.env` plus `sandbox.py`. Untrusted output is tested in /tmp/artemis-sandbox
  (Sandbox.md).
- **Scaling law**: loss as a power law of parameters and data (ScalingLaws.md).
- **Strict mode**: the constitution state where promotion is blocked until `approve_resume`.
- **Tool head**: a classifier over `TOOLS` that routes to an engine via the ToolRouter.
- **Transformer engine**: the single entry point to the model (Transformer.md).
- **Unguarded self_author**: the self-authoring engine with its own guards removed at Latoya's
  request. The promotion gate is now its only check.
- **Value head**: the scalar self-score head used for self-scored sample selection.
- **World**: an A2A environment with per-agent, per-environment signed grants (worlds.md).
