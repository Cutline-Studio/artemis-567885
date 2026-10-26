# Training: how Artemis learns

**Status:** mixed. The corpus builder, the model, the self-training cycles and the CPU smoke tests
are built. Stage 2 SFT and later stages are planned. **Every GPU run needs Latoya's approval.**

Artemis is one dense LLM: a byte-level, decoder-only transformer (`model.py`, `config.py`). Every
stage below trains the same weights. Sizes come from the scale ladder in `config.py`:

| size | layers | d_model | heads (kv) | d_ff | context | params |
|---|---|---|---|---|---|---|
| tiny | 4 | 128 | 4 (4) | 384 | 256 | 885,888 |
| small (default) | 8 | 256 | 8 (8) | 768 | 256 | 6,885,632 |
| medium | 12 | 512 | 8 (2, GQA) | 1536 | 1024 | 36,319,744 |

## One combined session
Latoya asked for every training doc to be combined into one training set and trained in **one
session**, not separate runs. `python3 -m artemis_dense.data.build_corpus` builds
`data/corpus/combined.txt` and `manifest.json` from:
1. every doc in `docs_index.json` with `in_corpus=true` and `review=approved` (everything.md is
   left out because it duplicates the docs);
2. the docs' `<!-- example -->` dialogues as instruction records (`data/instructions.py`);
3. the same dialogues in chat-template form (`data/conversations.py`), the Stage 2 input;
4. engine-verified math and physics problems (`data/problems/`);
5. a deterministic 10% sample of the dictionary, capped at 3× the doc bytes (seed 567885).

Last committed manifest: 377,188 bytes total (86,711 doc bytes, 8 dialogue records, 200 math and
50 physics problems, 1,681 dictionary entries). New docs and their example dialogues are included
the next time the corpus is rebuilt.

## Stages
1. **Stage 1: pre-training** *(CPU run started 2026-10-25)*. Next-byte prediction over the whole
   combined corpus. Teaches spelling, the docs' facts and the chat template's shape.
2. **Stage 2: supervised fine-tuning (SFT)** *(planned)*. Trains on `conversations.jsonl` with
   the loss on **assistant tokens only**, so the model learns to answer, not to imitate the user.
3. **Stage 3: preferences** *(planned)*. DPO on chosen/rejected pairs from the
   [mistake journal](Mistakes.md) (`mistakes.dpo_loss` is built; no run yet).
4. **Stage 4: tools** *(planned)*. Train the tool head (`config.tool_head`) and `<|tool|>` calls
   on logged, permitted tool use.
5. **Ongoing: self-training cycles** *(built, CPU)*. `engines/self_train.py` trains in cycles,
   evaluates, checkpoints or rolls back, and grows within human-set caps. Corrective samples from
   mistakes are replayed first ([Mistakes.md](Mistakes.md)); watched demos add behaviour cloning
   ([Watch.md](Watch.md)).

## Rules
- CPU runs: allowed, no cloud charges ([CPU.md](CPU.md)).
- GPU runs: one GPU, Latoya's OK, price cap, time limit, teardown, real cost reported
  ([GPU.md](GPU.md)).
- A trained checkpoint leaves the sandbox only through `self_train.promote(approved=True)` plus
  the constitution gate. Saving a trained model needs Latoya's explicit yes.
- Every reported result comes with the command, seed, versions and raw numbers.

## Honest limits
The corpus is small (~0.38 MB). Stage 1 on it mostly teaches the model to sound like its own docs.
Broad conversation needs far more data, Stage 2, and a GPU run.

## Example dialogues (training)
<!-- example -->
**User:** What does Stage 2 do?
**Artemis:** It fine-tunes me on example conversations, with the loss only on my replies, so I learn to answer instead of copying the user. It needs a GPU run you approve.

<!-- example -->
**User:** Why are your answers still odd?
**Artemis:** I've only had early training on a small set of my own docs. Stage 2 fine-tuning on more conversations is what makes replies sensible.

## Related
[CPU.md](CPU.md) · [GPU.md](GPU.md) · [Mistakes.md](Mistakes.md) · [Instructions.md](Instructions.md) ·
[Watch.md](Watch.md) · [ScalingLaws.md](ScalingLaws.md) · [everything.md](everything.md)
