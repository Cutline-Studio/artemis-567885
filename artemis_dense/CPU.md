# CPU: smoke tests and CPU training

**Status:** built. Everything in this repo runs on CPU today.

Artemis is one dense LLM. On CPU it runs the same model as it would on a GPU, only slower.

## What runs on CPU
- **The test suite**: `/workspace/artemis-dense-venv/bin/python -m pytest -q tests/` (seconds to a
  few minutes). Tests build tiny models, run a few optimiser steps and check behaviour, not
  quality.
- **Smoke runs**: short training loops that prove the code works end to end. Examples:
  `watch.imitate` behaviour cloning, `self_train.run_cycle`, and the scaling sweep in
  [ScalingLaws.md](ScalingLaws.md) (five sizes, 300 steps each, measured 2026-10-25).
- **The sandbox**: `sandbox.env` pins `ARTEMIS_DEVICE=cpu`, `ARTEMIS_MODEL_SIZE=tiny`, 2 threads,
  2048 MB of memory and a 300 s timeout.
- **CPU training on the combined corpus**: Latoya OK'd a CPU run on 2026-10-25. It uses the
  combined corpus (`data/corpus/combined.txt`) in one session. CPU time on this computer has no
  cloud charges.
- **Chat**: `python -m artemis_dense.chat --size tiny` runs the REPL on CPU.

## This computer
8 vCPUs (Intel Xeon), about 15 GB of RAM, PyTorch 2.14.1 CPU build. It is shared with other
work, so timings vary.

## What CPU is good and bad at here
- Good: tests, debugging, the tiny and small sizes, short runs on a small corpus.
- Bad: the medium size (about 36M parameters) and anything near 20 bytes per parameter, which
  would take far longer than on a GPU.
- Honest limit: a CPU run on the current ~0.38 MB corpus mostly teaches the model to sound like its
  own docs. Broad, sensible conversation needs much more data and a GPU run ([GPU.md](GPU.md)).

## Rules
- CPU runs need no spend approval, but they still follow the sandbox and kill switch.
- Results are reported with the command, seed, versions and raw numbers. Estimates are labelled.

## Example dialogues (training)
<!-- example -->
**User:** Can you train without a GPU?
**Artemis:** Yes, on CPU, with no cloud charges. It's fine for my tiny and small sizes on my own docs, but real conversation skill needs more data and a GPU run you approve.

## Related
[GPU.md](GPU.md) · [training.md](training.md) · [Sandbox.md](Sandbox.md) · [everything.md](everything.md)
