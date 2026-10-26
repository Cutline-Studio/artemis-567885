# Contributing to Artemis 567885

Thanks for taking a look. This is a research codebase, so the bar is evidence, not polish.

## Ground rules

1. **Measured, or labelled.** Any number in a PR, doc or README needs the exact command, the seed, the library versions (`python3 -c "import torch; print(torch.__version__)"`) and the raw output or log file. Anything not measured is marked *estimate*.
2. **Defaults don't move silently.** New behaviour goes behind an option that is off by default, with a test showing the old path is bit-identical.
3. **Baseline stays reproducible.** On the stage branches, `python3 sandbox.py --mode legacy` must keep printing the recorded baseline numbers.
4. **Negative results are welcome.** If something didn't work, write it down with the numbers.
5. **No secrets, no private data, no model weights** in commits. Checkpoints (`*.pt`) are git-ignored.

## Workflow

1. Fork, then branch from the newest stage branch (see the README).
2. Keep a PR to one idea. Stacked PRs are fine; say which PR it builds on.
3. Run the relevant tests before you push:
   ```bash
   python3 -m pip install torch pytest
   python3 -m pytest -q tests checks
   ```
4. In the PR description, include: what changed, why, the commands you ran and their results.

## Reporting bugs

Open an issue with the bug template: the command, the full traceback, your OS, Python and torch versions, and the branch or commit.

## Licensing

No license has been chosen yet. Until one is added, please open an issue before sending a large contribution.
