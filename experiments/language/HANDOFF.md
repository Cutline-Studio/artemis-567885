# Language benchmark: state and reproduction (2026-10-25)

Moved out of an ephemeral cloud workspace so the work can continue in Claude Code. The corpus
text, raw downloads, and model checkpoints are NOT committed. Both corpora can be rebuilt
byte-for-byte from their manifests: every document's cleaned SHA-256 is recorded, and the builders
refuse to overwrite.

## Frozen protocol v2 (`protocol_v2.json`, sha256 in `protocol_v2.sha256`)
- Code commit for all v2 runs: `96672bea9a38f7582d1deeba901f6cc59730cc75` (this branch's parent of the handoff commit).
- Development corpus manifest sha256: `ab94040953aeb890566c3f89fa11b54ed068ac12106aa41b4bcf93c2773c8397`.
- Arms: dense, moe-learned, moe-sinusoidal, moe-fourier; seeds 42, 1, 2; 5,000 steps; shuffled
  windows; validation every 1,000 steps; dropless routing in training and evaluation; aux in training loss only.

## Status
All 12 runs complete (2026-10-25). The 5 runs interrupted or never launched on 2026-10-23 were run
from a detached worktree at the frozen commit with `resume_protocol_v2.py`; the reruns reproduce the
interrupted partial validation records bit-for-bit. Report: `results_v2.md` (data: `results_v2.json`).
Append-only ledger: `runs_v2/ledger.jsonl`; interrupted outputs kept in `runs_v2/_interrupted/`.

Reporting labels agreed with the user and Codex:
- Active parameters = "shared parameters plus selected-expert parameters per byte" (an accounting
  convention, not measured FLOPs or memory traffic).
- Budgets match optimizer steps and byte positions, not compute. Report wall time alongside bpb.
- Sinusoidal vs Fourier is the parameter-matched position comparison. Learned-position and dense
  comparisons are qualified by their different parameter counts and architectures.
- Three seeds are preliminary evidence. This corpus is a development benchmark: its test scores have been viewed.

## Confirmation set (untouched; do not score until development results are final)
`confirm_manifest.json`: 45 books, 36 authors, 37,708,556 bytes, 16 translated (translator recorded).
Selected by Gutenberg ID from the user's "Lisa" manifest (`lisa_manifest_ids_only.json`; its text
files were not used), refetched from gutenberg.org with the development cleaning rules. Excluded
(41, each with a reason): same edition as development, author shared with development, or missing
header author. Max word-8-gram containment in the development corpus: 0.053%; none flagged.

## Kept separate
- `original_protocol_dense/`: dense, document-ordered windows, epoch-end validation only (3 seeds).
- `exploratory_causal_capacity_doc_order/`: 8 MoE runs at commit `0195ab7` (causal capacity,
  document-ordered windows). Exploratory only.
- The physics study is not touched by anything in this directory.
