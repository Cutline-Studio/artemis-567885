# Language benchmark: handoff (state as of 2026-10-25)

Moved out of an ephemeral cloud workspace so the work can continue in Claude Code. The corpus
text, raw downloads, and model checkpoints are NOT committed. Both corpora can be rebuilt
byte-for-byte from their manifests: every document's cleaned SHA-256 is recorded, and the builders
refuse to overwrite.

## Frozen protocol v2 (`protocol_v2.json`, sha256 in `protocol_v2.sha256`)
- Code commit for all v2 runs: `96672bea9a38f7582d1deeba901f6cc59730cc75` (this branch's parent of the handoff commit).
- Development corpus manifest sha256: `ab94040953aeb890566c3f89fa11b54ed068ac12106aa41b4bcf93c2773c8397`.
- Arms: dense, moe-learned, moe-sinusoidal, moe-fourier; seeds 42, 1, 2; 5,000 steps; shuffled
  windows; validation every 1,000 steps; dropless routing in training and evaluation; aux in training loss only.

## Status (append-only ledger: `runs_v2/ledger.jsonl`)
| arm | seed 42 | seed 1 | seed 2 |
|---|---|---|---|
| dense | completed | completed | interrupted (6 validations, no metrics) |
| moe-learned | completed | completed | not launched |
| moe-sinusoidal | completed | completed | not launched |
| moe-fourier | completed | interrupted (5 validations, no metrics) | not launched |

The two interrupted runs were stopped when the workspace went idle, not by an error. Their
partial `validation.jsonl` files are kept here and must not be counted as results.

## To finish the batch without changing the protocol
1. Rebuild the corpus: `python -I experiments/language/build_gutenberg_corpus.py --raw raw --out corpus`,
   then check that `sha256sum corpus/corpus_manifest.json` matches the value above. If the manifest
   differs only in `created_utc`, compare per-document `sha256` values instead. Every one must match.
2. Check out `96672be` in a separate worktree. `run_protocol_v2.py` refuses to launch unless HEAD
   equals the frozen commit and tracked files are clean.
3. `run_protocol_v2.py` uses absolute paths from the old workspace (`/home/claude/...`). Edit only
   `ROOT`/`CODE` for the new machine, and record that edit in the ledger.
4. Rerun the 5 incomplete runs into fresh output directories (the originals are kept). Log new
   launches to the same ledger.
5. `python -I summarize_v2.py` writes the report: bpb and routing tables kept separate, per-seed
   paired differences, selected checkpoints, and drop totals.

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
