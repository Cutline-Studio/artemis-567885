# Language benchmark: byte-level causal MoE, position-encoding ablation

Milestone (c) in PLAN.md: language generalization on held-out text. Separate from the physics
study; nothing here reads or writes `physics_simulation.py` or `runs/physics/`.

## Question
Does replacing a positional baseline with multiscale Fourier features change held-out bits per
byte for the same causal byte-level MoE, under a matched training budget?

## Corpus (frozen)
- 37 public-domain Project Gutenberg books, 19 authors, fetched once from
  `https://www.gutenberg.org/cache/epub/<ID>/pg<ID>.txt` by `experiments/language/build_gutenberg_corpus.py`.
- One book = one document. Header/footer removed by one marker rule (text strictly between
  `*** START OF ... PROJECT GUTENBERG EBOOK` and `*** END OF ...`), CRLF -> LF, outer whitespace
  stripped; no other edits. Title/author read from each file's own header and checked against the
  expected author before inclusion.
- Split **by author** (seeded shuffle, val then test filled to >= 10% of bytes each):

  | split | books | authors | bytes |
  |---|---|---|---|
  | train | 28 | 13 | 14,567,516 |
  | val | 6 | 4 (Conrad, London, Chopin, Baum) | 1,901,328 |
  | test | 3 | 2 (Melville, Stoker) | 2,172,605 |

- Checks: no exact cross-split duplicates; max word-8-gram containment of any val/test book in
  another split 0.017% (flag threshold 5%, none flagged). One in-body transcriber's note in
  *Moby-Dick* mentions Project Gutenberg and is kept per the rules.
- Full record (URLs, IDs, authors, rules, raw and cleaned byte counts, SHA-256): 
  `experiments/language/corpus_manifest.json`. The corpus text itself is not committed.
- Limitation: the test split has only two authors, so test bpb reflects those styles heavily;
  report validation alongside it.

## Protocol
`experiments/corpus_benchmark.py` (the user-supplied harness; `--model dense` is unchanged and
reproduces the original script bit-for-bit, checked on a 150-step run).
- Identical across arms for a given seed: corpus, document shuffle, batch composition (16 x 128
  bytes, chunk-reset context), 5,000 Adam steps at lr 1e-3, gradient clip 1.0, one thread.
- Evaluation: every byte except each file's first scored once; untrained model included as a
  checkpoint candidate; checkpoint chosen on validation; test scored once, after selection;
  reloaded checkpoint must reproduce its validation score.
- MoE arms (`byte_model.ArtemisByteLM`): d_model 32, 2 heads, 2 layers, 4 experts (d_ff 64),
  top-2; protocol v2 uses dropless routing (`capacity_mode="none"`); the exploratory runs used
  capacity factor 1.25 with `capacity_mode="causal"`; aux weight 0.01 (training loss
  only; reported bpb is cross-entropy alone).
- Position arms: `fourier` vs `sinusoidal` is the controlled comparison (same feature width 16,
  same Linear(16, 32) projection, identical parameter counts; only the feature map differs).
  `learned` and `none` are reference arms. Dense (37,888 params) is a reference, not
  capacity-matched to the MoE (59,744 total params, top-2 of 4 experts active).

## Causality
- Attention: upper-triangular mask (`causal=True`).
- Routing: the stage-1 capacity rule ranks assignments over the whole flattened batch, so a later
  token (or another document's token) can evict an earlier one; measured 15/15 prefixes changed by
  future-only edits. `capacity_mode="causal"` admits tokens in position order per sequence with
  cap(t) = ceil((t+1) * capacity_factor * top_k / E), so routing at t depends only on positions
  <= t of the same sequence, in training and evaluation. `ArtemisByteLM` refuses batch capacity.
  Tests: `tests/test_causal_positions.py`.

## Results
Protocol v2 (shuffled windows, step-based validation, dropless routing) is frozen in
`experiments/language/protocol_v2.json`. As of 2026-10-25, 7 of 12 runs are complete; the batch was
interrupted by workspace inactivity. See `experiments/language/HANDOFF.md` for status and how to
finish without changing the protocol. The report is produced by `experiments/language/summarize_v2.py`.
The causal-capacity runs described above are exploratory (`exploratory_causal_capacity_doc_order/`).

## Length generalization: separate protocol (not run)
In-context results above say nothing about positions beyond the 128-byte training context.
- `learned` has no vectors past its table; evaluating it there requires an explicit variant
  (e.g. a larger table whose extra rows are never trained), reported as such, or exclusion.
- Fourier features are defined at every t but nothing guarantees useful extrapolation; with
  t = n/128 and octaves up to 2^7, the lowest octave has period 256 bytes, so t > 2 repeats.
- Proposed: evaluate trained checkpoints at contexts 128, 256, 512 on the validation split, plus a
  control that scores the same bytes with a sliding 128-byte window, to separate "more context
  helps" from "the position code extrapolates". Choose any checkpoint on validation at 128 only.
