# Byte-level LM corpus benchmark (measured; in progress)

- `artemis_corpus_benchmark.py`: user's script, saved verbatim. 2-layer causal byte transformer, 37,888 params (verified).
- `verify_bytemodel.py` + console: param count and causal-mask checks (forward and gradient; future-input changes leave earlier logits exactly unchanged).
- `build_corpus.py` + `corpus/MANIFEST.md`: public-domain Project Gutenberg books, boilerplate removed, split by book and author (train ~12.0 MB, val ~1.23 MB, test ~1.02 MB). Corpus text is not committed; rebuild with the script and check SHA-256 against the manifest.
- `ledger_check/`: rerun of the earlier pasted `artemis_fixed_benchmark.py`. Measured test BPB 4.5745 vs unigram 5.7727 on 76 bytes; the pasted ledger's 7.018 was not reproduced.
- Runs: `python3 artemis_corpus_benchmark.py --data corpus --output runs/seed<S> --seed <S> --epochs 10 --steps 1000` for seeds 42, 1, 2 (CPU, 1 thread). Results will be added in a follow-up commit.
