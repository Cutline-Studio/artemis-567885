"""Train the FIXED byte-level BPE tokenizer for the dense model on the Stage 1 TRAIN split only
(the held-out split used for bits-per-byte is never seen by the tokenizer).

  python -m artemis_dense.data.build_tokenizer [--vocab-size 4096] [--out data/tokenizer/bpe4k]

Once a model is trained on a tokenizer, never retrain or edit that tokenizer.json (token-to-id
assignments are baked into the weights); make a new directory instead.
"""
import argparse, json, os, tempfile

from .. import tokenizer as tk
from ..train import CORPUS, PKG, split_corpus


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(PKG, "data", "tokenizer", "bpe4k"))
    a = ap.parse_args(argv)
    if os.path.exists(os.path.join(a.out, "tokenizer.json")):
        raise SystemExit(f"{a.out}/tokenizer.json exists and is fixed; choose a new --out")
    train_text, _ = split_corpus(open(os.path.join(CORPUS, "combined.txt"), encoding="utf-8").read())
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "combined_train_split.txt")
        open(p, "w", encoding="utf-8").write(train_text)
        cfg = tk.train_tokenizer([p], a.out, a.vocab_size, a.min_frequency)
    print(json.dumps(cfg, indent=2))
    return cfg


if __name__ == "__main__":
    main()
