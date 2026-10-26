#!/usr/bin/env python3
"""Train and use a byte-level BPE tokenizer for a decoder-only language model.

Install: python -m pip install tokenizers

Train on UTF-8 .txt files (one document per line is convenient):
  python -m artemis_dense.tokenizer train --files data/*.txt --out tokenizer --vocab-size 32000

Encode / decode:
  python -m artemis_dense.tokenizer encode --tokenizer tokenizer/tokenizer.json --text "Hello, world!"
  python -m artemis_dense.tokenizer decode --tokenizer tokenizer/tokenizer.json --ids 72 111 108

In model code, load the saved tokenizer.json once and pass the resulting integer
IDs to the embedding layer. Keep this tokenizer and its vocabulary fixed after
training the model; changing token-to-ID assignments invalidates learned weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

try:
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError as exc:  # Give a useful message instead of a long traceback.
    raise SystemExit("Missing dependency. Install it with: python -m pip install tokenizers") from exc


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    # Artemis chat-template tokens (engines/conversation): each maps to ONE id, never split by BPE.
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|end|>",
]


def train_tokenizer(
    files: Sequence[str],
    output_dir: str,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
) -> dict[str, object]:
    """Train a UTF-8 byte-level BPE tokenizer and save tokenizer.json + config."""
    paths = [Path(name) for name in files]
    missing = [str(path) for path in paths if not path.is_file()]
    if not paths or missing:
        raise ValueError("Provide existing training files; missing: " + ", ".join(missing))
    if vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size must be at least {256 + len(SPECIAL_TOKENS)} to fit the byte alphabet "
                         "and special tokens")
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = [AddedToken(token, special=True, normalized=False) for token in SPECIAL_TOKENS]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(path) for path in paths], trainer)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out / "tokenizer.json"
    tokenizer.save(str(tokenizer_path), pretty=True)

    config: dict[str, object] = {
        "tokenizer_type": "byte_level_bpe",
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "requested_vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "training_files": [path.name for path in paths],
        "encoding": "UTF-8",
    }
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def load_tokenizer(path: str) -> Tokenizer:
    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a tokenizer from UTF-8 text files")
    train_parser.add_argument("--files", nargs="+", required=True, help="Text files used as the training corpus")
    train_parser.add_argument("--out", required=True, help="Output directory for tokenizer files")
    train_parser.add_argument("--vocab-size", type=int, default=32_000)
    train_parser.add_argument("--min-frequency", type=int, default=2)

    encode_parser = subparsers.add_parser("encode", help="Encode text to token IDs")
    encode_parser.add_argument("--tokenizer", required=True, help="tokenizer.json or its directory")
    encode_parser.add_argument("--text", required=True)
    encode_parser.add_argument("--add-bos", action="store_true")
    encode_parser.add_argument("--add-eos", action="store_true")

    decode_parser = subparsers.add_parser("decode", help="Decode token IDs to text")
    decode_parser.add_argument("--tokenizer", required=True)
    decode_parser.add_argument("--ids", nargs="+", type=int, required=True)
    decode_parser.add_argument("--skip-special-tokens", action="store_true")

    info_parser = subparsers.add_parser("info", help="Show vocabulary and special-token IDs")
    info_parser.add_argument("--tokenizer", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            config = train_tokenizer(args.files, args.out, args.vocab_size, args.min_frequency)
            print(json.dumps(config, indent=2))
        elif args.command == "encode":
            tokenizer = load_tokenizer(args.tokenizer)
            ids = tokenizer.encode(args.text).ids
            if args.add_bos:
                ids.insert(0, tokenizer.token_to_id("<|bos|>"))
            if args.add_eos:
                ids.append(tokenizer.token_to_id("<|eos|>"))
            print(json.dumps(ids))
        elif args.command == "decode":
            tokenizer = load_tokenizer(args.tokenizer)
            print(tokenizer.decode(args.ids, skip_special_tokens=args.skip_special_tokens))
        else:
            tokenizer = load_tokenizer(args.tokenizer)
            print(json.dumps({
                "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
                "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
            }, indent=2))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
