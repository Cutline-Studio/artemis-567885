"""One codec interface over the two tokenizers the dense model accepts (config.tokenizer):
  byte  ids = UTF-8 bytes (vocab 256, the default; every existing checkpoint/test uses it)
  bpe   ids from the FIXED tokenizer.json trained by artemis_dense/tokenizer.py (chat-template
        tokens <|system|> <|user|> <|assistant|> <|tool|> <|end|> are single ids)
get_codec(cfg) -> codec with .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size,
.n_bytes(str) (UTF-8 byte count; used for bits-per-byte, the vocabulary-independent metric)."""
import functools


class ByteCodec:
    kind, vocab_size = "byte", 256

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids if 0 <= int(i) < 256).decode("utf-8", "replace")

    def token_id(self, s: str):
        return None


class BPECodec:
    kind = "bpe"

    def __init__(self, path: str):
        from .tokenizer import load_tokenizer
        self.path, self.tk = path, load_tokenizer(path)
        self.vocab_size = self.tk.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list:
        return self.tk.encode(text).ids

    def decode(self, ids) -> str:
        return self.tk.decode([int(i) for i in ids], skip_special_tokens=False)

    def token_id(self, s: str):
        return self.tk.token_to_id(s)


def n_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


@functools.lru_cache(maxsize=8)
def _bpe(path: str) -> BPECodec:
    return BPECodec(path)


def get_codec(cfg=None, kind: str = None, path: str = None):
    kind = kind or (getattr(cfg, "tokenizer", "byte") if cfg is not None else "byte")
    if kind == "byte":
        return ByteCodec()
    path = path or cfg.tokenizer_path
    c = _bpe(path)
    if cfg is not None and c.vocab_size != cfg.vocab_size:
        raise ValueError(f"tokenizer vocab {c.vocab_size} != model vocab {cfg.vocab_size}: the vocabulary must "
                         "stay fixed after training")
    return c
