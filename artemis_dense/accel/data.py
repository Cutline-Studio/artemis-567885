"""Methods 6, 7, 8, 13, 14, 15, 16: tokenizer hook, packing, sequence-length warmup, quality
filtering, exact + MinHash near-duplicate removal, synthetic verified problems, curriculum.

Everything is pure-Python/torch over local files. Filtering and dedup run on the TRAIN split only;
the held-out split is never filtered, so validation numbers stay comparable across presets.
"""
import hashlib, math, random, re, zlib
from typing import Callable, Dict, List, Optional, Sequence

import torch

_START = re.compile(r"^<(doc|define|problem|chat|instruction)\b")


# ---- corpus segmentation / split (same algorithm as train.split_corpus) -------------------------
def split_corpus(text: str, block_bytes: int = 2048, every: int = 20):
    blocks, cur, size = [], [], 0
    for line in text.splitlines(keepends=True):
        cur.append(line); size += len(line.encode())
        if size >= block_bytes:
            blocks.append("".join(cur)); cur, size = [], 0
    if cur:
        blocks.append("".join(cur))
    val = [b for i, b in enumerate(blocks) if i % every == every - 1]
    train = [b for i, b in enumerate(blocks) if i % every != every - 1]
    return "".join(train), "".join(val)


def segment(text: str) -> List[str]:
    """Split the combined corpus into records at top-level tags (<doc>, <define>, <problem>, ...)."""
    out, cur = [], []
    for line in text.splitlines(keepends=True):
        if _START.match(line) and cur:
            out.append("".join(cur)); cur = []
        cur.append(line)
    if cur:
        out.append("".join(cur))
    return [d for d in out if d.strip()]


# ---- 6 tokenizer hook -------------------------------------------------------------------------
class _Bytes:
    kind, vocab_size = "byte", 256

    def encode(self, s):
        return list(s.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) for i in ids if 0 <= int(i) < 256).decode("utf-8", "replace")

    def token_id(self, s):
        return None


def get_codec(kind: str = "byte", path: str = ""):
    """Uses the training worker's artemis_dense/tok.py (byte|bpe) when present; byte fallback."""
    try:
        from ..tok import get_codec as gc
        return gc(kind=kind, path=path or None)
    except ImportError:
        if kind != "byte":
            raise RuntimeError("BPE needs artemis_dense/tok.py + tokenizer.py (other worker's files)")
        return _Bytes()


def train_bpe(train_text: str, out_dir: str, vocab_size: int = 4096) -> str:
    """Train the user's BPE tokenizer (artemis_dense/tokenizer.py) on the train split only."""
    import os
    from ..tokenizer import train_tokenizer
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "train_split.txt")
    open(p, "w", encoding="utf-8").write(train_text)
    train_tokenizer([p], out_dir, vocab_size=vocab_size, min_frequency=2)
    return os.path.join(out_dir, "tokenizer.json")


# ---- 7 sequence packing -----------------------------------------------------------------------
def pack_documents(docs_ids: Sequence[List[int]], T: int, sep: Sequence[int] = ()) -> torch.Tensor:
    """Greedy pack whole documents (joined by `sep`) into rows of T+1 tokens with zero padding.
    Long docs are split into T+1 chunks. Every slot of every row is real text."""
    stream: List[int] = []
    for d in docs_ids:
        stream.extend(d); stream.extend(sep)
    n = len(stream) // (T + 1)
    if n == 0:
        raise ValueError("not enough tokens for one packed row")
    return torch.tensor(stream[:n * (T + 1)], dtype=torch.long).view(n, T + 1)


def pack_sft(records: Sequence[tuple], T: int, ignore: int = -100):
    """Pack masked (x, y) SFT records into rows of length T; returns X, Y, fill_fraction.
    (No padding between records; attention can see the previous record in the row, which is the
    standard trade-off of packing without block-diagonal masks.)"""
    rows_x, rows_y, cx, cy = [], [], [], []
    for x, y in records:
        x, y = list(x)[:T], list(y)[:T]
        if len(cx) + len(x) > T:
            rows_x.append(cx + [0] * (T - len(cx))); rows_y.append(cy + [ignore] * (T - len(cy)))
            cx, cy = [], []
        cx += x; cy += y
    if cx:
        rows_x.append(cx + [0] * (T - len(cx))); rows_y.append(cy + [ignore] * (T - len(cy)))
    X, Y = torch.tensor(rows_x), torch.tensor(rows_y)
    real = sum(len(list(x)[:T]) for x, _ in records)
    return X, Y, real / X.numel()


# ---- 8 sequence-length warmup -------------------------------------------------------------------
def seq_len_at(frac: float, T_max: int, T_min: int = 64, warm_frac: float = 0.3, multiple: int = 32) -> int:
    """Linear ramp from T_min to T_max over the first warm_frac of training (rounded to `multiple`)."""
    if warm_frac <= 0 or frac >= warm_frac:
        return T_max
    t = T_min + (T_max - T_min) * frac / warm_frac
    return int(max(T_min, min(T_max, round(t / multiple) * multiple)))


# ---- 13 quality filter --------------------------------------------------------------------------
def quality_score(doc: str) -> Dict[str, float]:
    s = doc.strip()
    n = max(1, len(s))
    words = re.findall(r"\S+", s)
    alpha = sum(c.isalpha() for c in s) / n
    lines = [l for l in s.splitlines() if l.strip()]
    dup_lines = 1 - len(set(lines)) / max(1, len(lines))
    rep = 1 - len(zlib.compress(s.encode())) / max(1, len(s.encode()))   # compressibility ~ repetition
    return {"chars": len(s), "words": len(words), "alpha": alpha, "dup_lines": dup_lines,
            "compress_ratio": rep, "mean_word": sum(map(len, words)) / max(1, len(words))}


def keep(doc: str, min_chars: int = 20, min_alpha: float = 0.25, max_dup_lines: float = 0.5,
         max_compress: float = 0.9, max_mean_word: float = 25.0) -> bool:
    q = quality_score(doc)
    return (q["chars"] >= min_chars and q["alpha"] >= min_alpha and q["dup_lines"] <= max_dup_lines
            and q["compress_ratio"] <= max_compress and q["mean_word"] <= max_mean_word)


def quality_filter(docs: List[str], **kw) -> List[str]:
    return [d for d in docs if keep(d, **kw)]


# ---- 14 exact + near-duplicate removal (MinHash) ---------------------------------------------------
def _shingles(s: str, k: int = 5):
    w = re.findall(r"\w+", s.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def minhash(s: str, num_perm: int = 64, k: int = 5) -> List[int]:
    sh = _shingles(s, k) or {s}
    hs = [int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "little") for x in sh]
    P, M = (1 << 61) - 1, (1 << 64) - 1
    rng = random.Random(1234)
    ab = [(rng.randrange(1, P), rng.randrange(0, P)) for _ in range(num_perm)]
    return [min(((a * h + b) % P) for h in hs) for a, b in ab]


def jaccard_est(m1, m2) -> float:
    return sum(a == b for a, b in zip(m1, m2)) / len(m1)


def dedup(docs: List[str], threshold: float = 0.8, num_perm: int = 64, bands: int = 16) -> List[str]:
    """Exact (sha1 of normalised text) then MinHash-LSH near-duplicate removal; first copy wins."""
    seen, out, sigs, buckets = set(), [], [], {}
    r = num_perm // bands
    for d in docs:
        h = hashlib.sha1(" ".join(d.split()).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        m = minhash(d, num_perm)
        cands = set()
        for b in range(bands):
            cands.update(buckets.get((b, tuple(m[b * r:(b + 1) * r])), ()))
        if any(jaccard_est(m, sigs[c]) >= threshold for c in cands):
            continue
        idx = len(out)
        out.append(d); sigs.append(m)
        for b in range(bands):
            buckets.setdefault((b, tuple(m[b * r:(b + 1) * r])), []).append(idx)
    return out


# ---- 15 synthetic verified problems ------------------------------------------------------------
def synthetic_problems(n_math: int = 300, n_physics: int = 100, seed: int = 7, exclude: str = "") -> List[str]:
    """New problems from the math/physics engines; only engine-verified answers are kept, and any
    problem whose question text occurs in `exclude` (the held-out split) is dropped."""
    from ..engines.math import problem_set as mps
    from ..engines.physics import problem_set as pps
    out, seen = [], set()
    for r in mps(n_math, seed=seed) + pps(n_physics, seed=seed):
        if not r.get("verified") or r["q"] in seen or (exclude and r["q"] in exclude):
            continue
        seen.add(r["q"])
        a = ", ".join(r["a"]) if isinstance(r["a"], list) else str(r["a"])
        out.append(f'<problem verified="true">\nQ: {r["q"]}\nA: {a}\n</problem>\n')
    return out


# ---- 16 curriculum ------------------------------------------------------------------------------
def difficulty(doc: str) -> float:
    """Cheap proxy: longer text with a higher character entropy is harder."""
    from collections import Counter
    n = max(1, len(doc))
    ent = -sum(c / n * math.log2(c / n) for c in Counter(doc).values())
    return math.log1p(len(doc.encode())) * (1 + ent / 8)


def curriculum_order(docs: List[str]) -> List[str]:
    return sorted(docs, key=difficulty)


def competence(frac: float, start: float = 0.3, full_at: float = 0.6) -> float:
    """Fraction of the easiest-first data available at training fraction `frac` (sqrt pacing)."""
    if frac >= full_at:
        return 1.0
    return min(1.0, math.sqrt(start ** 2 + (1 - start ** 2) * frac / full_at))
