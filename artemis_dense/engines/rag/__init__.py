"""RAG engine: local hybrid retrieval with citations (no external service, no network).

Index: every .md under artemis_dense/ (everything.md excluded; self-authored docs only when the
registry marks them approved + in_corpus), generated math/physics problem sets (data/problems/*.jsonl).
Chunks = markdown sections (split on headings, then ~900-char windows) with metadata
{path, section, sha256 (of the file), text}. The dictionary is NOT chunked: "define X" / "what does X
mean" queries route to skills.define(X) (exact headword lookup over 292k entries).
Retrieval: BM25 (k1=1.5, b=0.75) + dense cosine on the dense model's own mean-pooled embeddings
(transformer.embed). No offline sentence-embedding model is installed on this box, and the dense
model is untrained, so its embeddings carry little meaning: dense_weight defaults to 0.1 and results
are reported for both BM25-only and hybrid. Rerank: weighted sum + bonus for exact phrase / heading hits.
Answer: extractive (best sentences of the top chunks) with citations "path#section"; "I don't know"
when the best normalized BM25 score is below `min_confidence` or no query term matches.
Store: <index_dir>/chunks.jsonl + emb.npy + files.json (sha256 per file) for incremental refresh.
"""
import hashlib, json, math, os, re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an and or of to in on for is are be it this that with as by at from what which how does do "
           "i you we can will not no yes if then than into about".split())


def toks(s: str) -> List[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in STOP]


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def chunk_markdown(text: str, path: str, max_chars: int = 900) -> List[dict]:
    out, section, buf = [], "(top)", []

    def flush():
        body = "\n".join(buf).strip()
        for i in range(0, len(body), max_chars):
            piece = body[i:i + max_chars].strip()
            if piece:
                out.append({"path": path, "section": section, "text": piece})
    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            flush(); buf = []; section = m.group(1).strip()
        else:
            buf.append(line)
    flush()
    return out


class RagEngine:
    def __init__(self, index_dir: str, root: str = PKG, model=None, dense_weight: float = 0.1,
                 min_confidence: float = 0.25, allowed_prefixes: Optional[List[str]] = None):
        self.index_dir, self.root = os.path.abspath(index_dir), os.path.abspath(root)
        os.makedirs(self.index_dir, exist_ok=True)
        self.model, self.dense_weight, self.min_conf = model, dense_weight, min_confidence
        self.allowed = allowed_prefixes
        self.chunks: List[dict] = []
        self.files: Dict[str, str] = {}
        self.emb = None
        self._load()

    # ------------------------------------------------------------ sources
    def sources(self) -> List[str]:
        reg_path = os.path.join(self.root, "docs_index.json")
        blocked = set()
        if os.path.exists(reg_path):
            for d in json.load(open(reg_path))["docs"]:
                if not d.get("in_corpus") or d.get("review") != "approved":
                    blocked.add(d["file"])
        out = []
        for dp, dns, fns in os.walk(self.root):
            dns[:] = [d for d in dns if d not in ("__pycache__", "corpus", "dictionary")]
            for fn in fns:
                rel = os.path.relpath(os.path.join(dp, fn), self.root)
                if (fn.endswith(".md") and fn != "everything.md" and rel not in blocked) or \
                        (rel.startswith(os.path.join("data", "problems")) and fn.endswith(".jsonl")):
                    out.append(rel)
        return sorted(out)

    def _chunks_for(self, rel):
        p = os.path.join(self.root, rel)
        text = open(p, encoding="utf-8", errors="replace").read()
        if rel.endswith(".jsonl"):
            cs = [{"path": rel, "section": f"item {i}", "text": f"Q: {r['q']} A: {r['a']}"}
                  for i, r in enumerate(json.loads(l) for l in text.splitlines() if l.strip())]
        else:
            cs = chunk_markdown(text, rel)
        h = sha256_file(p)
        for c in cs:
            c["sha256"] = h
        return cs, h

    # ------------------------------------------------------------ build / refresh
    def refresh(self) -> dict:
        """Incremental: re-chunk new/changed files, drop deleted ones. Returns counts."""
        current = {r: sha256_file(os.path.join(self.root, r)) for r in self.sources()}
        changed = [r for r, h in current.items() if self.files.get(r) != h]
        removed = [r for r in self.files if r not in current]
        if changed or removed:
            drop = set(changed) | set(removed)
            self.chunks = [c for c in self.chunks if c["path"] not in drop]
            for r in changed:
                cs, h = self._chunks_for(r)
                self.chunks += cs
            self.files = current
            self._build_stats()
            self._save()
        return {"files": len(self.files), "chunks": len(self.chunks), "changed": len(changed), "removed": len(removed)}

    def refresh_file(self, abs_path: str):
        return self.refresh()

    def _build_stats(self):
        self.tf = [Counter(toks(c["section"] + " " + c["text"])) for c in self.chunks]
        self.dl = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avgdl = float(self.dl.mean()) if len(self.dl) else 1.0
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = max(1, len(self.tf))
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.emb = self._embed([c["text"] for c in self.chunks]) if self.model is not None else None

    def _embed(self, texts):
        import torch
        from ..transformer import embed
        T = self.model.cfg.context_len
        out = []
        for i in range(0, len(texts), 32):
            batch = [list(t.encode()[:T]) or [32] for t in texts[i:i + 32]]
            L = max(len(b) for b in batch)
            x = torch.tensor([b + [32] * (L - len(b)) for b in batch])
            out.append(embed(self.model, x).numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.cfg.d_model))

    def _save(self):
        with open(os.path.join(self.index_dir, "chunks.jsonl"), "w") as f:
            for c in self.chunks:
                f.write(json.dumps(c) + "\n")
        json.dump(self.files, open(os.path.join(self.index_dir, "files.json"), "w"))
        if self.emb is not None:
            np.save(os.path.join(self.index_dir, "emb.npy"), self.emb)

    def _load(self):
        p = os.path.join(self.index_dir, "chunks.jsonl")
        if os.path.exists(p):
            self.chunks = [json.loads(l) for l in open(p)]
            self.files = json.load(open(os.path.join(self.index_dir, "files.json")))
            self._build_stats()

    # ------------------------------------------------------------ retrieval
    def bm25(self, q: List[str]) -> np.ndarray:
        k1, b = 1.5, 0.75
        s = np.zeros(len(self.chunks))
        for i, tf in enumerate(self.tf):
            for w in q:
                f = tf.get(w)
                if f:
                    s[i] += self.idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * self.dl[i] / self.avgdl))
        return s

    def retrieve(self, query: str, k: int = 5, hybrid: bool = True, allowed_prefixes=None) -> List[dict]:
        q = toks(query)
        if not self.chunks or not q:
            return []
        s = self.bm25(q)
        top = s.max()
        norm = s / top if top > 0 else s
        conf = float(top / (sum(self.idf.get(w, 0) for w in set(q)) * 2.5 + 1e-9))   # 0..~1
        score = norm.copy()
        if hybrid and self.emb is not None and self.dense_weight > 0:
            qe = self._embed([query])[0]
            score = (1 - self.dense_weight) * norm + self.dense_weight * (self.emb @ qe)
        ql = query.lower()
        for i, c in enumerate(self.chunks):                      # rerank bonuses
            if any(w in c["section"].lower() for w in q):
                score[i] += 0.1
            if len(ql) > 8 and ql in c["text"].lower():
                score[i] += 0.3
        allowed = allowed_prefixes if allowed_prefixes is not None else self.allowed
        order = np.argsort(-score)
        out = []
        for i in order:
            c = self.chunks[i]
            if allowed is not None and not any(c["path"].startswith(p) for p in allowed):
                continue
            if s[i] <= 0:
                break
            out.append({**c, "score": round(float(score[i]), 4), "bm25": round(float(s[i]), 4),
                        "citation": f"{c['path']}#{c['section']}", "confidence": round(conf, 3)})
            if len(out) == k:
                break
        return out

    def answer(self, query: str, k: int = 3, allowed_prefixes=None) -> dict:
        m = re.match(r"\s*(define|what does)\s+['\"]?([A-Za-z][A-Za-z \-']*?)['\"]?(\s+mean)?\s*\??$", query, re.I)
        if m:
            from ..skills import define
            recs = define(m.group(2))
            if recs:
                return {"answer": recs[0]["definitions"][0], "citations": [f"dictionary:{recs[0]['source']}:{recs[0]['word']}"],
                        "confident": True}
        hits = self.retrieve(query, k, allowed_prefixes=allowed_prefixes)
        if not hits or hits[0]["confidence"] < self.min_conf:
            return {"answer": "I don't know: nothing in my indexed docs answers that confidently.",
                    "citations": [], "confident": False, "best": hits[0]["citation"] if hits else None}
        q = set(toks(query))
        sents = []
        for h in hits:
            for sent in re.split(r"(?<=[.!?])\s+|\n", h["text"]):
                ov = len(q & set(toks(sent)))
                if ov:
                    sents.append((ov, sent.strip(), h["citation"]))
        sents.sort(key=lambda t: -t[0])
        best = sents[:2] or [(0, hits[0]["text"][:300], hits[0]["citation"])]
        return {"answer": " ".join(s for _, s, _ in best), "citations": sorted({c for *_, c in best}),
                "confident": True}
