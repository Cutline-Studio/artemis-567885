"""Build artemis_dense/data/dictionary/dictionary.jsonl from openly licensed dictionaries.

Sources (download, verify sha256, then run):
  WordNet 3.1   https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz   (WordNet 3.0/3.1 licence, BSD-style)
  GCIDE 0.54    https://ftp.gnu.org/gnu/gcide/gcide-0.54.tar.xz       (GPL-3.0-or-later; based on the
                                                                       public-domain Webster's 1913)
Usage:
  python -m artemis_dense.data.build_dictionary --wordnet <dir with data.noun ...> --gcide <gcide-0.54 dir>
Record: {"word", "pos", "definitions": [...], "examples": [...], "synonyms": [...], "source"}
One record per (normalized word, pos, source); duplicates merged, definitions de-duplicated in order.
"""
import argparse, glob, hashlib, html, json, os, re, unicodedata
from collections import OrderedDict

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary")
WN_POS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
GC_POS = [("n.", "noun"), ("v. t.", "verb"), ("v. i.", "verb"), ("v.", "verb"), ("a.", "adjective"),
          ("adj.", "adjective"), ("adv.", "adverb"), ("prep.", "preposition"), ("conj.", "conjunction"),
          ("interj.", "interjection"), ("pron.", "pronoun")]


def norm_word(w: str) -> str:
    w = unicodedata.normalize("NFKC", w).replace("_", " ")
    w = re.sub(r'["*`\'\u00b4]', "", w) if not re.fullmatch(r"[\w\- ]*'[\w]+", w) else w.replace("*", "")
    return re.sub(r"\s+", " ", w).strip().lower()


def clean(s: str) -> str:
    s = re.sub(r"<[a-zA-Z]+/", "", s)                 # GCIDE entity forms like <ae/
    s = re.sub(r"<[^>]*>", "", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def add(entries, word, pos, source, defs=(), exs=(), syns=()):
    key = (norm_word(word), pos, source)
    if not key[0]:
        return
    e = entries.setdefault(key, {"word": key[0], "pos": pos, "definitions": [], "examples": [],
                                 "synonyms": [], "source": source})
    for field, vals in (("definitions", defs), ("examples", exs), ("synonyms", syns)):
        for v in vals:
            v = v.strip()
            if v and v not in e[field] and (field != "synonyms" or norm_word(v) != key[0]):
                e[field].append(v if field != "synonyms" else norm_word(v))


def parse_wordnet(d: str, entries):
    for f in ("data.noun", "data.verb", "data.adj", "data.adv"):
        for line in open(os.path.join(d, f), encoding="utf-8", errors="replace"):
            if line.startswith("  "):                  # licence header lines
                continue
            head, _, gloss = line.partition(" | ")
            parts = head.split()
            pos = WN_POS[parts[2]]
            n = int(parts[3], 16)
            words = [re.sub(r"\(.*\)$", "", parts[4 + 2 * i]) for i in range(n)]
            chunks = [c.strip() for c in gloss.strip().split(";")]
            defs = [c for c in chunks if c and not c.startswith('"')]
            exs = [c.strip('"') for c in chunks if c.startswith('"')]
            for w in words:
                add(entries, w, pos, "wordnet-3.1", ["; ".join(defs)] if defs else [], exs,
                    [x for x in words if x != w])


def gc_pos(raw: str) -> str:
    raw = clean(raw)
    for k, v in GC_POS:
        if raw.startswith(k):
            return v
    return raw or "unknown"


def parse_gcide(d: str, entries):
    ent = None
    for f in sorted(glob.glob(os.path.join(d, "CIDE.*"))):
        text = open(f, encoding="utf-8", errors="replace").read()
        for para in re.findall(r"<p>(.*?)</p>", text, re.S):
            m = re.search(r"<ent>(.*?)</ent>", para)
            if m:
                ent = clean(m.group(1))
            hw = re.search(r"<hw>(.*?)</hw>", para)
            word = ent or (clean(hw.group(1)) if hw else None)
            defs = [clean(x) for x in re.findall(r"<def>(.*?)</def>", para, re.S)]
            if not word or not defs:
                continue
            pm = re.search(r"<pos>(.*?)</pos>", para)
            exs = [clean(x) for x in re.findall(r"<as>(.*?)</as>", para, re.S)]
            syns = []
            for s in re.findall(r"<syn>(.*?)</syn>", para, re.S):
                syns += [clean(x) for x in re.findall(r"<mcol>(.*?)</mcol>", s)] or \
                        [x.strip() for x in re.sub(r"^\s*Syn\.\s*--", "", clean(s)).split(";")[0].split(",")]
            add(entries, word, gc_pos(pm.group(1)) if pm else "unknown", "gcide-0.54", defs, exs,
                [s for s in syns if s and len(s) < 40])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wordnet"); ap.add_argument("--gcide"); ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--archives", nargs="*", default=[])
    a = ap.parse_args(argv)
    entries = OrderedDict()
    if a.wordnet:
        parse_wordnet(a.wordnet, entries)
    if a.gcide:
        parse_gcide(a.gcide, entries)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "dictionary.jsonl")
    per_src, heads, nbytes, nwords = {}, set(), 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for e in sorted(entries.values(), key=lambda e: (e["word"], e["pos"], e["source"])):
            line = json.dumps(e, ensure_ascii=False)
            f.write(line + "\n")
            per_src[e["source"]] = per_src.get(e["source"], 0) + 1
            heads.add(e["word"])
            text = " ".join(e["definitions"] + e["examples"])
            nbytes += len(text.encode()); nwords += len(text.split())
    manifest = {"file": "dictionary.jsonl", "entries": len(entries), "entries_per_source": per_src,
                "unique_headwords": len(heads), "definition_text_bytes": nbytes,
                "byte_tokens": nbytes, "whitespace_words": nwords,
                "file_bytes": os.path.getsize(out), "file_sha256": sha256(out),
                "sources": [{"name": os.path.basename(p), "sha256": sha256(p)} for p in a.archives]}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
