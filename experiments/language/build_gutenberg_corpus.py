"""Build a small, documented, author-split Project Gutenberg corpus.

Run once; the resulting manifest is frozen and must not be regenerated in place:
    python -I scripts/build_gutenberg_corpus.py --raw raw --out corpus

Outputs
  raw/pg<ID>.txt                   exact bytes as downloaded (kept for audit)
  corpus/{train,val,test}/pg<ID>_<slug>.txt   one cleaned book per file
  corpus/corpus_manifest.json      sources, IDs, authors, rules, bytes, SHA-256, split, checks
"""
import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# (Gutenberg ID, expected author surname substring). Title and full author are read from
# each file's own header; the surname check catches a wrong ID before it enters the corpus.
BOOKS = [
    (1342, "Austen"), (161, "Austen"), (158, "Austen"), (141, "Austen"),
    (98, "Dickens"), (1400, "Dickens"), (730, "Dickens"), (46, "Dickens"),
    (74, "Twain"), (76, "Twain"), (86, "Twain"),
    (1661, "Doyle"), (2852, "Doyle"), (244, "Doyle"),
    (35, "Wells"), (36, "Wells"), (5230, "Wells"),
    (11, "Carroll"), (12, "Carroll"),
    (2701, "Melville"), (11231, "Melville"),
    (84, "Shelley"), (345, "Stoker"),
    (174, "Wilde"), (844, "Wilde"),
    (219, "Conrad"), (974, "Conrad"),
    (1260, "Bront"), (768, "Bront"),
    (120, "Stevenson"), (43, "Stevenson"),
    (215, "London"), (910, "London"),
    (55, "Baum"), (33, "Hawthorne"), (160, "Chopin"), (110, "Hardy"),
]
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
USER_AGENT = "Artemis-567885-corpus-builder/1.0 (research; one-time fetch, 2s spacing)"
SPLIT_SEED = 567885
VAL_FRACTION, TEST_FRACTION = 0.10, 0.10
SHINGLE_WORDS = 8
OVERLAP_FLAG = 0.05

START_RE = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)
END_RE = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)

RULES = [
    "Download https://www.gutenberg.org/cache/epub/<ID>/pg<ID>.txt; keep raw bytes unchanged in raw/.",
    "Decode as UTF-8 (strict); drop a leading BOM.",
    "Read 'Title:' and 'Author:' from the header before the START marker; require the expected author surname.",
    "Keep only the text strictly between the line '*** START OF THE/THIS PROJECT GUTENBERG EBOOK ...' and "
    "the line '*** END OF THE/THIS PROJECT GUTENBERG EBOOK ...'; fail if either marker is missing.",
    "Normalise line endings CRLF/CR -> LF; strip leading and trailing whitespace of the whole body; "
    "append one trailing LF. No other edits (no Unicode normalisation, no removal of in-body notes).",
    "Encode the result as UTF-8; that file is the document. One book = one document.",
    f"Split by author: authors shuffled with random.Random({SPLIT_SEED}); fill val until >= "
    f"{VAL_FRACTION:.0%} of total cleaned bytes, then test until >= {TEST_FRACTION:.0%}; all remaining authors train.",
    "Checks: exact SHA-256 duplicates across splits (fatal); word "
    f"{SHINGLE_WORDS}-gram shingle containment of each val/test document in train and val<->test "
    f"(flagged above {OVERLAP_FLAG:.0%}).",
]


def fetch(book_id, raw_dir):
    path = raw_dir / f"pg{book_id}.txt"
    if path.exists():
        return path.read_bytes(), False
    req = urllib.request.Request(URL.format(id=book_id), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError(f"pg{book_id}: HTTP {resp.status}")
        data = resp.read()
    path.write_bytes(data)
    return data, True


def header_field(header, name):
    m = re.search(rf"^{name}:\s*(.+?)\s*$", header, re.M)
    if not m:
        raise ValueError(f"missing '{name}:' in header")
    # Continuation lines (indented) belong to the same field, e.g. long titles.
    value = m.group(1)
    rest = header[m.end():].split("\n")
    for line in rest[1:]:
        if line.startswith((" ", "\t")) and line.strip():
            value += " " + line.strip()
        else:
            break
    return value


def clean(raw, expected_author):
    text = raw.decode("utf-8")
    if text.startswith("﻿"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start, end = START_RE.search(text), END_RE.search(text)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Gutenberg START/END markers not found in order")
    header = text[:start.start()]
    title, author = header_field(header, "Title"), header_field(header, "Author")
    if expected_author.lower() not in author.lower():
        raise ValueError(f"author mismatch: header says {author!r}, expected *{expected_author}*")
    body = text[start.end():end.start()].strip() + "\n"
    return title, author, body.encode("utf-8")


def slug(s, n=40):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n].strip("-")


def shingles(data):
    words = re.findall(r"[a-z0-9']+", data.decode("utf-8").lower())
    return {hash(" ".join(words[i:i + SHINGLE_WORDS])) for i in range(len(words) - SHINGLE_WORDS + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; the manifest is frozen. Refusing to overwrite.")

    books = []
    for i, (book_id, surname) in enumerate(BOOKS):
        raw, downloaded = fetch(book_id, args.raw)
        if downloaded and i < len(BOOKS) - 1:
            time.sleep(2.0)
        title, author, body = clean(raw, surname)
        books.append({
            "gutenberg_id": book_id, "title": title, "author": author,
            "source_url": URL.format(id=book_id),
            "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            "_body": body,
        })
        print(f"pg{book_id:<6} {len(body):>9,} B  {author}  |  {title[:60]}", flush=True)

    # Author-level split
    by_author = defaultdict(list)
    for b in books:
        by_author[b["author"]].append(b)
    authors = sorted(by_author)
    random.Random(SPLIT_SEED).shuffle(authors)
    total = sum(b["bytes"] for b in books)
    split_of, acc = {}, {"val": 0, "test": 0}
    for a in authors:
        size = sum(b["bytes"] for b in by_author[a])
        if acc["val"] < VAL_FRACTION * total:
            split_of[a] = "val"; acc["val"] += size
        elif acc["test"] < TEST_FRACTION * total:
            split_of[a] = "test"; acc["test"] += size
        else:
            split_of[a] = "train"
    for b in books:
        b["split"] = split_of[b["author"]]

    # Exact-duplicate check (fatal across splits)
    seen = {}
    for b in books:
        if b["sha256"] in seen and seen[b["sha256"]] != b["split"]:
            raise SystemExit(f"cross-split exact duplicate: pg{b['gutenberg_id']}")
        seen[b["sha256"]] = b["split"]

    # Substantial-overlap check (shingle containment)
    sh = {b["gutenberg_id"]: shingles(b["_body"]) for b in books}
    pool = {s: set().union(*[sh[b["gutenberg_id"]] for b in books if b["split"] == s]) for s in ("train", "val", "test")}
    overlap = []
    for b in books:
        if b["split"] == "train":
            continue
        own = sh[b["gutenberg_id"]]
        other = "test" if b["split"] == "val" else "val"
        for ref in ("train", other):
            frac = len(own & pool[ref]) / max(1, len(own))
            overlap.append({"gutenberg_id": b["gutenberg_id"], "split": b["split"], "against": ref,
                            "shingle_containment": round(frac, 5), "flagged": frac > OVERLAP_FLAG})

    # Write documents
    for b in books:
        d = args.out / b["split"]
        d.mkdir(parents=True, exist_ok=True)
        name = f"pg{b['gutenberg_id']}_{slug(b['title'])}.txt"
        (d / name).write_bytes(b.pop("_body"))
        b["path"] = f"{b['split']}/{name}"

    summary = {s: {"books": sum(b["split"] == s for b in books),
                   "authors": sorted({b["author"] for b in books if b["split"] == s}),
                   "bytes": sum(b["bytes"] for b in books if b["split"] == s)} for s in ("train", "val", "test")}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Artemis 567885 language benchmark corpus: public-domain Project Gutenberg books, author-split.",
        "license_note": "Project Gutenberg eBooks in the US public domain; Gutenberg headers/footers removed per rules.",
        "user_agent": USER_AGENT,
        "processing_rules": RULES,
        "split_seed": SPLIT_SEED,
        "summary": summary,
        "overlap_check": {"method": f"word {SHINGLE_WORDS}-gram shingle containment",
                          "flag_threshold": OVERLAP_FLAG,
                          "max_containment": max((o["shingle_containment"] for o in overlap), default=0.0),
                          "flagged": [o for o in overlap if o["flagged"]],
                          "all": overlap},
        "documents": sorted(books, key=lambda b: (b["split"], b["gutenberg_id"])),
    }
    (args.out / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("max shingle containment:", manifest["overlap_check"]["max_containment"],
          "| flagged:", len(manifest["overlap_check"]["flagged"]))


if __name__ == "__main__":
    main()
