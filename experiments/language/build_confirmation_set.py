"""Build the untouched confirmation set (one split, "confirm") from Gutenberg IDs.

Selection: books listed in the user-supplied "Lisa" corpus manifest (IDs only; its cleaned files are
NOT used) whose Gutenberg ID is not in the development corpus and whose author is not a development
author. Each book is refetched from gutenberg.org and cleaned with exactly the development rules
(build_gutenberg_corpus.clean). Translator(s) recorded from the header. Nothing here scores a model.

    python -I scripts/build_confirmation_set.py --lisa-manifest <manifest.json> --raw raw --out confirm
"""
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gutenberg_corpus as dev  # noqa: E402

DEV_MANIFEST = Path(__file__).resolve().parent.parent / "corpus" / "corpus_manifest.json"


def norm(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]+", " ", name).split()


def surname(name):
    parts = norm(name)
    return parts[-1] if parts else ""


def header_optional(raw, field):
    text = raw.decode("utf-8").lstrip("﻿").replace("\r\n", "\n")
    start = dev.START_RE.search(text)
    head = text[:start.start()] if start else ""
    try:
        return dev.header_field(head, field)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lisa-manifest", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite a frozen set.")
    args.raw.mkdir(parents=True, exist_ok=True)

    devm = json.loads(DEV_MANIFEST.read_text())
    dev_ids = {d["gutenberg_id"] for d in devm["documents"]}
    dev_surnames = {surname(d["author"]) for d in devm["documents"]}
    lisa = json.loads(args.lisa_manifest.read_text())

    kept, excluded = [], []
    candidates = [b for b in lisa if b["book_id"] not in dev_ids]
    for b in lisa:
        if b["book_id"] in dev_ids:
            excluded.append({"gutenberg_id": b["book_id"], "reason": "same Gutenberg ID as development corpus"})
    for i, b in enumerate(candidates):
        bid = b["book_id"]
        if surname(b["author"]) in dev_surnames:
            excluded.append({"gutenberg_id": bid, "listed_author": b["author"],
                             "reason": "listed author surname matches a development author (conservative)"})
            continue
        try:
            raw, downloaded = dev.fetch(bid, args.raw)
            if downloaded:
                time.sleep(2.0)
            header_author = header_optional(raw, "Author")
            translator = header_optional(raw, "Translator")
            # Same cleaning as development; author check against the header itself, so a book with
            # no Author line is excluded rather than guessed.
            if header_author is None:
                raise ValueError("no Author line in Gutenberg header")
            title, author, body = dev.clean(raw, surname(header_author))
        except Exception as e:  # recorded, never silently dropped
            excluded.append({"gutenberg_id": bid, "listed_author": b["author"], "reason": f"fetch/clean failed: {e}"})
            continue
        if surname(author) in dev_surnames:
            excluded.append({"gutenberg_id": bid, "header_author": author,
                             "reason": "header author surname matches a development author"})
            continue
        kept.append({"gutenberg_id": bid, "title": title, "author": author, "translator": translator,
                     "listed_author_in_lisa_manifest": b["author"],
                     "source_url": dev.URL.format(id=bid),
                     "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(),
                     "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "_body": body})
        print(f"pg{bid:<6} {len(body):>9,} B  {author}" + (f" (tr. {translator})" if translator else ""), flush=True)

    # Exact duplicates vs development (by cleaned hash) and within the set.
    dev_hashes = {d["sha256"] for d in devm["documents"]}
    seen = set()
    for k in kept:
        if k["sha256"] in dev_hashes or k["sha256"] in seen:
            raise SystemExit(f"exact duplicate: pg{k['gutenberg_id']}")
        seen.add(k["sha256"])

    # Substantial overlap: containment of each confirmation book's word 8-grams in ALL development text.
    dev_root = DEV_MANIFEST.parent
    dev_pool = set()
    for d in devm["documents"]:
        dev_pool |= dev.shingles((dev_root / d["path"]).read_bytes())
    overlap = []
    for k in kept:
        own = dev.shingles(k["_body"])
        frac = len(own & dev_pool) / max(1, len(own))
        overlap.append({"gutenberg_id": k["gutenberg_id"], "distinct_8grams": len(own),
                        "containment_in_development": round(frac, 5), "flagged": frac > dev.OVERLAP_FLAG})

    d = args.out / "confirm"
    d.mkdir(parents=True)
    for k in kept:
        name = f"pg{k['gutenberg_id']}_{dev.slug(k['title'])}.txt"
        (d / name).write_bytes(k.pop("_body"))
        k["path"] = f"confirm/{name}"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Artemis 567885 confirmation set: untouched; author-disjoint and edition-disjoint from the development corpus. Not to be scored until development results are final.",
        "selection_source": "Gutenberg IDs from the user-supplied 'Lisa' corpus manifest.json (its cleaned text files were not used)",
        "development_manifest_sha256": hashlib.sha256(DEV_MANIFEST.read_bytes()).hexdigest(),
        "processing_rules": dev.RULES[:6] + [
            "Translator read from the header 'Translator:' line when present (null otherwise).",
            "Excluded: same Gutenberg ID as development; author surname (from the Lisa list or the header) equal "
            "to any development author surname; missing header Author line; any fetch/clean failure.",
            f"Overlap: distinct word {dev.SHINGLE_WORDS}-grams of each book; containment = share of them that also "
            f"occur anywhere in the development corpus (train+val+test); flag above {dev.OVERLAP_FLAG:.0%}."],
        "summary": {"books": len(kept), "authors": sorted({k["author"] for k in kept}),
                    "bytes": sum(k["bytes"] for k in kept),
                    "translated_books": sum(1 for k in kept if k["translator"])},
        "overlap_check": {"max_containment": max((o["containment_in_development"] for o in overlap), default=0.0),
                          "flagged": [o for o in overlap if o["flagged"]], "all": overlap},
        "excluded": excluded,
        "documents": sorted(kept, key=lambda k: k["gutenberg_id"]),
    }
    (args.out / "confirm_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=1)[:1500])
    print("excluded:", len(excluded), "| max containment:", manifest["overlap_check"]["max_containment"],
          "| flagged:", len(manifest["overlap_check"]["flagged"]))


if __name__ == "__main__":
    main()
