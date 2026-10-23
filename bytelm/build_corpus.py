import hashlib, re
from pathlib import Path
SPLITS = {"train": [1023, 766, 2600, 2701, 135], "val": [1342, 105], "test": [76, 74]}
rows = []
for split, ids in SPLITS.items():
    d = Path("corpus") / split; d.mkdir(parents=True, exist_ok=True)
    for i in ids:
        raw = Path(f"raw/pg{i}.txt").read_bytes()
        txt = raw.decode("utf-8")
        title = re.search(r"^Title:\s*(.+)$", txt, re.M).group(1).strip()
        author = re.search(r"^Author:\s*(.+)$", txt, re.M).group(1).strip()
        s = re.search(r"^\*\*\* ?START OF TH(E|IS) PROJECT GUTENBERG EBOOK.*\*\*\*\s*$", txt, re.M)
        e = re.search(r"^\*\*\* ?END OF TH(E|IS) PROJECT GUTENBERG EBOOK.*$", txt, re.M)
        body = txt[s.end():e.start()].strip() + "\n"
        b = body.encode("utf-8")
        out = d / f"pg{i}.txt"; out.write_bytes(b)
        rows.append((split, i, title, author, len(b), hashlib.sha256(b).hexdigest()))
lines = ["# Artemis corpus manifest", "",
 "Source: Project Gutenberg plain-text ebooks (https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt), downloaded 2026-10-23.",
 "License: these works are in the public domain in the United States. The Project Gutenberg header and footer (license boilerplate outside the '*** START OF' / '*** END OF' markers) were removed; per the Project Gutenberg license, the stripped texts carry no Project Gutenberg trademark. Status outside the US was not checked.",
 "Split is at book level with distinct authors per split (train: Dickens, Tolstoy/Maude translation, Melville, Hugo; val: Austen; test: Twain). Builder: build_corpus.py. Bytes are UTF-8 file sizes after stripping.", "",
 "| split | id | title | author | bytes | sha256 |", "|---|---|---|---|---|---|"]
tot = {}
for r in rows:
    lines.append("| %s | %d | %s | %s | %d | %s |" % r); tot[r[0]] = tot.get(r[0], 0) + r[4]
lines += ["", "Totals: " + ", ".join(f"{k} {v:,} bytes" for k, v in tot.items())]
Path("corpus/MANIFEST.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
