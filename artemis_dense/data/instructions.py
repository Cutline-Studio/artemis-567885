"""Instruction-tuning data: (instruction, input, output) records from the docs' example dialogues.

Docs mark training dialogues like this (see CreateMdFile.md):
    <!-- example -->
    **User:** question text
    **Artemis:** answer text
Placeholders in parentheses (template stubs) are skipped.
"""
import json, os, re
from typing import List

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EX = re.compile(r"<!--\s*example\s*-->\s*\n\*\*User:\*\*\s*(.+?)\n\*\*Artemis:\*\*\s*(.+?)(?=\n\s*\n|\n<!--|\Z)", re.S)


def extract(md_text: str, source: str = "") -> List[dict]:
    out = []
    for q, a in _EX.findall(md_text):
        q, a = q.strip(), a.strip()
        if q.startswith("(") and a.startswith("("):
            continue
        out.append({"instruction": q, "input": "", "output": a, "source": source})
    return out


def build(pkg: str = PKG, out_path: str = None) -> List[dict]:
    recs = []
    for dp, _, fns in os.walk(pkg):
        if "/data/" in dp + "/" or "/generated" in dp:
            continue
        for fn in sorted(fns):
            if fn.endswith(".md") and fn != "everything.md":
                p = os.path.join(dp, fn)
                recs += extract(open(p, encoding="utf-8").read(), os.path.relpath(p, pkg))
    out_path = out_path or os.path.join(pkg, "data", "corpus", "instructions.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return recs


def to_text(r: dict) -> str:
    return f"### Instruction\n{r['instruction']}\n### Input\n{r['input']}\n### Response\n{r['output']}\n"
