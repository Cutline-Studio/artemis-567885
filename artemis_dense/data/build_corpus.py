"""Build the combined training corpus for the dense model (CPU/GPU agnostic; no training here).

  python3 -m artemis_dense.data.build_corpus [--dict-fraction 0.10] [--seed 567885]

Contents of data/corpus/combined.txt:
  1. every doc in docs_index.json with in_corpus=True AND review=approved (everything.md excluded:
     it duplicates the docs), each wrapped in <doc file=...> markers
  2. instruction records from the docs' example dialogues (data/instructions.py), as text
  2b. the same dialogues in chat-template form (data/conversations.py, <chat> blocks; Stage 2 SFT)
  2c. engine-verified math/physics problems (data/problems/*.jsonl from data/build_problems.py)
  3. a deterministic SAMPLE of the dictionary (default 10% of entries, capped so dictionary bytes
     <= 3x the doc bytes) so 31 MB of definitions don't drown out the ~0.2 MB of Artemis docs.
     The full dictionary stays available to the model at run time via the skills.define tool.
Writes data/corpus/manifest.json with per-file bytes (= byte tokens for the byte-level model).
"""
import argparse, hashlib, json, os, random, sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(PKG), "scripts"))
import docs_registry  # noqa: E402
from artemis_dense.data import instructions  # noqa: E402


def main(argv=None, root=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict-fraction", type=float, default=0.10)
    ap.add_argument("--dict-cap-ratio", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=567885)
    a = ap.parse_args(argv)
    root = root or os.path.dirname(PKG)
    pkg = docs_registry.pkg(root)
    out_dir = os.path.join(pkg, "data", "corpus")
    os.makedirs(out_dir, exist_ok=True)
    files, chunks = [], []
    for d in docs_registry.load(root)["docs"]:
        if d["file"] == "everything.md" or not d.get("in_corpus") or d.get("review") != "approved":
            continue
        p = os.path.join(pkg, d["file"])
        if not os.path.exists(p):
            continue
        body = open(p, encoding="utf-8").read()
        chunks.append(f'<doc file="{d["file"]}">\n{body}\n</doc>\n')
        files.append({"file": d["file"], "bytes": len(body.encode())})
    doc_bytes = sum(f["bytes"] for f in files)
    recs = instructions.build(pkg, os.path.join(out_dir, "instructions.jsonl"))
    inst_text = "".join(f"<instruction>\n{instructions.to_text(r)}</instruction>\n" for r in recs)
    chunks.append(inst_text)
    from artemis_dense.data import conversations
    conv = conversations.build(pkg)
    conv_text = "".join(f"<chat>\n{t}</chat>\n" for t in conv)
    chunks.append(conv_text)
    prob_text, prob_counts = "", {}
    for name in ("math", "physics"):
        pp = os.path.join(pkg, "data", "problems", f"{name}.jsonl")
        if os.path.exists(pp):
            rs = [json.loads(l) for l in open(pp) if l.strip()]
            rs = [r for r in rs if r.get("verified")]
            prob_counts[name] = len(rs)
            prob_text += "".join(f"<problem domain=\"{name}\">Q: {r['q']}\nA: {r['a']}</problem>\n" for r in rs)
    chunks.append(prob_text)
    dict_path = os.path.join(pkg, "data", "dictionary", "dictionary.jsonl")
    dict_bytes, dict_entries = 0, 0
    if os.path.exists(dict_path):
        rng = random.Random(a.seed)
        cap = int(doc_bytes * a.dict_cap_ratio)
        for line in open(dict_path, encoding="utf-8"):
            if rng.random() >= a.dict_fraction:
                continue
            e = json.loads(line)
            t = f"<define word=\"{e['word']}\" pos=\"{e['pos']}\" source=\"{e['source']}\">{' | '.join(e['definitions'])}</define>\n"
            if dict_bytes + len(t.encode()) > cap:
                break
            chunks.append(t); dict_bytes += len(t.encode()); dict_entries += 1
    text = "".join(chunks)
    out = os.path.join(out_dir, "combined.txt")
    open(out, "w", encoding="utf-8").write(text)
    m = {"file": "combined.txt", "total_bytes": len(text.encode()), "byte_tokens": len(text.encode()),
         "sha256": hashlib.sha256(text.encode()).hexdigest(), "docs": files, "doc_bytes": doc_bytes,
         "instruction_records": len(recs), "instruction_bytes": len(inst_text.encode()),
         "conversation_records": len(conv), "conversation_bytes": len(conv_text.encode()),
         "problems": prob_counts, "problem_bytes": len(prob_text.encode()),
         "dictionary": {"fraction": a.dict_fraction, "cap_ratio": a.dict_cap_ratio, "seed": a.seed,
                        "entries": dict_entries, "bytes": dict_bytes},
         "excluded": ["everything.md", "docs with review=pending or in_corpus=false"]}
    json.dump(m, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in m.items() if k != "docs"}, indent=2))
    return m


if __name__ == "__main__":
    main()
