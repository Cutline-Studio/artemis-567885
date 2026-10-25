"""Conversation-format SFT data for the chat template (engines/conversation): every doc example
dialogue (data/instructions.py) becomes system + user + assistant, rendered with the special tokens.
Writes data/corpus/conversations.jsonl ({"text", "source"}). This is the Stage 2 SFT input
(training.md, Stage 2); nothing is trained here."""
import json, os
from . import instructions
from ..engines.conversation import sft_records

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(pkg=PKG, out_path=None):
    recs = instructions.build(pkg, os.path.join(pkg, "data", "corpus", "instructions.jsonl"))
    texts = sft_records([[{"role": "user", "content": r["instruction"]}, {"role": "assistant", "content": r["output"]}]
                         for r in recs])
    out_path = out_path or os.path.join(pkg, "data", "corpus", "conversations.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for t, r in zip(texts, recs):
            f.write(json.dumps({"text": t, "source": r["source"]}, ensure_ascii=False) + "\n")
    return texts
