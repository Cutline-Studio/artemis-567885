"""Evaluate a stage-6 checkpoint on every non-overlapping window of the held-out (val)
and test splits; unigram baseline fit on train, scored on the same target bytes.
Usage: python3 tools/stage6_full_eval.py runs/stage6_<run>.pt [corpus_dir]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import train_bytes as tb
from model import ArtemisModel

ckpt = sys.argv[1]
root = tb.Path(sys.argv[2] if len(sys.argv) > 2 else tb.DEFAULT_CORPUS)
ck = torch.load(ckpt, weights_only=False)
c = ck["summary"]["config"]
torch.set_num_threads(c.get("threads") or torch.get_num_threads())
m = ArtemisModel(c["d_model"], c["n_heads"], c["d_ff"], c["experts"], c["layers"], top_k=c["top_k"],
                 capacity_factor=None, causal=True, pos_encoding=c["pos"], max_len=c["seq_len"] + 1)
m.load_state_dict(ck["state_dict"])
tr = tb.to_tensor(tb.load_split_dir(root / "train"))
print(f"checkpoint {ckpt}: {ck['summary']['steps']} steps, {ck['summary']['params']:,} params")
for split in ["val", "test"]:
    ho = tb.to_tensor(tb.load_split_dir(root / split))
    w = tb.eval_windows(ho, c["seq_len"] + 1, 10 ** 9)
    t = time.time(); b = tb.evaluate_bpb(m, w, 64)
    print(f"{split}: {w.shape[0]} windows x {c['seq_len']} = {w.shape[0] * c['seq_len']:,} predicted bytes | "
          f"model {b:.4f} bpb | unigram (train) {tb.unigram_bpb(tr, w[:, 1:]):.4f} bpb | {time.time() - t:.0f}s")
