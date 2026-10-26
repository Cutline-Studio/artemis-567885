"""Before/after report for trained checkpoints (same held-out text, same prompts, greedy decoding).

  python -m artemis_dense.eval_report --corpus checkpoints/combined_8ace1fe5.txt \
      checkpoints/byte/stage2_final.pt checkpoints/bpe/stage2_final.pt > checkpoints/report.json

For each checkpoint: bits-per-byte on the held-out split (train.split_corpus) for an untrained
model of the same config+seed ("before") and the checkpoint ("after"), the 5 fixed sample prompts,
REPL replies through `python -m artemis_dense.chat --ckpt` (RAG on), and generation speed.
"""
import argparse, io, json, os, sys, time

import torch

from . import chat, train as T
from .engines import transformer as tx
from .tok import get_codec

REPL_LINES = ["My name is Latoya. What can you do?", "define serendipity", "What is 12 * 7?", "/facts"]


def report(path, val_text, seed=567885):
    m = tx.load(path)
    codec = get_codec(m.cfg)
    torch.manual_seed(seed)
    init = tx.build(m.cfg, seed=seed)
    out = {"ckpt": path, "tokenizer": m.cfg.tokenizer, "vocab": m.cfg.vocab_size, "params": m.num_params(),
           "trained_info": m.trained_info,
           "before": T.bits_per_byte(init, codec, val_text), "after": T.bits_per_byte(m, codec, val_text)}
    t0 = time.time()
    out["samples"] = {p: T.sample(m, codec, p) for p in T.SAMPLE_PROMPTS}
    gen_s = time.time() - t0
    n_tok = sum(len(codec.encode(s)) for s in out["samples"].values())
    out["gen_tokens_per_s"] = round(n_tok / gen_s, 1)
    out["gen_bytes_per_s"] = round(sum(len(s.encode()) for s in out["samples"].values()) / gen_s, 1)
    buf = io.StringIO()
    chat.main(["--ckpt", path, "--max-new", "96"], stdin=io.StringIO("\n".join(REPL_LINES) + "\n/quit\n"), stdout=buf)
    out["repl_transcript"] = buf.getvalue()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--corpus", default=os.path.join(T.CORPUS, "combined.txt"))
    a = ap.parse_args(argv)
    _, val = T.split_corpus(open(a.corpus, encoding="utf-8").read())
    res = [report(p, val) for p in a.ckpts]
    json.dump(res, sys.stdout, indent=2, ensure_ascii=False)
    return res


if __name__ == "__main__":
    main()
