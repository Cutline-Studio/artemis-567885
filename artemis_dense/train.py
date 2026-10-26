"""Stage 1 pre-training + Stage 2 SFT for the ONE dense model, CPU or GPU, byte or BPE tokens.

  python -m artemis_dense.train --size small --tokenizer byte --run byte_small \
      --stage1-steps 3000 --stage2-steps 300 [--tokenizer-path checkpoints/bpe/tokenizer.json]

Data: data/corpus/combined.txt (Stage 1) and Stage 2 chat records built by sft_records():
  conversations.jsonl (doc dialogues) + verified math/physics problems as Q->A chats + dictionary
  entries from the TRAIN split as "define W" -> definition chats.
Held-out split (split_corpus): the corpus is cut into ~2 KB line-aligned blocks and every 20th
block is validation (interleaved, so docs, problems and dictionary all appear in it). Metrics are
reported as bits-per-byte on that exact text, which is comparable across byte and BPE vocabularies.
Stage 2 loss is on assistant tokens only (assistant content + <|end|>); everything else is -100.
Checkpoints go to checkpoints/<run>/ (gitignored); the JSONL log + summary.json record seed,
versions, command, steps, tokens and wall time. No network access, no servers, no GPU rental.
"""
import argparse, json, math, os, platform, random, sys, time

import torch
import torch.nn.functional as F

from .config import DenseConfig
from .engines import transformer as tx
from .engines.conversation import DEFAULT_SYSTEM, END, TOK, sft_records as render_sft
from .tok import get_codec, n_bytes

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)
CORPUS = os.path.join(PKG, "data", "corpus")
IGNORE = -100
SAMPLE_PROMPTS = ["define serendipity", "My name is Latoya. What can you do?", "What is the sandbox for?",
                  "What is 12 * 7?", "What is the constitution?"]


# ------------------------------------------------------------------ data
def split_corpus(text: str, block_bytes: int = 2048, every: int = 20):
    """Deterministic interleaved split -> (train_text, val_text)."""
    blocks, cur, size = [], [], 0
    for line in text.splitlines(keepends=True):
        cur.append(line); size += len(line.encode())
        if size >= block_bytes:
            blocks.append("".join(cur)); cur, size = [], 0
    if cur:
        blocks.append("".join(cur))
    val = [b for i, b in enumerate(blocks) if i % every == every - 1]
    train = [b for i, b in enumerate(blocks) if i % every != every - 1]
    return "".join(train), "".join(val)


def _chat(user: str, assistant: str):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def sft_dialogues(train_text: str, pkg: str = PKG, conv: str = None):
    """Stage 2 dialogues: doc dialogues + verified problems + dictionary entries seen in train_text."""
    import re
    out = []
    conv = conv or os.path.join(pkg, "data", "corpus", "conversations.jsonl")
    from .engines.conversation import ChatTemplate
    if os.path.exists(conv):
        for l in open(conv, encoding="utf-8"):
            if l.strip():
                t = ChatTemplate.parse(json.loads(l)["text"])
                out.append([{"role": x.role, "content": x.content} for x in t if x.role != "system"])
    for name in ("math", "physics"):
        pp = os.path.join(pkg, "data", "problems", f"{name}.jsonl")
        if os.path.exists(pp):
            for l in open(pp):
                r = json.loads(l) if l.strip() else None
                if r and r.get("verified"):
                    out.append(_chat(r["q"], str(r["a"])))
    for m in re.finditer(r'<define word="([^"]+)" pos="([^"]*)" source="[^"]*">(.*?)</define>', train_text):
        word, pos, defs = m.group(1), m.group(2), m.group(3).split(" | ")
        out.append(_chat(f"define {word}", f"{word} ({pos}): {defs[0]}"))
    return out


def encode_masked(dialogue, codec, system: str = DEFAULT_SYSTEM):
    """Encode one chat segment by segment -> (ids, targets) with loss on assistant tokens only."""
    ids, tgt_mask = [], []
    turns = [{"role": "system", "content": system}] + dialogue
    for t in turns:
        head = codec.encode(TOK[t["role"]])
        body = codec.encode(t["content"] + END) + codec.encode("\n")
        ids += head + body
        on = t["role"] == "assistant"
        tgt_mask += [False] * len(head) + [on] * (len(body) - 1) + [False]   # trailing "\n" not trained
    x = ids[:-1]
    y = [tok if m else IGNORE for tok, m in zip(ids[1:], tgt_mask[1:])]
    return x, y


# ------------------------------------------------------------------ eval
@torch.no_grad()
def bits_per_byte(model, codec, text: str, T: int = None, max_windows: int = None) -> dict:
    """Contiguous non-overlapping windows: every token but the first is predicted exactly once."""
    model.eval()
    T = T or model.cfg.context_len
    ids = torch.tensor(codec.encode(text))
    nll, ntok, w = 0.0, 0, 0
    for s in range(0, len(ids) - 1, T):
        chunk = ids[s:s + T + 1]
        if len(chunk) < 2:
            break
        logits, _ = model(chunk[None, :-1])
        nll += F.cross_entropy(logits[0].float(), chunk[1:], reduction="sum").item()
        ntok += len(chunk) - 1; w += 1
        if max_windows and w >= max_windows:
            break
    # bytes of the predicted tokens only (all but the first token)
    nb = n_bytes(codec.decode(ids[:ntok + 1].tolist())) - n_bytes(codec.decode(ids[:1].tolist()))
    return {"nll_per_token": nll / ntok, "ppl_per_token": math.exp(nll / ntok), "tokens": ntok,
            "bytes": nb, "bits_per_byte": nll / math.log(2) / nb}


@torch.no_grad()
def sft_loss(model, records) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, y in records:
        x, y = torch.tensor([x[:model.cfg.context_len]]), torch.tensor([y[:model.cfg.context_len]])
        logits, _ = model(x)
        tot += F.cross_entropy(logits[0].float(), y[0], ignore_index=IGNORE, reduction="sum").item()
        n += int((y != IGNORE).sum())
    return tot / max(n, 1)


@torch.no_grad()
def sample(model, codec, user: str, max_new: int = 120, system: str = DEFAULT_SYSTEM) -> str:
    prompt = f"{TOK['system']}{system}{END}\n{TOK['user']}{user}{END}\n{TOK['assistant']}"
    ids = codec.encode(prompt)
    end_id = codec.token_id(END)
    out = tx.generate_ids(model, ids, max_new, stop_ids=[end_id] if end_id is not None else None)
    txt = codec.decode(out)
    return txt.split(END)[0]


# ------------------------------------------------------------------ train loops
def _progress(step, steps, t0, minutes):
    """Schedule position in [0, 1]: by wall time when a minutes budget is set (robust to a shared,
    contended CPU), else by step."""
    if minutes:
        return min(1.0, (time.time() - t0) / (60 * minutes))
    return step / max(1, steps)


def _lr(step, steps, t0, minutes, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = _progress(step, steps, t0, minutes)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def stage1(model, train_ids, steps, B, T, lr, warmup, g, log, ckpt_dir, eval_fn, ckpt_every, eval_every,
           minutes=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    data = torch.tensor(train_ids)
    t0, seen, best = time.time(), 0, {"bpb": float("inf"), "step": 0}
    step, done = 0, not minutes and steps <= 0          # 0 steps: skip Stage 1 (e.g. with --init-from)
    while not done:
        model.train()
        ix = torch.randint(0, len(data) - T - 1, (B,), generator=g)
        x = torch.stack([data[i:i + T] for i in ix]); y = torch.stack([data[i + 1:i + T + 1] for i in ix])
        for pg in opt.param_groups:
            pg["lr"] = _lr(step, steps, t0, minutes, lr, warmup)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        seen += B * T
        done = (_progress(step + 1, steps, t0, minutes) >= 1.0) if minutes else step + 1 >= steps
        rec = {"stage": 1, "step": step + 1, "loss": round(loss.item(), 4), "tokens": seen,
               "lr": round(opt.param_groups[0]["lr"], 6), "elapsed_s": round(time.time() - t0, 1)}
        if (step + 1) % eval_every == 0 or done:
            rec["val"] = eval_fn()
            if rec["val"]["bits_per_byte"] < best["bpb"]:     # keep the best held-out checkpoint
                best.update(bpb=rec["val"]["bits_per_byte"], step=step + 1)
                tx.save(model, os.path.join(ckpt_dir, "stage1_best.pt"))
        if step % 25 == 0 or "val" in rec:
            log(rec)
        if (step + 1) % ckpt_every == 0 or done:
            tx.save(model, os.path.join(ckpt_dir, "stage1_last.pt"))
        step += 1
    best["steps_run"] = step
    return seen, time.time() - t0, best


def stage2(model, records, steps, B, lr, warmup, g, log, ckpt_dir, eval_fn, eval_every, minutes=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    T = model.cfg.context_len
    t0, seen, best2 = time.time(), 0, {"nll": float("inf"), "step": 0}
    step, done = 0, False
    while not done:
        model.train()
        idx = torch.randint(0, len(records), (B,), generator=g).tolist()
        batch = [records[i] for i in idx]
        L = min(T, max(len(x) for x, _ in batch))
        X = torch.zeros(B, L, dtype=torch.long); Y = torch.full((B, L), IGNORE, dtype=torch.long)
        for j, (x, y) in enumerate(batch):
            x, y = x[:L], y[:L]
            X[j, :len(x)] = torch.tensor(x); Y[j, :len(y)] = torch.tensor(y)
        for pg in opt.param_groups:
            pg["lr"] = _lr(step, steps, t0, minutes, lr, warmup)
        logits, _ = model(X)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), Y.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        seen += int((Y != IGNORE).sum())
        done = (_progress(step + 1, steps, t0, minutes) >= 1.0) if minutes else step + 1 >= steps
        rec = {"stage": 2, "step": step + 1, "loss": round(loss.item(), 4), "assistant_tokens": seen,
               "elapsed_s": round(time.time() - t0, 1)}
        if (step + 1) % eval_every == 0 or done:
            rec["val"] = eval_fn()
            v = rec["val"].get("sft_val_nll_per_token", float("inf"))
            if v < best2["nll"]:                           # keep the best held-out SFT checkpoint
                best2.update(nll=v, step=step + 1)
                tx.save(model, os.path.join(ckpt_dir, "stage2_best.pt"))
        if step % 25 == 0 or "val" in rec:
            log(rec)
        step += 1
    log({"event": "stage2_best", **best2})
    tx.save(model, os.path.join(ckpt_dir, "stage2_last.pt"))
    if best2["step"] and best2["step"] != step:            # finish on the best held-out SFT weights
        model.load_state_dict(tx.load(os.path.join(ckpt_dir, "stage2_best.pt")).state_dict())
    return seen, time.time() - t0, step, best2


def versions() -> dict:
    v = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
         "threads": torch.get_num_threads(), "cpu_count": os.cpu_count()}
    try:
        import tokenizers
        v["tokenizers"] = tokenizers.__version__
    except ImportError:
        pass
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--context-len", type=int, default=512)
    ap.add_argument("--tokenizer", choices=("byte", "bpe"), default="byte")
    ap.add_argument("--tokenizer-path", default="")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--seed", type=int, default=567885)
    ap.add_argument("--stage1-steps", type=int, default=2000)
    ap.add_argument("--stage2-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sft-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sft-lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--eval-windows", type=int, default=0, help="0 = whole held-out split")
    ap.add_argument("--corpus", default=os.path.join(CORPUS, "combined.txt"))
    ap.add_argument("--conversations", default=None, help="default data/corpus/conversations.jsonl")
    ap.add_argument("--stage1-minutes", type=float, default=0.0, help="wall-time budget; overrides steps")
    ap.add_argument("--stage2-minutes", type=float, default=0.0)
    ap.add_argument("--sft-eval-every", type=int, default=0)
    ap.add_argument("--init-from", default="", help="start from this checkpoint's weights")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); random.seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    run_dir = os.path.join(a.out, a.run); os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "train_log.jsonl"), "a")

    def log(rec):
        rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        logf.write(json.dumps(rec) + "\n"); logf.flush(); print(json.dumps(rec), flush=True)

    text = open(a.corpus, encoding="utf-8").read()
    train_text, val_text = split_corpus(text)
    if not val_text:
        raise SystemExit("corpus too small for a held-out split (needs >= 20 blocks of ~2 KB)")
    codec = get_codec(kind=a.tokenizer, path=a.tokenizer_path or None)
    over = {"context_len": a.context_len, "dropout": a.dropout, "tokenizer": a.tokenizer, "vocab_size": codec.vocab_size}
    if a.tokenizer == "bpe":
        over["tokenizer_path"] = os.path.abspath(a.tokenizer_path)
    model = tx.build(a.size, seed=a.seed, **over)
    if a.init_from:                                   # continue from a saved Stage 1 checkpoint
        src = tx.load(a.init_from)
        if src.cfg.to_dict() != model.cfg.to_dict():
            raise SystemExit("--init-from config differs from --size/--context-len/--tokenizer")
        model.load_state_dict(src.state_dict())
    train_ids = codec.encode(train_text)
    dialogues = sft_dialogues(train_text, conv=a.conversations)
    recs = [encode_masked(d, codec) for d in dialogues]
    sft_val = [r for i, r in enumerate(recs) if i % 10 == 9]
    sft_train = [r for i, r in enumerate(recs) if i % 10 != 9]
    ew = a.eval_windows or None
    eval1 = lambda: {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in bits_per_byte(model, codec, val_text, max_windows=ew).items()}
    eval2 = lambda: {"sft_val_nll_per_token": round(sft_loss(model, sft_val), 4), **eval1()}
    summary = {"run": a.run, "command": "python -m artemis_dense.train " + " ".join(argv or sys.argv[1:]),
               "args": vars(a), "seed": a.seed, "versions": versions(), "model_card": tx.model_card(model),
               "corpus": {"sha256_combined": __import__("hashlib").sha256(text.encode()).hexdigest(),
                          "train_bytes": n_bytes(train_text), "val_bytes": n_bytes(val_text),
                          "train_tokens": len(train_ids), "bytes_per_token": round(n_bytes(train_text) / len(train_ids), 3),
                          "sft_records_train": len(sft_train), "sft_records_val": len(sft_val)},
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    log({"event": "start", **summary})
    summary["before"] = eval2()
    summary["samples_before"] = {p: sample(model, codec, p) for p in SAMPLE_PROMPTS}
    log({"event": "before", "val": summary["before"]})
    t1_tokens, t1_s, best = stage1(model, train_ids, a.stage1_steps, a.batch, a.context_len, a.lr, a.warmup, g, log,
                             run_dir, eval1, a.ckpt_every, a.eval_every, a.stage1_minutes)
    if best["step"] and best["step"] != best["steps_run"]:   # Stage 2 starts from the best held-out weights
        model.load_state_dict(tx.load(os.path.join(run_dir, "stage1_best.pt")).state_dict())
    summary["stage1_best"] = best
    log({"event": "stage1_best", **best})
    summary["after_stage1"] = eval2()
    summary["samples_after_stage1"] = {p: sample(model, codec, p) for p in SAMPLE_PROMPTS}
    t2_tokens, t2_s, t2_steps, best2 = stage2(model, sft_train, a.stage2_steps, a.sft_batch, a.sft_lr, min(a.warmup, 30),
                                       g, log, run_dir, eval2, a.sft_eval_every or max(50, a.stage2_steps // 4),
                                       a.stage2_minutes)
    summary["after_stage2"] = eval2()
    summary["samples_after_stage2"] = {p: sample(model, codec, p) for p in SAMPLE_PROMPTS}
    summary["stage1"] = {"steps": best["steps_run"], "tokens": t1_tokens, "wall_s": round(t1_s, 1),
                         "tokens_per_s": round(t1_tokens / max(t1_s, 1e-9), 1),
                         "epochs": round(t1_tokens / len(train_ids), 2)}
    summary["stage2_best"] = best2
    summary["stage2"] = {"steps": t2_steps, "assistant_tokens": t2_tokens, "wall_s": round(t2_s, 1)}
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    model.trained_info = {"run": a.run, "stage1_steps": best["steps_run"], "stage1_best_step": best["step"],
                          "stage2_steps": t2_steps, "stage2_best_step": best2["step"],
                          "val_bits_per_byte": summary["after_stage2"]["bits_per_byte"]}
    tx.save(model, os.path.join(run_dir, "stage2_final.pt"))
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    log({"event": "done", "after_stage2": summary["after_stage2"], "stage1": summary["stage1"],
         "stage2": summary["stage2"]})
    return summary


if __name__ == "__main__":
    main()
