"""Stage 6: tokenization-free byte-level training loop on CPU (PLAN.md).

ArtemisModel (model.py) is trained with next_byte_loss on raw UTF-8 bytes of
public-domain Project Gutenberg text; no tokenizer. Reports held-out
bits-per-byte (bpb) periodically, a unigram baseline (byte distribution
measured on train, evaluated on held-out), and temperature/top-k samples.

Data (pick one):
  --corpus DIR   a corpus built by bytelm/build_corpus.py: DIR/train/*.txt for
                 training and DIR/<--heldout-split>/*.txt (default "val") as
                 held-out text (book- and author-level split, see
                 bytelm/corpus/MANIFEST.md). Default DIR: $ARTEMIS_CORPUS or
                 /workspace/artemis-corpus-bench/corpus.
  --text FILE    one plain-text file; the last --heldout-frac of its bytes is
                 the held-out split (contiguous, no overlap with train).
Corpus text is never written into the repo; only metrics go to runs/.

Outputs (runs/stage6/<run-name>/): curve.csv and curve.jsonl (one row per eval:
step, elapsed_s, train loss, held-out bpb, unigram bpb), samples.txt,
summary.json. The model checkpoint goes to runs/stage6_<run-name>.pt
(git-ignored).
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from model import ArtemisModel, next_byte_loss, bits_per_byte
from moe import expert_load_entropy

DEFAULT_CORPUS = os.environ.get("ARTEMIS_CORPUS", "/workspace/artemis-corpus-bench/corpus")


# ---------------------------------------------------------------- data
def load_split_dir(d: Path) -> bytes:
    files = sorted(Path(d).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files in {d}")
    return b"".join(f.read_bytes() for f in files)


def load_data(corpus: Optional[str], text: Optional[str], heldout_split: str = "val",
              heldout_frac: float = 0.1):
    """Return (train_bytes, heldout_bytes, description)."""
    if text:
        raw = Path(text).read_bytes()
        cut = int(len(raw) * (1.0 - heldout_frac))
        if cut <= 0 or cut >= len(raw):
            raise ValueError("heldout_frac leaves an empty split")
        return raw[:cut], raw[cut:], f"text={text} (last {heldout_frac:.0%} held out)"
    root = Path(corpus or DEFAULT_CORPUS)
    return (load_split_dir(root / "train"), load_split_dir(root / heldout_split),
            f"corpus={root} train/ vs {heldout_split}/")


def to_tensor(b: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(b), dtype=torch.uint8).clone()


def random_windows(data: torch.Tensor, batch: int, length: int, gen: torch.Generator) -> torch.Tensor:
    """[batch, length] random contiguous windows (int64)."""
    if data.numel() < length:
        raise ValueError(f"data has {data.numel()} bytes, need at least {length}")
    starts = torch.randint(0, data.numel() - length + 1, (batch,), generator=gen)
    idx = starts[:, None] + torch.arange(length)[None, :]
    return data[idx].long()


def eval_windows(data: torch.Tensor, length: int, max_windows: int) -> torch.Tensor:
    """Fixed, evenly spaced, non-overlapping held-out windows [N, length] (deterministic)."""
    n_avail = data.numel() // length
    if n_avail == 0:
        raise ValueError(f"held-out data has {data.numel()} bytes, need at least {length}")
    n = min(n_avail, max_windows)
    stride = n_avail // n
    starts = torch.arange(n) * stride * length
    return data[starts[:, None] + torch.arange(length)[None, :]].long()


# ---------------------------------------------------------------- metrics
def byte_counts(data) -> torch.Tensor:
    t = data if isinstance(data, torch.Tensor) else to_tensor(bytes(data))
    return torch.bincount(t.long().flatten(), minlength=256).double()


def unigram_bpb(train, heldout, smoothing: float = 1.0) -> float:
    """Cross-entropy (bits/byte) of a unigram byte model fit on `train` (counts +
    `smoothing` per byte value, i.e. Laplace) evaluated on `heldout`. With
    smoothing=0 and heldout == train this is the byte-distribution entropy."""
    p = byte_counts(train) + smoothing
    p = p / p.sum()
    h = byte_counts(heldout)
    if smoothing == 0 and bool(((p == 0) & (h > 0)).any()):
        return float("inf")
    nz = h > 0
    return float(-(h[nz] * torch.log2(p[nz])).sum() / h.sum())


@torch.no_grad()
def evaluate(model: ArtemisModel, windows: torch.Tensor, batch: int = 64) -> dict:
    """Held-out metrics over `windows` in eval mode (no routing noise).

    bpb / loss_nats: mean next-byte cross-entropy over every predicted position.
    drop_rate: dropped (token, slot) assignments / all assignments, over all layers.
    entropy_routed[l]: normalised expert-load entropy (expert_load_entropy) of the
    router's top-k choices in layer l before capacity; entropy_kept[l]: same for
    the assignments that were actually processed. *_mean: mean over layers.
    """
    was = model.training
    model.eval()
    tot, n = 0.0, 0
    blocks = list(model.blocks)
    routed = [torch.zeros(b.moe.num_experts, dtype=torch.float64) for b in blocks]
    kept = [torch.zeros(b.moe.num_experts, dtype=torch.float64) for b in blocks]
    dropped, assigns = 0, 0
    for i in range(0, windows.size(0), batch):
        w = windows[i:i + batch]
        logits, _, d = model(w)
        k = w.size(0) * (w.size(1) - 1)
        tot += float(next_byte_loss(logits, w)) * k
        n += k
        dropped += int(d)
        for li, b in enumerate(blocks):
            st = b.moe.last_stats
            idx = st["topk_indices"]
            E = b.moe.num_experts
            routed[li] += torch.bincount(idx.reshape(-1), minlength=E).double()
            kept[li] += torch.bincount(idx[st["kept_slots"]].reshape(-1), minlength=E).double()
            assigns += idx.numel()
    model.train(was)
    loss = tot / n
    er = [expert_load_entropy(c) for c in routed]
    ek = [expert_load_entropy(c) for c in kept]
    return dict(loss_nats=loss, bpb=bits_per_byte(loss), drop_rate=dropped / assigns if assigns else 0.0,
                dropped=dropped, assignments=assigns,
                entropy_routed=er, entropy_kept=ek,
                entropy_routed_mean=sum(er) / len(er), entropy_kept_mean=sum(ek) / len(ek),
                load_routed=[c.tolist() for c in routed])


def evaluate_bpb(model: ArtemisModel, windows: torch.Tensor, batch: int = 64) -> float:
    """Mean next-byte cross-entropy over every predicted position of `windows`, in bpb."""
    return evaluate(model, windows, batch)["bpb"]


@torch.no_grad()
def sample(model: ArtemisModel, prompt: bytes, n_new: int, temperature: float = 1.0,
           top_k: Optional[int] = None, seed: int = 0, context: Optional[int] = None) -> bytes:
    """Autoregressively generate n_new bytes after `prompt`.

    temperature <= 0 means greedy; top_k keeps the k most likely bytes. Context is
    cropped to the last `context` bytes (default: the model's max_len). Uses its own
    torch.Generator(seed), so output is deterministic for a fixed seed.
    """
    was = model.training
    model.eval()
    ctx = context or model.body.max_len
    gen = torch.Generator().manual_seed(seed)
    seq = list(prompt) if prompt else [ord("\n")]
    start = len(seq)
    for _ in range(n_new):
        x = torch.tensor([seq[-ctx:]], dtype=torch.long)
        logits = model(x)[0][0, -1].double()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k is not None and 0 < top_k < logits.numel():
                kth = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = int(torch.multinomial(F.softmax(logits, -1), 1, generator=gen))
        seq.append(nxt)
    model.train(was)
    out = bytes(seq[start:])
    return (prompt + out) if prompt else out


def lr_at(step: int, frac: float, base: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Linear warmup over `warmup` steps, then cosine to min_ratio*base as frac (0..1) -> 1."""
    if step < warmup:
        return base * (step + 1) / warmup
    frac = min(max(frac, 0.0), 1.0)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * frac)))


# ---------------------------------------------------------------- training
def train(args, train_b: bytes, held_b: bytes, data_desc: str = "", log=print) -> dict:
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)
    tr, ho = to_tensor(train_b), to_tensor(held_b)
    L = args.seq_len + 1
    ev = eval_windows(ho, L, args.eval_windows)
    uni_eval = unigram_bpb(tr, ev[:, 1:])       # same target bytes as the model eval
    uni_full = unigram_bpb(tr, ho)              # whole held-out split
    cf = getattr(args, "capacity_factor", None)
    cmode = getattr(args, "capacity_mode", "sequence")
    model = ArtemisModel(args.d_model, args.n_heads, args.d_ff, args.experts, args.layers,
                         top_k=args.top_k, capacity_factor=cf, causal=True,
                         pos_encoding=args.pos, max_len=L, dispatch=args.dispatch,
                         capacity_mode=cmode)
    leaky = cf is not None and cmode == "global"
    if leaky:
        log("[stage6] WARNING: capacity_mode=global with a finite capacity factor ranks tokens over the "
            "flattened batch and leaks future bytes (check 5); results are labelled leaky")
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["step", "elapsed_s", "lr", "train_loss_nats", "train_bpb", "aux", "dropped",
              "train_aux_mean", "train_drop_rate", "heldout_bpb", "heldout_drop_rate", "heldout_entropy_routed",
              "unigram_bpb"]
    fcsv = open(out_dir / "curve.csv", "w", newline="")
    wcsv = csv.DictWriter(fcsv, fieldnames=fields); wcsv.writeheader()
    fjl = open(out_dir / "curve.jsonl", "w")
    log(f"[stage6] {data_desc} | train {len(train_b):,} B, held-out {len(held_b):,} B "
        f"(eval {ev.size(0)} windows x {args.seq_len} predicted bytes)")
    log(f"[stage6] params {n_params:,} | d={args.d_model} layers={args.layers} experts={args.experts} "
        f"top_k={args.top_k} seq={args.seq_len} batch={args.batch} | unigram bpb {uni_eval:.4f} "
        f"(eval windows) / {uni_full:.4f} (full held-out)")

    rows, t0, step, run_loss, run_n = [], time.time(), 0, 0.0, 0
    first_loss = None
    # assignments per training step = batch * L tokens * top_k * layers
    assign_per_step = args.batch * L * args.top_k * args.layers
    run_drop, total_drop, run_aux = 0, 0, 0.0
    last_ev = {}

    def do_eval(step, lr):
        nonlocal run_loss, run_n, run_drop, run_aux, last_ev
        evr = evaluate(model, ev, args.eval_batch)
        last_ev = evr
        hb = evr["bpb"]
        tl = run_loss / run_n if run_n else None   # mean train loss since the previous eval
        tdr = run_drop / (run_n * assign_per_step) if run_n else None
        row = dict(step=step, elapsed_s=round(time.time() - t0, 2), lr=lr,
                   train_loss_nats=round(tl, 5) if tl is not None else None,
                   train_bpb=round(bits_per_byte(tl), 5) if tl is not None else None,
                   aux=round(last_aux, 5), dropped=last_drop,
                   train_aux_mean=round(run_aux / run_n, 5) if run_n else None,
                   train_drop_rate=round(tdr, 6) if tdr is not None else None,
                   heldout_bpb=round(hb, 5), heldout_drop_rate=round(evr["drop_rate"], 6),
                   heldout_entropy_routed=round(evr["entropy_routed_mean"], 5),
                   unigram_bpb=round(uni_eval, 5))
        rows.append(row); wcsv.writerow(row); fcsv.flush()
        fjl.write(json.dumps(row) + "\n"); fjl.flush()
        trs = f"{row['train_bpb']:.4f}" if tl is not None else "  -   "
        log(f"step {step:6d} t={row['elapsed_s']:7.1f}s lr={lr:.2e} train {trs} bpb "
            f"| held-out {hb:.4f} bpb (unigram {uni_eval:.4f})")
        run_loss, run_n, run_drop, run_aux = 0.0, 0, 0, 0.0
        return hb

    model.train()
    last_aux, last_drop, lr = 0.0, 0, args.lr
    do_eval(0, 0.0)
    t0 = time.time()  # budget counts training (+ periodic evals), not the step-0 eval
    while True:
        elapsed = time.time() - t0
        if (args.max_steps and step >= args.max_steps) or (args.time_budget and elapsed >= args.time_budget):
            break
        frac = max(step / args.max_steps if args.max_steps else 0.0,
                   elapsed / args.time_budget if args.time_budget else 0.0)
        lr = lr_at(step, frac, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x = random_windows(tr, args.batch, L, gen)
        logits, aux, dropped = model(x)
        loss = next_byte_loss(logits, x)
        total = loss + args.aux_alpha * aux
        opt.zero_grad(set_to_none=True)
        total.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1
        lv = loss.item()
        if not math.isfinite(lv):
            raise FloatingPointError(f"non-finite loss at step {step}")
        first_loss = lv if first_loss is None else first_loss
        run_loss += lv; run_n += 1
        last_aux, last_drop = aux.item(), int(dropped)
        run_drop += last_drop; total_drop += last_drop; run_aux += last_aux
        if step % args.eval_every == 0:
            do_eval(step, lr)
    train_time = time.time() - t0
    if not rows or rows[-1]["step"] != step:
        do_eval(step, lr)
    final_bpb = rows[-1]["heldout_bpb"]

    samples = []
    for p in args.prompt:
        s = sample(model, p.encode("utf-8"), args.sample_len, args.temperature, args.sample_top_k,
                   seed=args.seed, context=args.seq_len)
        samples.append({"prompt": p, "temperature": args.temperature, "top_k": args.sample_top_k,
                        "seed": args.seed, "text": s.decode("utf-8", errors="replace")})
    with open(out_dir / "samples.txt", "w") as f:
        for s in samples:
            f.write(f"--- prompt={s['prompt']!r} T={s['temperature']} top_k={s['top_k']} seed={s['seed']}\n")
            f.write(s["text"] + "\n")
    summary = dict(run_name=args.run_name, data=data_desc, train_bytes=len(train_b), heldout_bytes=len(held_b),
                   eval_windows=int(ev.size(0)), eval_predicted_bytes=int(ev.size(0) * args.seq_len),
                   params=n_params, steps=step, train_time_s=round(train_time, 1),
                   bytes_seen=step * args.batch * args.seq_len,
                   first_train_bpb=round(bits_per_byte(first_loss), 4) if first_loss else None,
                   initial_heldout_bpb=rows[0]["heldout_bpb"], final_heldout_bpb=final_bpb,
                   best_heldout_bpb=min(r["heldout_bpb"] for r in rows),
                   final_heldout_loss_nats=round(last_ev["loss_nats"], 5),
                   capacity_factor=cf, capacity_mode=cmode, leaky=leaky,
                   train_drop_rate=round(total_drop / (step * assign_per_step), 6) if step else None,
                   heldout_drop_rate=round(last_ev["drop_rate"], 6),
                   heldout_entropy_routed=[round(e, 5) for e in last_ev["entropy_routed"]],
                   heldout_entropy_kept=[round(e, 5) for e in last_ev["entropy_kept"]],
                   heldout_entropy_routed_mean=round(last_ev["entropy_routed_mean"], 5),
                   heldout_entropy_kept_mean=round(last_ev["entropy_kept_mean"], 5),
                   heldout_load_routed=last_ev["load_routed"],
                   final_aux=round(last_aux, 5),
                   unigram_bpb_eval_windows=round(uni_eval, 5), unigram_bpb_full_heldout=round(uni_full, 5),
                   config={k: v for k, v in vars(args).items() if k not in ("prompt",)},
                   torch=torch.__version__, threads=torch.get_num_threads(), samples=samples)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fcsv.close(); fjl.close()
    if args.save_checkpoint:
        torch.save({"state_dict": model.state_dict(), "summary": summary},
                   Path(args.out_dir).parent / f"stage6_{args.run_name}.pt")
    log(f"[stage6] done: {step} steps in {train_time:.1f}s | held-out {final_bpb:.4f} bpb vs unigram "
        f"{uni_eval:.4f} | samples:")
    for s in samples:
        log(s["text"])
    summary["model"] = model
    return summary


def _cf(v: str):
    return None if str(v).lower() in ("none", "inf", "dropless") else float(v)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=None)
    p.add_argument("--text", default=None)
    p.add_argument("--heldout-split", default="val")
    p.add_argument("--heldout-frac", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--pos", default="sinusoidal", choices=["none", "sinusoidal", "learned"])
    p.add_argument("--dispatch", default="loop", choices=["loop", "vectorized"])
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--aux-alpha", type=float, default=0.01)
    p.add_argument("--capacity-factor", type=_cf, default=None,
                   help="per-expert capacity factor; 'none' (default) = dropless")
    p.add_argument("--capacity-mode", default="sequence", choices=["sequence", "global"],
                   help="sequence (default): per-sequence, position-ordered, causal-safe; "
                        "global: flattened-batch ranking (leaks future bytes, check 5)")
    p.add_argument("--max-steps", type=int, default=0, help="0 = no step limit")
    p.add_argument("--time-budget", type=float, default=900.0, help="seconds of training, 0 = none")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-windows", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--prompt", action="append", default=None)
    p.add_argument("--sample-len", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--sample-top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=567885)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out-dir", default="runs/stage6")
    p.add_argument("--run-name", default="run")
    p.add_argument("--save-checkpoint", action="store_true")
    return p


def main(argv: Sequence[str] = None):
    args = build_parser().parse_args(argv)
    if args.prompt is None:
        args.prompt = ["It was a "]
    if not (args.max_steps or args.time_budget):
        raise SystemExit("set --max-steps and/or --time-budget")
    tr, ho, desc = load_data(args.corpus, args.text, args.heldout_split, args.heldout_frac)
    s = train(args, tr, ho, desc)
    s.pop("model")
    return s


if __name__ == "__main__":
    main()
