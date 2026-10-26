"""Stage 9(c): byte-level Continuous-Language-Field (CLF) prototype (PLAN.md).

Raw bytes from `webstream.py` (HTTP Range, cached, rate limited) -> continuous vectors

    x_i = Embedding(256, d)[b_i]                       (learned byte embedding)
        + w_v * (b_i / 255)                            (optional byte/255 scalar, --value-scalar)
        + Linear(gamma(t_i))                           (Fourier features of the position t_i)

-> the stage-3/5 causal MoE blocks (ArtemisModel.body, causal attention mask, dropless
top-k routing, so no byte can influence an earlier position) -> LayerNorm -> Linear(d, 256)
-> next-byte softmax. No tokenizer.

Position modes (`pos_mode`):
  "doc"    t_i = absolute byte offset in the document / position_scale (the CLF coordinate;
           the spec model). gamma = fourier.FourierFeatures(1, position_frequencies),
           computed in float64 then cast, so fine bands keep their precision.
  "window" same encoder, but t_i = offset inside the training window / position_scale
           (control: order information only, no document coordinate).
  "none"   no positional input at all (the no-position ablation).

Held-out bits-per-byte (bpb) are measured on the last `heldout_frac` of the book body
(contiguous, never trained on) and compared with a unigram baseline fit on the train bytes.

Only metrics/logs go to runs/stage9c/; the fetched text stays in the git-ignored cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fourier import FourierFeatures
from model import ArtemisModel
from train_bytes import byte_counts, unigram_bpb, lr_at
import webstream as ws

POS_MODES = ("doc", "window", "none")
LN2 = math.log(2.0)


# ---------------------------------------------------------------- data
def fetch_document(url: str = ws.DEFAULT_URL, cache_dir: Optional[str] = "runs/stage9c/cache",
                   chunk_size: int = 131072, fetcher: Optional[ws.RangeFetcher] = None,
                   max_bytes: Optional[int] = None, log=print):
    """Read a whole document with WebByteStream (real bytes only: fallback="error").

    Returns (data: bytes, info: dict). Chunks come from the RangeFetcher disk cache when
    present, so a re-run sends no requests."""
    fetcher = fetcher or ws.RangeFetcher(cache_dir=cache_dir, log=log)
    stream = ws.WebByteStream(url, chunk_size=chunk_size, end=max_bytes, fetcher=fetcher,
                              fallback="error")
    parts, sources, statuses = [], [], []
    for r in stream:
        if r.synthetic:                      # cannot happen with fallback="error"; belt and braces
            raise ws.WebStreamError("synthetic bytes are not allowed in stage 9c")
        parts.append(r.data)
        sources.append(r.source)
        statuses.append(r.status)
    data = b"".join(parts)
    if not data:
        raise ws.WebStreamError(f"no bytes read from {url}")
    info = dict(url=url, n_bytes=len(data), chunks=len(parts), chunk_size=chunk_size,
                sources={s: sources.count(s) for s in sorted(set(sources))},
                statuses=sorted(set(statuses), key=str), network_requests=fetcher.network_requests,
                sha256=hashlib.sha256(data).hexdigest(), synthetic=False)
    return data, info


def gutenberg_body(data: bytes):
    """(start, end) byte offsets of the text between the Project Gutenberg
    '*** START OF ...' and '*** END OF ...' marker lines; (0, len) if absent."""
    s = data.find(b"*** START OF")
    start = 0
    if s >= 0:
        nl = data.find(b"\n", s)
        start = nl + 1 if nl >= 0 else s
    e = data.find(b"*** END OF", start)
    end = e if e >= 0 else len(data)
    return start, end


def split_document(data: bytes, heldout_frac: float = 0.1, body: Optional[tuple] = None):
    """Contiguous split of the body: train = first (1 - f), held-out = last f.

    Returns dict(train=(offset, bytes), heldout=(offset, bytes)) with absolute offsets
    into `data`, so the document coordinate t is the same as in the original file."""
    if not 0.0 < heldout_frac < 1.0:
        raise ValueError("heldout_frac must be in (0, 1)")
    a, b = body if body is not None else (0, len(data))
    cut = a + int((b - a) * (1.0 - heldout_frac))
    if cut <= a or cut >= b:
        raise ValueError("split leaves an empty part")
    return dict(train=(a, data[a:cut]), heldout=(cut, data[cut:b]))


def to_tensor(b: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(b), dtype=torch.uint8).clone() if b else torch.empty(0, dtype=torch.uint8)


def random_windows(data: torch.Tensor, base_offset: int, batch: int, length: int, gen: torch.Generator):
    """([batch, length] int64 ids, [batch] absolute start offsets)."""
    if data.numel() < length:
        raise ValueError(f"need at least {length} bytes, have {data.numel()}")
    starts = torch.randint(0, data.numel() - length + 1, (batch,), generator=gen)
    idx = starts[:, None] + torch.arange(length)[None, :]
    return data[idx].long(), starts + base_offset


def tiled_windows(data: torch.Tensor, base_offset: int, length: int):
    """Every non-overlapping window [N, length] in order (deterministic), and start offsets."""
    n = data.numel() // length
    if n == 0:
        raise ValueError(f"need at least {length} bytes, have {data.numel()}")
    starts = torch.arange(n) * length
    return data[starts[:, None] + torch.arange(length)[None, :]].long(), starts + base_offset


# ---------------------------------------------------------------- metrics
def bpb_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean cross-entropy in bits of `targets` under softmax(`logits`) ([..., 256] / [...])."""
    nats = F.cross_entropy(logits.reshape(-1, logits.size(-1)).double(), targets.reshape(-1).long())
    return float(nats) / LN2


def byte_entropy_bits(data) -> float:
    """Shannon entropy (bits/byte) of the empirical byte distribution of `data`."""
    c = byte_counts(data)
    p = c[c > 0] / c.sum()
    return float(-(p * torch.log2(p)).sum())


# ---------------------------------------------------------------- model
class ByteCLF(nn.Module):
    """Byte embedding (+ optional byte/255 scalar) + projected gamma(t) -> causal MoE -> 256 logits.

    forward(ids [B, T] int, offsets [B] int = absolute start offset of each row)
      -> (logits [B, T, 256], aux_loss_sum, dropped_count_sum)
    The MoE body, final LayerNorm and head are those of an ArtemisModel built with
    causal=True, pos_encoding="none" and capacity_factor=None (dropless, causal-safe)."""

    VOCAB = 256

    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 256, num_experts: int = 4,
                 num_layers: int = 2, top_k: int = 2, pos_mode: str = "doc",
                 position_frequencies: int = 20, position_scale: float = 2.0 ** 20,
                 value_scalar: bool = False, dispatch: str = "loop"):
        super().__init__()
        if pos_mode not in POS_MODES:
            raise ValueError(f"pos_mode must be one of {POS_MODES}, got {pos_mode!r}")
        if position_scale <= 0:
            raise ValueError("position_scale must be > 0")
        self.pos_mode = pos_mode
        self.position_scale = float(position_scale)
        self.lm = ArtemisModel(d_model, n_heads, d_ff, num_experts, num_layers, top_k=top_k,
                               capacity_factor=None, causal=True, pos_encoding="none",
                               dispatch=dispatch)
        if pos_mode != "none":
            self.gamma = FourierFeatures(1, position_frequencies)
            self.pos_proj = nn.Linear(self.gamma.out_dim, d_model)
            nn.init.normal_(self.pos_proj.weight, std=0.02)
            nn.init.zeros_(self.pos_proj.bias)
        else:
            self.gamma, self.pos_proj = None, None
        self.value_proj = nn.Linear(1, d_model, bias=False) if value_scalar else None

    @property
    def causal_safe(self) -> bool:
        return self.lm.causal_safe

    @property
    def blocks(self):
        return self.lm.blocks

    def positions(self, T: int, offsets: torch.Tensor) -> torch.Tensor:
        """float64 [B, T] coordinates t for the configured pos_mode."""
        rel = torch.arange(T, dtype=torch.float64, device=offsets.device)[None, :]
        if self.pos_mode == "doc":
            return (offsets.to(torch.float64)[:, None] + rel) / self.position_scale
        return rel.expand(offsets.numel(), T) / self.position_scale

    def position_features(self, T: int, offsets: torch.Tensor) -> torch.Tensor:
        """gamma(t) [B, T, out_dim] in float32 (computed in float64)."""
        t = self.positions(T, offsets)
        return self.gamma(t.unsqueeze(-1)).to(torch.float32)

    def embed(self, ids: torch.Tensor, offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ids.dim() != 2:
            raise ValueError(f"ids must have shape [B, T], got {tuple(ids.shape)}")
        if ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
            raise TypeError(f"ids must be integer byte values, got {ids.dtype}")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) > 255):
            raise ValueError("byte ids must be in 0..255")
        B, T = ids.shape
        x = self.lm.embed(ids.long())
        if self.value_proj is not None:
            x = x + self.value_proj(ids.to(torch.float32).unsqueeze(-1) / 255.0)
        if self.pos_proj is not None:
            if offsets is None:
                offsets = torch.zeros(B, dtype=torch.long, device=ids.device)
            offsets = torch.as_tensor(offsets).reshape(-1).to(ids.device)
            if offsets.numel() != B:
                raise ValueError(f"offsets must have {B} entries, got {offsets.numel()}")
            x = x + self.pos_proj(self.position_features(T, offsets))
        return x

    def forward(self, ids: torch.Tensor, offsets: Optional[torch.Tensor] = None):
        x = self.embed(ids, offsets)
        x, aux, dropped = self.lm.body(x)
        return self.lm.head(self.lm.ln_f(x)), aux, dropped


def next_byte_nats(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1))


@torch.no_grad()
def evaluate_bpb(model: ByteCLF, windows: torch.Tensor, offsets: torch.Tensor, batch: int = 64) -> float:
    """Mean next-byte bits over every predicted position (t >= 1 of each window)."""
    was = model.training
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, windows.size(0), batch):
        dev = next(model.parameters()).device
        w, o = windows[i:i + batch].to(dev), offsets[i:i + batch].to(dev)
        logits = model(w, o)[0]
        k = w.size(0) * (w.size(1) - 1)
        tot += float(F.cross_entropy(logits[:, :-1].reshape(-1, 256).double(), w[:, 1:].reshape(-1),
                                     reduction="sum"))
        n += k
    model.train(was)
    return tot / n / LN2


# ---------------------------------------------------------------- training
def train_variant(pos_mode: str, train_part, held_part, args, log=print) -> dict:
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)      # same batches for every variant
    tr_off, tr_b = train_part
    ho_off, ho_b = held_part
    tr, ho = to_tensor(tr_b), to_tensor(ho_b)
    L = args.seq_len + 1
    ev, ev_off = tiled_windows(ho, ho_off, L)
    if args.eval_windows and ev.size(0) > args.eval_windows:
        keep = torch.linspace(0, ev.size(0) - 1, args.eval_windows).round().long()
        ev, ev_off = ev[keep], ev_off[keep]
    model = ByteCLF(args.d_model, args.n_heads, args.d_ff, args.experts, args.layers, args.top_k,
                    pos_mode=pos_mode, position_frequencies=args.position_frequencies,
                    position_scale=args.position_scale, value_scalar=args.value_scalar,
                    dispatch=args.dispatch)
    device = torch.device(getattr(args, "device", "cpu") or "cpu")
    model = model.to(device)
    assert model.causal_safe
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log(f"[stage9c:{pos_mode}] params {n_params:,} | d={args.d_model} layers={args.layers} "
        f"experts={args.experts} top_k={args.top_k} seq={args.seq_len} batch={args.batch} "
        f"steps={args.steps} | eval {ev.size(0)} windows x {args.seq_len} predicted bytes")
    curve = []
    b0 = evaluate_bpb(model, ev, ev_off, args.eval_batch)
    curve.append(dict(step=0, elapsed_s=0.0, train_bpb=None, heldout_bpb=round(b0, 5)))
    log(f"[stage9c:{pos_mode}] step 0 held-out {b0:.4f} bpb")
    t0 = time.time()
    run, rn = 0.0, 0
    model.train()
    for step in range(1, args.steps + 1):
        lr = lr_at(step - 1, (step - 1) / max(1, args.steps), args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, off = random_windows(tr, tr_off, args.batch, L, gen)
        x, off = x.to(device), off.to(device)
        logits, aux, _ = model(x, off)
        loss = next_byte_nats(logits, x)
        opt.zero_grad(set_to_none=True)
        (loss + args.aux_alpha * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        run += float(loss.detach()); rn += 1
        if step % args.eval_every == 0 or step == args.steps:
            hb = evaluate_bpb(model, ev, ev_off, args.eval_batch)
            row = dict(step=step, elapsed_s=round(time.time() - t0, 2), lr=lr,
                       train_bpb=round(run / rn / LN2, 5), heldout_bpb=round(hb, 5))
            curve.append(row)
            log(f"[stage9c:{pos_mode}] step {step:5d} t={row['elapsed_s']:7.1f}s lr={lr:.2e} "
                f"train {row['train_bpb']:.4f} | held-out {hb:.4f} bpb")
            run, rn = 0.0, 0
    train_time = time.time() - t0
    final = curve[-1]["heldout_bpb"]
    # also report bpb on train-split windows (same count) to show the generalisation gap
    trw, tro = tiled_windows(tr, tr_off, L)
    keep = torch.linspace(0, trw.size(0) - 1, ev.size(0)).round().long()
    train_eval = evaluate_bpb(model, trw[keep], tro[keep], args.eval_batch)
    return dict(pos_mode=pos_mode, params=n_params, steps=args.steps, train_time_s=round(train_time, 1),
                bytes_seen=args.steps * args.batch * L, device=str(device), initial_heldout_bpb=round(b0, 5),
                final_heldout_bpb=final, best_heldout_bpb=min(r["heldout_bpb"] for r in curve),
                final_train_windows_bpb=round(train_eval, 5),
                eval_windows=int(ev.size(0)), eval_predicted_bytes=int(ev.size(0) * args.seq_len),
                curve=curve)


def build_parser():
    p = argparse.ArgumentParser(description="Stage 9(c): byte-level CLF prototype")
    p.add_argument("--url", default=ws.DEFAULT_URL)
    p.add_argument("--cache-dir", default="runs/stage9c/cache")
    p.add_argument("--chunk-size", type=int, default=131072)
    p.add_argument("--heldout-frac", type=float, default=0.1)
    p.add_argument("--variants", default="doc,none,window")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--dispatch", default="loop", choices=["loop", "vectorized"])
    p.add_argument("--position-frequencies", type=int, default=20)
    p.add_argument("--position-scale", type=float, default=2.0 ** 20)
    p.add_argument("--value-scalar", action="store_true")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--aux-alpha", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-windows", type=int, default=0, help="0 = every held-out window")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=567885)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu or cuda (cuda:N)")
    p.add_argument("--out", default="runs/stage9c/results.json")
    return p


def main(argv: Sequence[str] = None):
    args = build_parser().parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    log = lambda m: print(m, flush=True)
    t_start = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    data, info = fetch_document(args.url, args.cache_dir, args.chunk_size,
                                log=lambda m: print(m, file=sys.stderr, flush=True))
    body = gutenberg_body(data)
    parts = split_document(data, args.heldout_frac, body)
    (tr_off, tr_b), (ho_off, ho_b) = parts["train"], parts["heldout"]
    L = args.seq_len + 1
    ev, _ = tiled_windows(to_tensor(ho_b), ho_off, L)
    targets = ev[:, 1:]
    uni = dict(train_entropy_bpb=round(byte_entropy_bits(tr_b), 5),
               heldout_xent_bpb_full=round(unigram_bpb(tr_b, ho_b), 5),
               heldout_xent_bpb_eval_targets=round(unigram_bpb(tr_b, targets), 5))
    log(f"[stage9c] {info['url']} {info['n_bytes']:,} B sha256 {info['sha256'][:16]}… "
        f"({info['chunks']} chunks, sources {info['sources']}, {info['network_requests']} requests)")
    log(f"[stage9c] body {body[0]:,}-{body[1]:,} | train {len(tr_b):,} B @ {tr_off:,} | "
        f"held-out {len(ho_b):,} B @ {ho_off:,}")
    log(f"[stage9c] unigram: train entropy {uni['train_entropy_bpb']:.4f} bpb | held-out cross-entropy "
        f"{uni['heldout_xent_bpb_full']:.4f} (full) / {uni['heldout_xent_bpb_eval_targets']:.4f} (eval targets)")
    results = {}
    for v in [s.strip() for s in args.variants.split(",") if s.strip()]:
        results[v] = train_variant(v, parts["train"], parts["heldout"], args, log)
    out = dict(stage="9c", started=t_start, finished=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               data=dict(info, body=list(body), train_offset=tr_off, train_bytes=len(tr_b),
                         heldout_offset=ho_off, heldout_bytes=len(ho_b),
                         heldout_frac=args.heldout_frac),
               unigram=uni, variants=results, config=vars(args), torch=torch.__version__,
               threads=torch.get_num_threads())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    log("[stage9c] summary (held-out bpb, lower is better):")
    log(f"  unigram (train dist on held-out) {uni['heldout_xent_bpb_eval_targets']:.4f}")
    for v, r in results.items():
        log(f"  {v:7s} final {r['final_heldout_bpb']:.4f} best {r['best_heldout_bpb']:.4f} "
            f"(train windows {r['final_train_windows_bpb']:.4f}, {r['train_time_s']}s)")
    return out


if __name__ == "__main__":
    main()
