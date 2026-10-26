"""Time-budgeted Stage-1 trainer that switches the 20 accelerated-learning methods on by flag.

  python -m artemis_dense.accel.trainer --preset baseline --minutes 15 --run ab_base --threads 2
  python -m artemis_dense.accel.trainer --preset fast     --minutes 15 --run ab_fast --threads 2
  python -m artemis_dense.accel.trainer --preset fast --set lora=true --set kd_alpha=0.3 ...

The wall-clock budget covers ALL work a preset chooses to do (tokenizer training, filtering,
dedup, synthetic problems, size probes, autotune, LR range test, growth, self-play); only the
periodic validation passes are paused out of it. The held-out split is identical for every
preset (same split as artemis_dense/train.py) and is never filtered. Metric: bits-per-byte.
CPU only unless gpu_approved; no network, no servers, no downloads.
"""
import argparse, copy, dataclasses, hashlib, json, math, os, platform, random, sys, time

import torch
import torch.nn.functional as F

from ..config import DenseConfig
from ..model import ArtemisDense
from . import data as D
from . import device as dev
from .config import AccelConfig, PRESETS
from .distill import kd_loss
from .growth import grow, rebuild_optimizer, shrink_for_growth
from .lora import apply_lora, lora_parameters
from .replay import MistakeReplay, per_sample_loss
from .schedule import choose_size, lr_at, lr_range_test
from .selfplay import best_of_n

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PKG)
CORPUS = os.path.join(PKG, "data", "corpus", "combined.txt")
_DC_FIELDS = {f.name for f in dataclasses.fields(DenseConfig)}


def build_model(size: str, vocab: int, context_len: int, seed: int, tokenizer="byte", tok_path="", **over):
    from ..config import SCALES
    kw = {**SCALES[size], "vocab_size": vocab, "context_len": context_len, **over}
    if "tokenizer" in _DC_FIELDS:                  # field added by the BPE integration
        kw.update(tokenizer=tokenizer, tokenizer_path=tok_path or ("<path>" if tokenizer == "bpe" else ""))
    torch.manual_seed(seed)
    return ArtemisDense(DenseConfig(**kw))


@torch.no_grad()
def bits_per_byte(model, codec, text: str, T: int, max_windows: int = 0) -> dict:
    model.eval()
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
    nb = len(codec.decode(ids[:ntok + 1].tolist()).encode()) if max_windows else len(text.encode())
    return {"bits_per_byte": round(nll / math.log(2) / nb, 4), "nll_per_token": round(nll / ntok, 4),
            "tokens": ntok, "bytes": nb}


def versions():
    v = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
         "threads": torch.get_num_threads(), "cpu_count": os.cpu_count()}
    try:
        import tokenizers; v["tokenizers"] = tokenizers.__version__
    except ImportError:
        pass
    return v


class Clock:
    """Budget clock that can be paused (validation passes don't consume training budget)."""

    def __init__(self, seconds):
        self.budget, self.t0, self.paused = seconds, time.perf_counter(), 0.0

    def used(self):
        return time.perf_counter() - self.t0 - self.paused

    def frac(self):
        return min(1.0, self.used() / self.budget)

    def left(self):
        return self.budget - self.used()

    def pause(self):
        c = self

        class _P:
            def __enter__(s): s.t = time.perf_counter()
            def __exit__(s, *a): c.paused += time.perf_counter() - s.t
        return _P()


def run(a: AccelConfig, minutes: float, run_name: str, out_root: str = None, corpus: str = CORPUS,
        threads: int = 0, eval_every_s: float = 120.0, eval_windows: int = 0, log_fn=print) -> dict:
    if threads:
        torch.set_num_threads(threads)
    random.seed(a.seed); torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    out = os.path.join(out_root or os.path.join(ROOT, "checkpoints", "accel"), run_name)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train_log.jsonl"), "w")

    def log(rec):
        rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        logf.write(json.dumps(rec) + "\n"); logf.flush(); log_fn(json.dumps(rec))

    # ---- approvals first (1, 5, 9, 10-generate) ----------------------------------------------
    device = dev.resolve_device(a.device, a.gpu_approved)
    if a.ddp and not a.ddp_approved:
        raise dev.ApprovalRequired("ddp=True needs ddp_approved (multi-GPU rental)")
    if a.init_pretrained:
        from .pretrained import check_approval
        check_approval(a.pretrained_approved, a.license_ack)
    if a.teacher_generate:
        raise dev.ApprovalRequired("teacher_generate is planned and needs approval (teacher weights/API)")

    clock = Clock(minutes * 60)
    text = open(corpus, encoding="utf-8").read()
    train_text, val_text = D.split_corpus(text)
    status = {}
    summary = {"run": run_name, "accel": a.to_dict(), "minutes": minutes, "versions": versions(),
               "command": "python -m artemis_dense.accel.trainer " + " ".join(sys.argv[1:]),
               "corpus": {"path": os.path.relpath(corpus, ROOT) if corpus.startswith(ROOT) else corpus,
                          "sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "train_bytes": len(train_text.encode()), "val_bytes": len(val_text.encode())},
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    # ---- data methods (13, 14, 15, 16) ---------------------------------------------------------
    segmented = a.quality_filter or a.dedup or a.synthetic or a.curriculum or a.pack
    if segmented:
        docs = D.segment(train_text); n0 = len(docs)
        if a.quality_filter:
            docs = D.quality_filter(docs); status["13_quality_filter"] = f"kept {len(docs)}/{n0}"
        if a.dedup:
            n1 = len(docs); docs = D.dedup(docs, a.minhash_threshold)
            status["14_dedup"] = f"kept {len(docs)}/{n1}"
        if a.synthetic:
            syn = D.synthetic_problems(a.synthetic_math, a.synthetic_physics, seed=a.seed % 1000 + 1, exclude=val_text)
            docs += syn; status["15_synthetic"] = f"+{len(syn)} verified problems (val-overlap removed)"
        if a.curriculum:
            docs = D.curriculum_order(docs); status["16_curriculum"] = "easy->hard, sqrt competence pacing"
        else:
            random.Random(a.seed).shuffle(docs)
        train_body = "".join(docs)
    else:
        docs, train_body = None, train_text

    # ---- 6 tokenizer -----------------------------------------------------------------------------
    tok_path = a.tokenizer_path
    if a.tokenizer == "bpe" and not tok_path:
        tok_path = D.train_bpe(train_body, os.path.join(out, "bpe"), vocab_size=4096)
    codec = D.get_codec(a.tokenizer, tok_path)
    status["6_tokenizer"] = f"{a.tokenizer} vocab={codec.vocab_size}"
    if docs is not None:
        docs_ids = [codec.encode(d) for d in docs]
        stream = [t for d in docs_ids for t in d]
    else:
        stream = codec.encode(train_body)
    data = torch.tensor(stream, dtype=torch.long)
    summary["corpus"].update(train_tokens=len(stream), bytes_per_token=round(len(train_body.encode()) / len(stream), 3))
    T_max = a.context_len

    def crop_batch(T, B, frac):
        hi = len(data) - T - 1
        if a.curriculum:
            hi = max(T + 1, int(hi * D.competence(frac, a.curriculum_start, a.curriculum_full_at)))
        ix = torch.randint(0, hi, (B,), generator=g)
        return torch.stack([data[i:i + T] for i in ix]), torch.stack([data[i + 1:i + T + 1] for i in ix])

    packs, perm_state = {}, {}

    def pack_batch(T, B, frac):                 # 7: whole-document packing, rows without replacement
        if T not in packs:
            packs[T] = D.pack_documents(docs_ids, T, sep=codec.encode("\n"))
            perm_state[T] = []
        rows = packs[T]
        n = len(rows)
        if a.curriculum:
            n = max(B, int(n * D.competence(frac, a.curriculum_start, a.curriculum_full_at)))
        q = perm_state[T]
        while len(q) < B:
            q.extend(torch.randperm(n, generator=g).tolist())
        idx, perm_state[T] = q[:B], q[B:]
        r = rows[idx]
        return r[:, :-1], r[:, 1:]

    batch_fn = pack_batch if a.pack else crop_batch

    # ---- 17 scaling-law sizing ---------------------------------------------------------------
    size = a.size
    if a.scaling_size:
        tps = {}
        for cand in ("tiny", "small"):
            m = build_model(cand, codec.vocab_size, T_max, a.seed, a.tokenizer, tok_path)
            o = torch.optim.AdamW(m.parameters(), lr=1e-3)
            x, y = crop_batch(T_max, a.batch, 1.0)

            def st(m=m, o=o, x=x, y=y):
                _, l = m(x, y); o.zero_grad(); l.backward(); o.step()
            tps[cand] = a.batch * T_max / dev.time_steps(st, n=2)
        pick = choose_size(tps, clock.left())
        size = pick["choice"]; summary["scaling_size"] = pick
        status["17_scaling_size"] = f"chose {size} (est. tokens/s {({k: round(v) for k, v in tps.items()})})"

    # ---- model (+ 9 pretrained, 11 growth start, 12 LoRA) ---------------------------------------
    target_layers = None
    if a.init_pretrained:
        from .pretrained import load_local
        model = load_local(a.init_pretrained, a.pretrained_approved, a.license_ack, T_max)
    else:
        from ..config import SCALES
        over = {}
        if a.grow:
            target_layers = SCALES[size]["n_layers"]
            over["n_layers"] = shrink_for_growth(dict(SCALES[size]), a.grow_start_layers)["n_layers"]
        model = build_model(size, codec.vocab_size, T_max, a.seed, a.tokenizer, tok_path, **over)
    model.to(device)
    teacher = None
    if a.distill and a.teacher_path:
        from ..engines import transformer as tx
        teacher = tx.load(a.teacher_path).to(device).eval()
        if teacher.cfg.vocab_size != codec.vocab_size:
            raise ValueError("teacher vocab must match the student's tokenizer")
        status["10_distill"] = f"KL to local teacher {os.path.basename(a.teacher_path)}"
    if a.lora:
        n = apply_lora(model, a.lora_r, a.lora_alpha, tuple(a.lora_targets))
        status["12_lora"] = f"{n} adapters, {sum(p.numel() for p in lora_parameters(model))} trainable params"
    params = lambda: [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # ---- 2/3 autotune precision + attention, 4 compile --------------------------------------------
    prec, attn = a.precision, a.attention
    if "auto" in (prec, attn):
        x, y = batch_fn(T_max, a.batch, 1.0)

        def make_step(p, at):
            m = copy.deepcopy(model); o = torch.optim.AdamW(m.parameters(), lr=1e-3)

            def st():
                with dev.autocast(p, device), dev.attention_ctx(at):
                    _, l = m(x, y)
                o.zero_grad(); l.backward(); o.step()
            return st
        tune = dev.autotune(make_step, ("fp32", "bf16") if prec == "auto" else (prec,),
                            ("default", "math", "flash") if attn == "auto" else (attn,))
        prec, attn = tune["precision"], tune["attention"]
        summary["autotune"] = tune
    status["2_precision"], status["3_attention"] = prec, attn
    fwd, cstat = dev.maybe_compile(model, a.compile, probe=(lambda cm: cm(batch_fn(T_max, 2, 1.0)[0])) if a.compile else None)
    status["4_compile"] = cstat

    # ---- 18 LR range test -------------------------------------------------------------------------
    peak = a.lr
    if a.lr_find:
        rt = lr_range_test(model, lambda: batch_fn(T_max, a.batch, 0.0), steps=30)
        peak = rt["suggested_lr"]; summary["lr_range_test"] = rt
        status["18_lr"] = f"range test -> peak {peak:.2e}, warmup+cosine"
    else:
        status["18_lr"] = f"fixed peak {peak:.0e}, warmup+cosine"
    warm_frac = a.warmup_frac

    replay = MistakeReplay(a.replay_buffer, a.replay_frac, a.mistakes_dir, codec) if a.mistake_replay else None
    grow_points = []
    if a.grow and target_layers and model.num_layers < target_layers:   # spread the adds over grow_at[0]..grow_at[-1]
        n_add, lo, hi = target_layers - model.num_layers, min(a.grow_at), max(a.grow_at)
        grow_points = [lo + (hi - lo) * i / max(1, n_add - 1) for i in range(n_add)]
    sp_points = sorted(a.self_play_at) if a.self_play else []
    sp_problems = []
    if a.self_play:
        from ..engines.math import problem_set
        sp_problems = [p for p in problem_set(60, seed=a.seed % 1000 + 99) if p["verified"] and p["q"] not in val_text]
        sp_problems = sp_problems[:a.self_play_prompts]
    summary["before"] = bits_per_byte(model, codec, val_text, T_max, eval_windows)
    log({"event": "start", "status": status, "params": model.num_params(), "before": summary["before"],
         "prep_s": round(clock.used(), 1)})

    step, seen_tokens, seen_bytes_est, last_eval, evals = 0, 0, 0.0, clock.used(), []
    bpt = summary["corpus"]["bytes_per_token"]
    while clock.left() > 0:
        frac = clock.frac()
        # 11 growth events
        while grow_points and frac >= grow_points[0]:
            grow_points.pop(0)
            if target_layers and model.num_layers < target_layers:
                ev = grow(model, 1, a.grow_ffn_to if not grow_points else 0, generator=g)
                opt = rebuild_optimizer(opt, model)
                fwd, _ = dev.maybe_compile(model, a.compile and cstat == "compiled")
                log({"event": "grow", "step": step, **ev})
        # 20 self-play events
        while sp_points and frac >= sp_points[0]:
            sp_points.pop(0)
            r = best_of_n(model, codec, sp_problems, n=a.self_play_n, seed=step)
            if r["kept"]:
                extra = torch.tensor([t for k in r["kept"] for t in codec.encode(k)])
                data = torch.cat([data, extra])
                if a.pack:
                    docs_ids.extend(codec.encode(k) for k in r["kept"]); packs.clear()
            log({"event": "self_play", "step": step, **{k: v for k, v in r.items() if k != "kept"},
                 "kept": len(r["kept"])})
        T = D.seq_len_at(frac, T_max, a.seq_min, a.seq_warmup_frac) if a.seq_warmup else T_max
        x, y = batch_fn(T, a.batch, frac)
        if replay is not None:
            x, y, _ = replay.mix(x, y, g)
        x, y = x.to(device), y.to(device)
        lr = lr_at(frac, peak, warm_frac) if warm_frac > 0 else \
            (peak * (step + 1) / a.warmup_steps if step < a.warmup_steps else lr_at(frac, peak, 0.0))
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train()
        with dev.autocast(prec, device), dev.attention_ctx(attn):
            logits, _ = fwd(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
            if teacher is not None:
                with torch.no_grad():
                    tl, _ = teacher(x)
                loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kd_loss(logits, tl, a.kd_temp)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params(), 1.0); opt.step()
        if replay is not None:
            replay.observe(x, y, per_sample_loss(logits, y))
        step += 1; seen_tokens += x.numel()
        if step % 25 == 0:
            log({"step": step, "loss": round(loss.item(), 4), "T": T, "lr": round(lr, 6),
                 "layers": model.num_layers, "tokens": seen_tokens, "used_s": round(clock.used(), 1)})
        if clock.used() - last_eval >= eval_every_s:
            with clock.pause():
                ev = {"step": step, "used_s": round(clock.used(), 1), "train_tokens": seen_tokens,
                      **bits_per_byte(model, codec, val_text, T_max, eval_windows or 20)}
                evals.append(ev); log({"event": "eval", **ev})
            last_eval = clock.used()

    summary["final"] = bits_per_byte(model, codec, val_text, T_max, 0)
    summary.update(status=status, steps=step, train_tokens_seen=seen_tokens,
                   train_bytes_seen_est=round(seen_tokens * bpt), wall_budget_s=minutes * 60,
                   train_used_s=round(clock.used(), 1), eval_paused_s=round(clock.paused, 1),
                   evals=evals, final_params=model.num_params(), final_layers=model.num_layers,
                   finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    model.trained_info = {"run": run_name, "accel_preset_flags": a.to_dict(), "val_bits_per_byte": summary["final"]["bits_per_byte"]}
    torch.save({"cfg": model.cfg.to_dict(), "model": model.state_dict(), "trained_info": model.trained_info},
               os.path.join(out, "final.pt"))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, default=str)
    log({"event": "done", "final": summary["final"], "steps": step, "tokens": seen_tokens})
    return summary


def _parse_set(items):
    over = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            over[k] = json.loads(v)
        except json.JSONDecodeError:
            over[k] = v
    return over


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast", choices=sorted(PRESETS))
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=567885)
    ap.add_argument("--eval-every-s", type=float, default=120)
    ap.add_argument("--set", action="append", help="override any AccelConfig field: key=json")
    ns = ap.parse_args(argv)
    cfg = AccelConfig.preset(ns.preset, seed=ns.seed, **_parse_set(ns.set))
    return run(cfg, ns.minutes, ns.run, ns.out, ns.corpus, ns.threads, ns.eval_every_s)


if __name__ == "__main__":
    main()
