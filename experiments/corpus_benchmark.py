"""Byte-level language benchmark on a local train/val/test corpus (one file = one document).

Based on the user-supplied artemis_corpus_benchmark.py; `--model dense` is that script's model and
protocol unchanged. `--model moe --position {none,learned,sinusoidal,fourier}` swaps in the repo's
causal byte-level MoE (byte_model.ArtemisByteLM) under the identical data order, budget, evaluation,
and checkpoint selection. The MoE aux loss is added to the training loss only; every reported
bits-per-byte value is cross-entropy alone.
"""
import argparse
import sys
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from byte_model import ArtemisByteLM  # noqa: E402


class ByteModel(nn.Module):
    """Two-layer causal byte transformer: 37,888 parameters at context=128."""

    def __init__(self, context=128):
        super().__init__()
        self.byte = nn.Embedding(256, 32)
        self.position = nn.Embedding(context, 32)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=32,
                nhead=2,
                dim_feedforward=64,
                dropout=0.0,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 256)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.byte(tokens) + self.position(positions)

        future_mask = torch.ones(
            length, length, dtype=torch.bool, device=tokens.device
        ).triu(1)

        for block in self.blocks:
            x = block(x, src_mask=future_mask)

        return self.head(self.norm(x))


def load_corpus(root):
    """One file is one document. Reject exact cross-split duplicates."""
    documents, manifest = {}, {}
    seen = {}
    counts = Counter()

    for split in ("train", "val", "test"):
        documents[split], manifest[split] = [], []

        for path in sorted((root / split).rglob("*")):
            if not path.is_file():
                continue

            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()

            if digest in seen and seen[digest] != split:
                raise ValueError(f"Cross-split duplicate: {path}")
            seen[digest] = split

            documents[split].append(raw)
            manifest[split].append({
                "path": str(path.relative_to(root)),
                "bytes": len(raw),
                "sha256": digest,
            })

            if split == "train":
                counts.update(raw)

        targets = sum(max(0, len(raw) - 1)
                      for raw in documents[split])
        if targets == 0:
            raise ValueError(f"No next-byte targets in {split}")

    return documents, manifest, counts


def windows(documents, context):
    """Chunk-reset context; score every byte except each file's first once."""
    for raw in documents:
        for start in range(0, len(raw) - 1, context):
            yield raw[start:start + context + 1]


def batches(documents, context, batch_size, device, chunks=None):
    """Batches of chunk-reset windows; `chunks` overrides the document-ordered windows."""
    pending = []

    def pack(chunks):
        length = max(len(chunk) - 1 for chunk in chunks)
        x = torch.zeros(
            len(chunks), length, dtype=torch.long, device=device
        )
        y = torch.full_like(x, -100)

        for row, chunk in enumerate(chunks):
            values = torch.tensor(
                list(chunk), dtype=torch.long, device=device
            )
            x[row, :len(chunk) - 1] = values[:-1]
            y[row, :len(chunk) - 1] = values[1:]

        # Padding trails real inputs, so causal attention cannot expose it
        # to real target positions. Padded targets are excluded from loss.
        return x, y

    for chunk in (windows(documents, context) if chunks is None else chunks):
        pending.append(chunk)
        if len(pending) == batch_size:
            yield pack(pending)
            pending = []

    if pending:
        yield pack(pending)


def build_model(args):
    if args.model == "dense":
        return ByteModel(args.context)
    return ArtemisByteLM(args.position, d_model=32, n_heads=2, d_ff=64,
                         num_experts=args.experts, num_layers=2, top_k=2,
                         capacity_factor=args.capacity_factor,
                         capacity_mode=args.capacity_mode, context=args.context)


def run(model, x):
    """Returns (logits, aux_loss, dropped_assignments) for either model type."""
    out = model(x)
    if isinstance(out, tuple):
        return out
    return out, out.new_zeros(()), 0


def moe_layers(model):
    body = getattr(model, "body", None)
    return [b.moe for b in body.blocks] if body is not None else []


def parameter_counts(model, args):
    total = sum(p.numel() for p in model.parameters())
    layers = moe_layers(model)
    inactive = sum((m.num_experts - m.top_k) * sum(p.numel() for p in m.experts[0].parameters())
                   for m in layers)
    return {"total": total, "active_per_byte": total - inactive,
            "active_definition": (
                "active_per_byte = total parameters minus the parameters of the (num_experts - top_k) "
                "experts not selected for a byte, in every MoE layer. All shared parameters count as "
                "active: the full 256-row byte-embedding table, the position module, attention, layer "
                "norms, router gates, and the output head. Dense model: active_per_byte = total.")}


class RoutingStats:
    """Accumulates router top-k choices over real (non-padding) bytes, per MoE layer."""

    def __init__(self, model):
        self.layers = moe_layers(model)
        self.counts = [torch.zeros(m.num_experts, dtype=torch.long) for m in self.layers]
        self.aux_sum, self.batches, self.dropped = 0.0, 0, 0

    def add(self, real_mask, aux, dropped):
        keep = real_mask.reshape(-1).cpu()
        for i, m in enumerate(self.layers):
            idx = m.last_topk.cpu()[keep]
            self.counts[i] += torch.bincount(idx.reshape(-1), minlength=m.num_experts)
        self.aux_sum += float(aux)
        self.batches += 1
        self.dropped += int(dropped)

    def summary(self):
        if not self.layers:
            return None
        per_layer = []
        for c in self.counts:
            frac = (c.double() / max(1, int(c.sum()))).tolist()
            ent = -sum(f * math.log(f) for f in frac if f > 0) / math.log(len(frac))
            per_layer.append({"load_fraction": [round(f, 5) for f in frac],
                              "normalized_entropy": round(ent, 5), "max_load": round(max(frac), 5)})
        return {"per_layer": per_layer,
                "mean_aux_loss_summed_over_layers": self.aux_sum / max(1, self.batches),
                "dropped_assignments": self.dropped}


def evaluate(model, documents, unigram_bits, args, device):
    model.eval()
    neural_nll, baseline_bits, scored, dropped, slots = 0.0, 0.0, 0, 0, 0
    routing = RoutingStats(model)

    with torch.inference_mode():
        for x, y in batches(
            documents, args.context, args.batch_size, device
        ):
            logits, aux, d = run(model, x)
            if routing.layers:
                routing.add(y != -100, aux, d)
            dropped += d
            slots += x.numel() * (0 if args.model == "dense" else 2 * 2)  # tokens * top_k * layers
            neural_nll += F.cross_entropy(
                logits.flatten(0, 1),
                y.flatten(),
                ignore_index=-100,
                reduction="sum",
            ).item()

            targets = y[y != -100]
            baseline_bits += unigram_bits[targets].sum().item()
            scored += targets.numel()

    expected = sum(max(0, len(raw) - 1) for raw in documents)
    if scored != expected or scored == 0:
        raise RuntimeError(f"Target count: {scored}; expected {expected}")

    result = {
        "neural_bpb": neural_nll / (scored * math.log(2)),
        "unigram_bpb": baseline_bits / scored,
        "scored_bytes": scored,
        "dropped_assignment_fraction": dropped / slots if slots else 0.0,
        "dropped_assignments": dropped,
        "routing": routing.summary(),
    }
    if not all(math.isfinite(result[k])
               for k in ("neural_bpb", "unigram_bpb")):
        raise RuntimeError("Nonfinite evaluation result")
    return result


def main(args):
    if min(args.context, args.batch_size, args.epochs,
           args.steps, args.threads) < 1 or args.lr <= 0 or args.eval_every < 0:
        raise ValueError("Budgets, dimensions, threads, and LR must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    documents, manifest, counts = load_corpus(args.data)

    # Refuse to overwrite any previous run.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    total_train_bytes = sum(counts.values())
    unigram_bits = torch.tensor([
        -math.log2((counts[b] + 1) / (total_train_bytes + 256))
        for b in range(256)
    ], dtype=torch.float64, device=device)

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
    }
    config.update({
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_counts": parameter_counts(model, args),
        "layers": 2,
        "width": 32,
        "heads": 2,
        "feedforward_width": 64,
        "model_class": type(model).__name__,
        "evaluation": "chunk-reset; skip first byte of each file",
    })

    checkpoint_path = args.output / "best.pt"
    best_bpb = math.inf
    best_record = None
    steps, consumed = 0, 0
    started = time.perf_counter()
    train_seconds = 0.0
    train_routing = RoutingStats(model)
    train_ce_sum, train_batches = 0.0, 0

    last_validated = None

    def validate(epoch):
        nonlocal best_bpb, best_record, last_validated, train_routing, train_ce_sum, train_batches
        if last_validated == steps:
            return
        last_validated = steps
        since = {"batches": train_batches,
                 "mean_train_ce_bpb": train_ce_sum / train_batches / math.log(2) if train_batches else None,
                 "routing": train_routing.summary()}
        train_routing = RoutingStats(model)
        train_ce_sum, train_batches = 0.0, 0
        metrics = evaluate(
            model, documents["val"], unigram_bits, args, device
        )
        record = {
            "epoch": epoch,
            "optimizer_steps": steps,
            "training_target_bytes": consumed,
            "training_seconds": train_seconds,
            **metrics,
            "train_since_last_validation": since,
        }
        with (args.output / "validation.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({"validation": record}), flush=True)

        if metrics["neural_bpb"] < best_bpb:
            best_bpb = metrics["neural_bpb"]
            best_record = dict(record)
            torch.save({
                "state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "config": config,
                "selection": best_record,
            }, checkpoint_path)

    # Include the untrained model among candidate checkpoints.
    validate(epoch=0)

    for epoch in range(1, args.epochs + 1):
        training_docs = list(documents["train"])
        random.shuffle(training_docs)
        epoch_chunks = None
        if args.shuffle_windows:
            # Mix windows across documents so a batch is not consecutive chunks of one book.
            epoch_chunks = list(windows(training_docs, args.context))
            random.shuffle(epoch_chunks)

        for x, y in batches(
            training_docs, args.context, args.batch_size, device, chunks=epoch_chunks
        ):
            step_started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, aux, dropped_now = run(model, x)
            ce = F.cross_entropy(
                logits.flatten(0, 1),
                y.flatten(),
                ignore_index=-100,
                reduction="mean",
            )
            loss = ce + args.aux_weight * aux  # aux in the training loss only, never in reported bpb
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            train_seconds += time.perf_counter() - step_started
            train_ce_sum += ce.item()
            train_batches += 1
            if train_routing.layers:
                train_routing.add(y != -100, aux.detach(), dropped_now)

            steps += 1
            consumed += (y != -100).sum().item()
            if args.eval_every and steps % args.eval_every == 0:
                validate(epoch)
            if steps >= args.steps:
                break

        validate(epoch)
        if steps >= args.steps:
            break

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])

    reloaded_val = evaluate(
        model, documents["val"], unigram_bits, args, device
    )
    if not math.isclose(
        reloaded_val["neural_bpb"], best_bpb,
        rel_tol=1e-6, abs_tol=1e-6,
    ):
        raise RuntimeError("Reloaded checkpoint failed validation agreement")

    # Test evaluation happens only after checkpoint selection.
    test = evaluate(
        model, documents["test"], unigram_bits, args, device
    )

    ledger = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "torch_version": str(torch.__version__),
        "device": str(device),
        "optimizer_steps": steps,
        "training_target_bytes": consumed,
        "total_elapsed_seconds": time.perf_counter() - started,
        "training_seconds": train_seconds,
        "best_validation": best_record,
        "test": test,
        "checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--model", choices=["dense", "moe"], default="dense")
    parser.add_argument("--position", choices=["none", "learned", "sinusoidal", "fourier"],
                        default="learned")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--capacity-mode", choices=["none", "causal"], default="none",
                        help="none = dropless routing (default); causal = per-sequence prefix capacity")
    parser.add_argument("--shuffle-windows", action="store_true",
                        help="shuffle training windows across documents each epoch")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="also validate every N optimizer steps (0 = epoch ends only)")
    parser.add_argument("--aux-weight", type=float, default=0.01)
    main(parser.parse_args())
