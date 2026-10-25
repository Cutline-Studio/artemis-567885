"""Self-training engine for the dense model (rewrite of expand.py + supervisor.py ideas for dense growth).

One cycle:
  1. load the last good checkpoint (or fresh init), apply any growth decided last cycle
  2. train `steps_per_cycle` steps on byte batches from `batch_fn`
  3. evaluate on a fixed held-out batch
  4. guard: non-finite loss anywhere, or eval loss > best * (1 + regression_tol)
            -> ROLL BACK (restore last good checkpoint, discard this cycle) and log why
  5. otherwise checkpoint; if relative improvement < stall_rel_improvement, plan growth
     within caps: add a layer (until num_layers_max), else widen d_ff (until d_ff_max).
     Growth is function-preserving (zero-init output projections), so it can't spike loss.

Everything lives in `run_dir`: state.json, ckpt.pt, ckpt_prev.pt, self_train_log.jsonl,
exec_signal.jsonl. No network, no GPU required (device defaults to CPU).
"""
import json, math, os, shutil, time
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional

import torch

from ..config import DenseConfig
from ..model import ArtemisDense
from .. import killswitch


@dataclass
class TrainConfig:
    lr: float = 3e-4
    steps_per_cycle: int = 50
    batch_size: int = 8
    seq_len: int = 64
    seed: int = 567885
    stall_rel_improvement: float = 0.01
    regression_tol: float = 0.10      # roll back if eval loss > best * 1.10
    d_ff_step: int = 256
    grad_clip: float = 1.0
    sample_selection: str = "random"  # "random" | "hard": the model scores a pool, train on the hardest
    pool_factor: int = 4              # pool size = pool_factor * batch_size when sample_selection == "hard"


DEFAULT_TEXT = (b"Artemis 567885 is one dense transformer. Every token uses every weight. "
                b"It trains itself in cycles, grows when progress stalls, and rolls back on failure. ") * 8


def text_batch_fn(data: bytes = DEFAULT_TEXT):
    """Return batch_fn(generator, B, T) -> (x, y) sampling windows of raw bytes."""
    buf = torch.tensor(list(data), dtype=torch.long)

    def fn(g: torch.Generator, B: int, T: int):
        ix = torch.randint(0, len(buf) - T - 1, (B,), generator=g)
        x = torch.stack([buf[i:i + T] for i in ix])
        y = torch.stack([buf[i + 1:i + T + 1] for i in ix])
        return x, y
    return fn


class RolledBack(Exception):
    pass


class SelfTrainer:
    def __init__(self, run_dir: str, model_cfg: Optional[DenseConfig] = None,
                 train_cfg: Optional[TrainConfig] = None, batch_fn: Optional[Callable] = None,
                 inject: Optional[Callable[[ArtemisDense, int, int, torch.Tensor], torch.Tensor]] = None):
        self.run_dir = os.path.abspath(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.tc = train_cfg or TrainConfig()
        self.batch_fn = batch_fn or text_batch_fn()
        self.inject = inject                 # test hook: inject(model, cycle, step, loss) -> loss
        self.p = {k: os.path.join(self.run_dir, v) for k, v in {
            "state": "state.json", "ckpt": "ckpt.pt", "prev": "ckpt_prev.pt",
            "log": "self_train_log.jsonl", "signal": "exec_signal.jsonl"}.items()}
        if os.path.exists(self.p["state"]):
            self.state = json.load(open(self.p["state"]))
        else:
            self.state = {"cycle": 0, "best_eval": None, "pending_growth": None, "history": [],
                          "model_cfg": (model_cfg or DenseConfig()).to_dict()}
            self._save_state()

    # ------------------------------------------------------------- persistence
    def _save_state(self):
        tmp = self.p["state"] + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=2)
        os.replace(tmp, self.p["state"])

    def _log(self, rec: dict):
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **rec}
        with open(self.p["log"], "a") as f:
            f.write(json.dumps(rec) + "\n")

    def load_model(self) -> ArtemisDense:
        cfg = DenseConfig.from_dict(dict(self.state["model_cfg"]))
        torch.manual_seed(self.tc.seed)
        if os.path.exists(self.p["ckpt"]):
            ck = torch.load(self.p["ckpt"], weights_only=False)
            cfg = DenseConfig.from_dict(ck["cfg"])
            m = ArtemisDense(cfg)
            m.load_state_dict(ck["model"])
        else:
            m = ArtemisDense(cfg)
        return m

    def _checkpoint(self, model: ArtemisDense):
        if os.path.exists(self.p["ckpt"]):
            shutil.copy2(self.p["ckpt"], self.p["prev"])
        torch.save({"cfg": model.cfg.to_dict(), "model": model.state_dict()}, self.p["ckpt"] + ".tmp")
        os.replace(self.p["ckpt"] + ".tmp", self.p["ckpt"])
        self.state["model_cfg"] = model.cfg.to_dict()

    # ------------------------------------------------------------- growth
    def plan_growth(self, cfg: DenseConfig) -> Optional[dict]:
        caps = cfg.caps
        if cfg.n_layers < caps.get("num_layers_max", cfg.n_layers):
            return {"kind": "add_layer"}
        new_ff = min(cfg.d_ff + self.tc.d_ff_step, caps.get("d_ff_max", cfg.d_ff))
        if new_ff > cfg.d_ff:
            return {"kind": "widen_ffn", "d_ff": new_ff}
        return None   # at caps

    @staticmethod
    def apply_growth(model: ArtemisDense, growth: dict, generator=None) -> str:
        if growth["kind"] == "add_layer":
            i = model.add_layer()
            return f"added layer {i} (zero-init wo/w_down => identity)"
        if growth["kind"] == "widen_ffn":
            old = model.cfg.d_ff
            model.widen_ffn(growth["d_ff"], generator)
            return f"widened d_ff {old}->{growth['d_ff']} (zero-init new w_down cols)"
        raise ValueError(growth)

    # ------------------------------------------------------------- self-scored sample selection
    @staticmethod
    @torch.no_grad()
    def select_hard(model: ArtemisDense, x: torch.Tensor, y: torch.Tensor, k: int):
        """The model scores its own predictions (mean next-byte log-prob per window) and keeps
        the k windows it finds hardest (lowest log-prob). Returns (x, y, kept_scores)."""
        was = model.training
        model.eval()
        logits, _ = model(x)
        lp = torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1).mean(-1)
        model.train(was)
        idx = torch.argsort(lp)[:k]
        return x[idx], y[idx], lp[idx]

    def next_batch(self, model: ArtemisDense, g: torch.Generator):
        B, T = self.tc.batch_size, self.tc.seq_len
        if self.tc.sample_selection == "hard":
            x, y = self.batch_fn(g, B * self.tc.pool_factor, T)
            x, y, _ = self.select_hard(model, x, y, B)
            return x, y
        return self.batch_fn(g, B, T)

    # ------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, model: ArtemisDense) -> float:
        g = torch.Generator().manual_seed(self.tc.seed - 1)
        x, y = self.batch_fn(g, self.tc.batch_size * 2, self.tc.seq_len)
        model.eval()
        _, loss = model(x, y)
        return float(loss)

    # ------------------------------------------------------------- one cycle
    def run_cycle(self) -> dict:
        killswitch.check(self.run_dir)
        t0 = time.time()
        cyc = self.state["cycle"]
        model = self.load_model()
        growth_note = None
        if self.state.get("pending_growth"):
            growth_note = self.apply_growth(model, self.state["pending_growth"],
                                            torch.Generator().manual_seed(self.tc.seed + 7 + cyc))
        opt = torch.optim.AdamW(model.parameters(), lr=self.tc.lr)
        g = torch.Generator().manual_seed(self.tc.seed + cyc)
        losses = []
        rec = {"cycle": cyc, "growth_applied": growth_note, "n_layers": model.num_layers,
               "d_ff": model.cfg.d_ff, "params": model.num_params()}
        try:
            model.train()
            for step in range(self.tc.steps_per_cycle):
                killswitch.check(self.run_dir)   # Killed propagates: nothing is checkpointed
                x, y = self.next_batch(model, g)
                _, loss = model(x, y)
                if self.inject:
                    loss = self.inject(model, cyc, step, loss)
                if not torch.isfinite(loss):
                    raise RolledBack(f"non-finite train loss at step {step}: {loss.item()}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.tc.grad_clip)
                opt.step()
                losses.append(loss.item())
            ev = self.evaluate(model)
            if not math.isfinite(ev):
                raise RolledBack(f"non-finite eval loss {ev}")
            best = self.state["best_eval"]
            if best is not None and ev > best * (1 + self.tc.regression_tol):
                raise RolledBack(f"eval regression {ev:.4f} > best {best:.4f} * (1+{self.tc.regression_tol})")
        except RolledBack as e:
            # discard this cycle's weights: the on-disk checkpoint is the last good one
            rec.update({"status": "rolled_back", "reason": str(e),
                        "restored": "ckpt.pt" if os.path.exists(self.p["ckpt"]) else "fresh_init"})
            self.state["pending_growth"] = None   # don't retry the same growth blindly
            self.state["cycle"] = cyc + 1
            self.state["history"].append(rec)
            self._save_state(); self._log(rec)
            return rec

        w = max(1, len(losses) // 5)
        start, end = sum(losses[:w]) / w, sum(losses[-w:]) / w
        rel = (start - end) / start if start > 0 else 0.0
        self._checkpoint(model)
        best = self.state["best_eval"]
        self.state["best_eval"] = ev if best is None else min(best, ev)
        growth = self.plan_growth(model.cfg) if rel < self.tc.stall_rel_improvement else None
        self.state["pending_growth"] = growth
        rec.update({"status": "ok", "train_loss_start": round(start, 6), "train_loss_end": round(end, 6),
                    "rel_improvement": round(rel, 6), "eval_loss": round(ev, 6),
                    "planned_growth": growth, "seconds": round(time.time() - t0, 3)})
        self.state["cycle"] = cyc + 1
        self.state["history"].append(rec)
        self._save_state(); self._log(rec)
        return rec

    # ------------------------------------------------------------- code_exec hook (stub)
    def add_execution_signal(self, record: dict) -> str:
        """Record a code_exec result as future training/eval data. STUB: appends to
        exec_signal.jsonl only; nothing here trains on it."""
        keep = {k: record.get(k) for k in ("task", "code", "passed", "stdout", "stderr", "iterations")}
        with open(self.p["signal"], "a") as f:
            f.write(json.dumps(keep) + "\n")
        return self.p["signal"]
