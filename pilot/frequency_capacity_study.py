"""Frequency-and-capacity study for a coordinate-network surrogate.

Target: h(x, y, t) = sin(k * (x - 0.5 t)),  k in {1, 2, 4, 8}
Domain: x, y ~ U[0, 2pi], t ~ U[0, 2]. Unless a run says norm=0, inputs are
linearly normalised to [-1, 1] (x/pi - 1, y/pi - 1, t - 1).
Splits: 2000 train / 500 val / 500 test, drawn ONCE from a torch.Generator with
fixed DATA_SEED=20261023. Every run (all phases, all seeds) uses the same
train/val/test points; the run seed varies only initialisation (and nothing
else is random in full-batch Adam). The y-resampling draw is also fixed.

What this is: SUPERVISED REGRESSION against analytical labels (MSE on labelled
points). It is NOT physics-informed: no PDE residual appears in the loss.
Train/val/test are independent samples from the SAME domain, so test error
measures in-domain interpolation, not temporal or out-of-domain generalisation.

Selection protocol: every choice (w0_first, w0_hidden, lr, epoch budget, tanh lr,
best checkpoint) is made on VALIDATION MSE using selection seeds 100-102.
Test MSE is computed only in the final phases (seeds 0-4) and never used to
choose anything.

Phases (in order)
  ablation   original k=1 setup (SIREN 32x2, w0=30/30, lr 1e-3):
             norm {off,on} x epochs {1000,5000}, seeds 0-4, test evaluated.
  sweep_w0   w0_first {1,3,10,30} x w0_hidden {1,10,30}, lr 1e-3, 3000 epochs.
  sweep_tanh tanh MLP 32x2 lr {3e-4,1e-3,3e-3}, 3000 epochs (baseline gets its own lr).
  sweep_lr   selected w0 pair, lr {3e-4,1e-3,3e-3}, 3000 epochs  (only lr varies).
  sweep_ep   selected w0 pair, lr 1e-3, epochs {1000,3000,10000} (only budget varies).
  A          final SIREN 32x2 at selected settings, every k, seeds 0-4.
  width      depth 2, width {16,64,128} (32 = phase A), every k.
  depth      width 32, depth {1,3,4} (2 = phase A), every k.
  tanh       tanh MLP with identical layer shapes (=> identical parameter count)
             for every width/depth config above, selected tanh lr.
  xt_only    SIREN 32x2 with y removed from the input (2 inputs), every k.
Latency is measured afterwards, sequentially in one process (1 thread), with the
identical warmup/reps/batch procedure for the analytical formula and the network.
"""
import argparse, csv, json, os, platform, sys, time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn

N_TRAIN, N_VAL, N_TEST = 2000, 500, 500
THRESHOLD = 1e-4
KS = [1, 2, 4, 8]
STUDY_SEEDS = [0, 1, 2, 3, 4]
SEL_SEEDS = [100, 101, 102]
W0_FIRST = [1.0, 3.0, 10.0, 30.0]
W0_HIDDEN = [1.0, 10.0, 30.0]
LRS = [3e-4, 1e-3, 3e-3]
BUDGETS = [1000, 3000, 10000]
WIDTHS = [16, 32, 64, 128]
DEPTHS = [1, 2, 3, 4]
CURVE_EVERY = 50
LAT_WARMUP, LAT_REPS, LAT_BATCH = 10, 100, 500
C = 0.5
DATA_SEED = 20261023
BUDGET_TIE_DECADES = 0.1


class SirenLayer(nn.Module):
    def __init__(self, fin, fout, w0, is_first):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(fin, fout)
        with torch.no_grad():
            b = 1.0 / fin if is_first else np.sqrt(6.0 / fin) / w0
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


def build_model(kind, fin, width, depth, w0f, w0h):
    layers = []
    for i in range(depth):
        if kind == "siren":
            layers.append(SirenLayer(fin, width, w0f if i == 0 else w0h, is_first=(i == 0)))
        else:
            layers += [nn.Linear(fin, width), nn.Tanh()]
        fin = width
    out = nn.Linear(fin, 1)
    if kind == "siren":
        with torch.no_grad():
            b = np.sqrt(6.0 / fin) / (w0h if depth > 1 else w0f)
            out.weight.uniform_(-b, b)
    layers.append(out)
    return nn.Sequential(*layers)


def analytical(raw, k):
    return torch.sin(k * (raw[:, 0:1] - C * raw[:, 2:3]))


def featurise(raw, norm, use_y):
    if norm:
        x, y, t = raw[:, 0:1] / np.pi - 1.0, raw[:, 1:2] / np.pi - 1.0, raw[:, 2:3] - 1.0
    else:
        x, y, t = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3]
    return torch.cat([x, y, t] if use_y else [x, t], dim=1)


def sample(n, gen):
    x = torch.rand(n, 1, generator=gen) * 2 * np.pi
    y = torch.rand(n, 1, generator=gen) * 2 * np.pi
    t = torch.rand(n, 1, generator=gen) * 2.0
    return torch.cat([x, y, t], dim=1)


def latency(fn, inp):
    with torch.no_grad():
        for _ in range(LAT_WARMUP):
            fn(inp)
        ts = []
        for _ in range(LAT_REPS):
            t0 = time.perf_counter()
            fn(inp)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def run_one(cfg):
    torch.set_num_threads(1)
    phase, kind, k, width, depth = cfg["phase"], cfg["model"], cfg["k"], cfg["width"], cfg["depth"]
    w0f, w0h, lr, epochs, seed = cfg["w0_first"], cfg["w0_hidden"], cfg["lr"], cfg["epochs"], cfg["seed"]
    norm, use_y = cfg["norm"], cfg["use_y"]
    gen = torch.Generator().manual_seed(DATA_SEED)
    tr_raw, va_raw, te_raw = sample(N_TRAIN, gen), sample(N_VAL, gen), sample(N_TEST, gen)
    y_resample = torch.rand(N_TEST, generator=gen) * 2 * np.pi
    f = lambda r: featurise(r, norm, use_y)
    tr_x, va_x, te_x = f(tr_raw), f(va_raw), f(te_raw)
    tr_y, va_y, te_y = analytical(tr_raw, k), analytical(va_raw, k), analytical(te_raw, k)

    torch.manual_seed(seed)
    model = build_model(kind, 3 if use_y else 2, width, depth, w0f, w0h)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    best_val, best_ep, first_thr, best_state, curve = float("inf"), -1, None, None, []
    first_thr_s = None
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = mse(model(tr_x), tr_y)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = mse(model(va_x), va_y).item()
        if v < best_val:
            best_val, best_ep = v, ep
            best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        if first_thr is None and v <= THRESHOLD:
            first_thr, first_thr_s = ep, time.perf_counter() - t0
        if ep == 1 or ep % CURVE_EVERY == 0:
            curve.append([ep, loss.item(), v])  # train MSE is pre-step loss of this epoch
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    rec = dict(cfg)
    rec.update(params=params, best_val_mse=best_val, best_epoch=best_ep,
               threshold_status="reached" if best_val <= THRESHOLD else "threshold not reached",
               first_threshold_epoch=first_thr if first_thr else "",
               first_threshold_seconds=first_thr_s if first_thr else "",
               train_seconds=train_s, final_train_mse=curve[-1][1] if curve else "")
    with torch.no_grad():
        rec["best_ckpt_train_mse"] = mse(model(tr_x), tr_y).item()
        rec["val_over_train_at_best"] = best_val / max(rec["best_ckpt_train_mse"], 1e-30)
        if phase not in ("sweep_w0", "sweep_tanh", "sweep_lr", "sweep_ep"):
            p = model(te_x)
            rec["test_mse"] = mse(p, te_y).item()
            rec["test_rel_l2"] = (torch.norm(p - te_y) / torch.norm(te_y)).item()
            if use_y:
                mut = te_raw.clone(); mut[:, 1] = y_resample
                rec["msq_pred_change_y_resample"] = torch.mean((p - model(f(mut))) ** 2).item()
            else:
                rec["msq_pred_change_y_resample"] = 0.0  # y is not an input (structural zero)
    state = best_state if phase not in ("sweep_w0", "sweep_tanh", "sweep_lr", "sweep_ep") else None
    return rec, curve, state


FIELDS = ["run_id", "phase", "model", "k", "width", "depth", "w0_first", "w0_hidden", "lr", "epochs",
          "norm", "use_y", "seed", "params", "best_val_mse", "best_epoch", "threshold_status",
          "first_threshold_epoch", "first_threshold_seconds", "train_seconds", "final_train_mse", "best_ckpt_train_mse",
          "val_over_train_at_best", "test_mse", "test_rel_l2", "msq_pred_change_y_resample",
          "net_lat_median_ms", "net_lat_p95_ms", "ana_lat_median_ms", "ana_lat_p95_ms"]


def cfg(phase, model, k, width, depth, w0f, w0h, lr, epochs, seed, norm=True, use_y=True):
    return dict(phase=phase, model=model, k=k, width=width, depth=depth,
                w0_first=w0f if model == "siren" else "", w0_hidden=w0h if model == "siren" else "",
                lr=lr, epochs=epochs, norm=int(norm), use_y=int(use_y), seed=seed)


def score(rows, key):
    """mean log10(best val MSE) per key over all k and selection seeds (lower is better)."""
    out = {}
    for r in rows:
        out.setdefault(key(r), []).append(np.log10(max(r["best_val_mse"], 1e-30)))
    return {kk: float(np.mean(v)) for kk, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    env = dict(data_seed=DATA_SEED, python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
               cpu_model=cpu, logical_cpus=os.cpu_count(), os=platform.platform(),
               torch_num_threads_per_training_process=1, training_worker_processes=a.workers,
               latency_phase="sequential, single process, torch.set_num_threads(1)",
               cmd=" ".join([sys.executable] + sys.argv))
    print("ENV", json.dumps(env), flush=True)
    json.dump(env, open(os.path.join(a.out, "frequency_capacity_env.json"), "w"), indent=2)
    fcsv = open(os.path.join(a.out, "frequency_capacity_runs.csv"), "w", newline="")
    wr = csv.DictWriter(fcsv, fieldnames=FIELDS); wr.writeheader()
    fcur = open(os.path.join(a.out, "frequency_capacity_curves.jsonl"), "w")
    pool = mp.get_context("fork").Pool(a.workers)
    all_rows, states, counter = [], {}, [0]
    t_all = time.perf_counter()

    def batch(cfgs, label):
        t0 = time.perf_counter()
        res = pool.map(run_one, cfgs, chunksize=1)
        rows = []
        for rec, curve, state in res:
            counter[0] += 1
            rec["run_id"] = counter[0]
            fcur.write(json.dumps({"run_id": rec["run_id"], "phase": rec["phase"], "model": rec["model"],
                                   "k": rec["k"], "width": rec["width"], "depth": rec["depth"],
                                   "seed": rec["seed"], "curve_epoch_train_val": curve}) + "\n")
            if state is not None:
                states[rec["run_id"]] = (rec, state)
            print("RUN", json.dumps(rec), flush=True)
            rows.append(rec)
        fcur.flush()
        all_rows.extend(rows)
        print(f"PHASE_DONE {label} runs={len(rows)} wall={time.perf_counter()-t0:.1f}s", flush=True)
        return rows

    # 1) independent: ablation, w0 grid, tanh lr sweep
    abl = [cfg("ablation", "siren", 1, 32, 2, 30.0, 30.0, 1e-3, ep, s, norm=n)
           for n in (False, True) for ep in (1000, 5000) for s in STUDY_SEEDS]
    g = [cfg("sweep_w0", "siren", k, 32, 2, f0, h0, 1e-3, 3000, s)
         for f0 in W0_FIRST for h0 in W0_HIDDEN for k in KS for s in SEL_SEEDS]
    tl = [cfg("sweep_tanh", "tanh", k, 32, 2, None, None, lr, 3000, s) for lr in LRS for k in KS for s in SEL_SEEDS]
    rows = batch(abl + g + tl, "ablation+sweep_w0+sweep_tanh")
    sw = score([r for r in rows if r["phase"] == "sweep_w0"], lambda r: (r["w0_first"], r["w0_hidden"]))
    W0F, W0H = min(sw, key=sw.get)
    print("SELECT_W0 mean log10 best-val MSE:", json.dumps({f"{a_}/{b_}": v for (a_, b_), v in sw.items()}),
          f"-> w0_first={W0F} w0_hidden={W0H}", flush=True)
    st = score([r for r in rows if r["phase"] == "sweep_tanh"], lambda r: r["lr"])
    TANH_LR = min(st, key=st.get)
    print("SELECT_TANH_LR:", json.dumps(st), f"-> {TANH_LR}", flush=True)

    # 2) one factor at a time from the selected w0 pair
    lrc = [cfg("sweep_lr", "siren", k, 32, 2, W0F, W0H, lr, 3000, s) for lr in LRS for k in KS for s in SEL_SEEDS]
    epc = [cfg("sweep_ep", "siren", k, 32, 2, W0F, W0H, 1e-3, ep, s) for ep in BUDGETS for k in KS for s in SEL_SEEDS]
    rows = batch(lrc + epc, "sweep_lr+sweep_ep")
    sl = score([r for r in rows if r["phase"] == "sweep_lr"], lambda r: r["lr"])
    LR = min(sl, key=sl.get)
    se = score([r for r in rows if r["phase"] == "sweep_ep"], lambda r: r["epochs"])
    best_e = min(se.values())
    EPOCHS = min(e for e, v in se.items() if v <= best_e + BUDGET_TIE_DECADES)
    print("SELECT_LR:", json.dumps(sl), f"-> {LR}", flush=True)
    print("SELECT_EPOCHS:", json.dumps(se), f"tie={BUDGET_TIE_DECADES} decades -> {EPOCHS}", flush=True)
    sel = dict(w0_first=W0F, w0_hidden=W0H, lr=LR, epochs=EPOCHS, tanh_lr=TANH_LR,
               scores=dict(w0={f"{a_}/{b_}": v for (a_, b_), v in sw.items()}, lr=sl, epochs=se, tanh_lr=st))
    json.dump(sel, open(os.path.join(a.out, "frequency_capacity_selection.json"), "w"), indent=2)

    # 3) final evaluation on seeds 0-4 (test used here only)
    fin = [cfg("A", "siren", k, 32, 2, W0F, W0H, LR, EPOCHS, s) for k in KS for s in STUDY_SEEDS]
    fin += [cfg("width", "siren", k, w, 2, W0F, W0H, LR, EPOCHS, s) for w in WIDTHS if w != 32 for k in KS for s in STUDY_SEEDS]
    fin += [cfg("depth", "siren", k, 32, d, W0F, W0H, LR, EPOCHS, s) for d in DEPTHS if d != 2 for k in KS for s in STUDY_SEEDS]
    shapes = [(w, 2) for w in WIDTHS] + [(32, d) for d in DEPTHS if d != 2]
    fin += [cfg("tanh", "tanh", k, w, d, None, None, TANH_LR, EPOCHS, s) for (w, d) in shapes for k in KS for s in STUDY_SEEDS]
    fin += [cfg("xt_only", "siren", k, 32, 2, W0F, W0H, LR, EPOCHS, s, use_y=False) for k in KS for s in STUDY_SEEDS]
    fin.sort(key=lambda c: -c["width"] * c["depth"])  # long jobs first
    batch(fin, "final")
    pool.close(); pool.join()

    # 4) latency, sequential, identical procedure for formula and network
    t0 = time.perf_counter()
    for rid, (rec, state) in states.items():
        m = build_model(rec["model"], 3 if rec["use_y"] else 2, rec["width"], rec["depth"],
                        rec["w0_first"] or 1.0, rec["w0_hidden"] or 1.0)
        m.load_state_dict(state); m.eval()
        gen = torch.Generator().manual_seed(DATA_SEED)
        sample(N_TRAIN, gen); sample(N_VAL, gen); te_raw = sample(N_TEST, gen)
        te_x = featurise(te_raw, rec["norm"], rec["use_y"])
        rec["net_lat_median_ms"], rec["net_lat_p95_ms"] = latency(m, te_x)
        rec["ana_lat_median_ms"], rec["ana_lat_p95_ms"] = latency(lambda r: analytical(r, rec["k"]), te_raw)
    print(f"PHASE_DONE latency runs={len(states)} wall={time.perf_counter()-t0:.1f}s", flush=True)
    for r in sorted(all_rows, key=lambda r: r["run_id"]):
        wr.writerow(r)
    fcsv.close(); fcur.close()
    print(f"TOTAL_WALL_SECONDS {time.perf_counter() - t_all:.1f}", flush=True)


if __name__ == "__main__":
    main()
