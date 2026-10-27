"""Stage 8b physics study: run one physics_simulation.train config with a fixed step
budget and write final.json, error_vs_t.json, train_log.jsonl and console to
runs/stage8b/<name>/.

Usage: python3 tools/stage8b_physics_study.py NAME [--steps 6358] [--threads 2] [train kwargs as k=v ...]
e.g.   python3 tools/stage8b_physics_study.py cosine lr_schedule=cosine
"""
import argparse, json, os, platform, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
import physics_simulation as P

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--steps", type=int, default=6358)  # = iteration-1 run (120 s time budget reached step 6358)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("kw", nargs="*")
a = ap.parse_args()
kw = {}
for item in a.kw:
    k, v = item.split("=", 1)
    try:
        v = float(v) if any(c in v for c in ".e") else int(v)
    except ValueError:
        pass
    kw[k] = v
out = os.path.join(ROOT, "runs", "stage8b", a.name)
os.makedirs(out, exist_ok=True)
log = os.path.join(out, "train_log.jsonl")
if os.path.exists(log):
    os.remove(log)
env = dict(python=sys.version.split()[0], torch=torch.__version__, os=platform.platform(), threads=a.threads,
           steps=a.steps, seed=kw.get("seed", 567885), kwargs=kw, start=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
print("ENV", json.dumps(env), flush=True)
t0 = time.time()
model, final, hist = P.train(seconds=1e9, max_steps=a.steps, log_path=log, threads=a.threads, **kw)
wall = time.time() - t0
final["wall_seconds"] = round(wall, 1)
final["config"] = dict(name=a.name, steps=a.steps, threads=a.threads, **kw)
evt = P.error_vs_t(model)
with open(os.path.join(out, "final.json"), "w") as f:
    json.dump(final, f, indent=2)
with open(os.path.join(out, "error_vs_t.json"), "w") as f:
    json.dump(evt, f, indent=2)
print("FINAL", json.dumps({k: v for k, v in final.items() if k != "expert_usage_top1_by_region"}), flush=True)
print("ERROR_VS_T", json.dumps(evt), flush=True)
