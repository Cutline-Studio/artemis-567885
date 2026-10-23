"""Check 4b: short CPU PINN training (physics_simulation.train) with
PDE-residual + initial-condition + inflow-boundary-condition losses (weights
w_ic = w_bc = 10, aux alpha = 0.01) AND a held-out solution-error metric against
the validated Lax-Wendroff reference (physics_simulation.heldout_reference:
161x161 grid, every 4th node, t in {0.25, 0.5, 0.75, 1.0}; 6724 points never
sampled in training). Logged every --log-every steps and at the end.

Usage: python3 checks/check4_pinn_train.py --seconds 180 --seed 567885
"""
import argparse, json, os, sys, platform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import physics_simulation as P

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=180)
ap.add_argument("--seed", type=int, default=567885)
ap.add_argument("--log-every", type=int, default=250)
a = ap.parse_args()
print("ENV", json.dumps(dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
                             os=platform.platform(), seconds=a.seconds, seed=a.seed,
                             threads=max(1, min(8, os.cpu_count() or 1)))), flush=True)
ref = P.heldout_reference()
print("HELDOUT", json.dumps(dict(points=int(ref["coords"].shape[0]), fd_vs_exact_rel_l2=ref["fd_vs_exact_rel_l2"],
                                 solver={k: (float(v) if isinstance(v, (np.floating,)) else v)
                                         for k, v in ref["solver"].items()})), flush=True)
model, final, hist = P.train(seconds=a.seconds, seed=a.seed, log_every=a.log_every, heldout=ref)
for h in hist:
    print("LOG", json.dumps(h), flush=True)
print("FINAL", json.dumps({k: v for k, v in final.items() if k != "expert_usage_top1_by_region"}), flush=True)
