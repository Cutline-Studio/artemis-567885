"""Check 2 (real checkpoints): reload the three ACTUAL best.pt files written by
bytelm/artemis_corpus_benchmark.py (seeds 42/1/2) into fresh ByteModel instances,
re-run the script's own evaluate() on the full val and test splits, and compare
with the numbers recorded in each run's metrics.json. Also save->reload the
loaded state_dict again and require torch.equal logits on a fixed input.

Usage: python3 checks/check2_real_checkpoints.py --runs <dir with seed*/best.pt> --data <corpus dir>
"""
import argparse, hashlib, json, sys, os, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checks._load import load

B = load("bytelm/artemis_corpus_benchmark.py", "byte_bench")

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, required=True)
ap.add_argument("--data", type=Path, required=True)
a = ap.parse_args()
torch.set_num_threads(1)  # the runs used --threads 1
print("torch", torch.__version__, "threads", torch.get_num_threads())
docs, manifest, counts = B.load_corpus(a.data)
total = sum(counts.values())
import math
uni = torch.tensor([-math.log2((counts[b] + 1) / (total + 256)) for b in range(256)], dtype=torch.float64)
ok = True
for seed in (42, 1, 2):
    run = a.runs / f"seed{seed}"
    met = json.loads((run / "metrics.json").read_text())
    raw = (run / "best.pt").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    torch.manual_seed(12345)
    m = B.ByteModel(128); m.load_state_dict(ck["state_dict"]); m.eval()
    args = SimpleNamespace(context=128, batch_size=16)
    val = B.evaluate(m, docs["val"], uni, args, "cpu")
    test = B.evaluate(m, docs["test"], uni, args, "cpu")
    x = torch.randint(0, 256, (4, 128), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        l1 = m(x)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "re.pt"); torch.save(m.state_dict(), p)
        torch.manual_seed(54321); m2 = B.ByteModel(128); m2.load_state_dict(torch.load(p, weights_only=True)); m2.eval()
        with torch.no_grad():
            l2 = m2(x)
    rec = dict(seed=seed, sha256_matches_metrics=(sha == met["checkpoint_sha256"]),
               val_bpb_reloaded=val["neural_bpb"], val_bpb_recorded=met["best_validation"]["neural_bpb"],
               val_abs_diff=abs(val["neural_bpb"] - met["best_validation"]["neural_bpb"]),
               test_bpb_reloaded=test["neural_bpb"], test_bpb_recorded=met["test"]["neural_bpb"],
               test_abs_diff=abs(test["neural_bpb"] - met["test"]["neural_bpb"]),
               scored_val=val["scored_bytes"], scored_test=test["scored_bytes"],
               resave_logits_equal=bool(torch.equal(l1, l2)), resave_max_abs_diff=(l1 - l2).abs().max().item())
    print(json.dumps(rec), flush=True)
    ok &= rec["sha256_matches_metrics"] and rec["resave_logits_equal"] and rec["val_abs_diff"] == 0.0 and rec["test_abs_diff"] == 0.0
print("CHECK2_REAL_CHECKPOINTS", "PASS" if ok else "FAIL (see diffs above)")
