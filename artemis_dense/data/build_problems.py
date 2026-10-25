"""Generate math + physics problem sets with engine-verified answers -> data/problems/{math,physics}.jsonl.
Only records with verified=True are written.  python3 -m artemis_dense.data.build_problems [--n 200]"""
import argparse, json, os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None, out_dir=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out_dir = out_dir or os.path.join(PKG, "data", "problems")
    os.makedirs(out_dir, exist_ok=True)
    from ..engines.math import problem_set as mps
    sets = {"math": mps(a.n, a.seed)}
    try:
        from ..engines.physics import problem_set as pps
        sets["physics"] = pps(max(1, a.n // 4), a.seed)
    except ImportError:
        pass
    counts = {}
    for name, recs in sets.items():
        ok = [r for r in recs if r["verified"]]
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in ok:
                f.write(json.dumps(r) + "\n")
        counts[name] = {"generated": len(recs), "verified_written": len(ok)}
    print(json.dumps(counts))
    return counts


if __name__ == "__main__":
    main()
