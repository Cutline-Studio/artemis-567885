"""Run every arm x seed of the frozen protocol_v2.json with 2 workers.

Guards before each launch: code checkout HEAD equals the frozen commit, tracked files unchanged,
corpus manifest hash unchanged, output directory does not exist. Every launch and outcome
(completed, failed, refused) is appended to runs_v2/ledger.jsonl; nothing is deleted.
"""
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

ROOT = Path("/home/claude/language-benchmark")
CODE = Path("/home/claude/wt-stage2")
PROTO = json.loads((ROOT / "protocol_v2.json").read_text())
OUT = ROOT / "runs_v2"
LEDGER = OUT / "ledger.jsonl"
lock = Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def log(entry):
    with lock, LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": now(), **entry}) + "\n")


def guard():
    head = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(CODE), "diff", "--quiet", "HEAD", "--"], capture_output=True).returncode
    man = hashlib.sha256((ROOT / "corpus" / "corpus_manifest.json").read_bytes()).hexdigest()
    problems = []
    if head != PROTO["code_commit"]:
        problems.append(f"HEAD {head} != frozen {PROTO['code_commit']}")
    if dirty:
        problems.append("tracked files differ from frozen commit")
    if man != PROTO["corpus_manifest_sha256"]:
        problems.append("corpus manifest hash changed")
    return head, problems


def run(arm, seed):
    out = OUT / arm / f"seed{seed}"
    cmd = [sys.executable, "-I", str(CODE / "experiments" / "corpus_benchmark.py"),
           "--data", str(ROOT / "corpus"), "--output", str(out), "--seed", str(seed),
           *PROTO["common_args"], *PROTO["arm_args"][arm]]
    head, problems = guard()
    if problems or out.exists():
        log({"arm": arm, "seed": seed, "status": "refused", "reasons": problems or ["output exists"]})
        return
    (OUT / arm).mkdir(parents=True, exist_ok=True)
    log({"arm": arm, "seed": seed, "status": "launched", "commit": head, "command": cmd})
    started = time.time()
    with open(OUT / arm / f"seed{seed}.log", "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    log({"arm": arm, "seed": seed, "status": "completed" if rc == 0 else "failed",
         "exit_code": rc, "wall_seconds": round(time.time() - started, 1)})


def main():
    OUT.mkdir(exist_ok=True)
    log({"status": "batch_started", "protocol_sha256": hashlib.sha256((ROOT / "protocol_v2.json").read_bytes()).hexdigest()})
    jobs = [(arm, seed) for seed in PROTO["seeds"] for arm in PROTO["arm_args"]]
    with ThreadPoolExecutor(max_workers=PROTO["parallel_workers"]) as pool:
        list(pool.map(lambda j: run(*j), jobs))
    log({"status": "batch_finished"})


if __name__ == "__main__":
    main()
