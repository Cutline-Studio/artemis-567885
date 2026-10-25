"""Finish protocol v2: launch only the runs without a completed result, unchanged settings.

Uses run_protocol_v2.run() as-is, pointed at a detached worktree checked out at the frozen commit
(the branch head now carries a later documentation-only commit). Interrupted outputs were moved to
runs_v2/_interrupted/ beforehand and stay there.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_protocol_v2 as r  # noqa: E402

r.CODE = Path("/home/claude/wt-frozen")
JOBS = [("dense", 2), ("moe-fourier", 1), ("moe-learned", 2), ("moe-sinusoidal", 2), ("moe-fourier", 2)]

r.log({"status": "resume_started", "code_worktree": str(r.CODE), "jobs": JOBS,
       "interrupted_outputs_moved_to": "runs_v2/_interrupted/"})
with ThreadPoolExecutor(max_workers=r.PROTO["parallel_workers"]) as pool:
    list(pool.map(lambda j: r.run(*j), JOBS))
r.log({"status": "resume_finished"})
