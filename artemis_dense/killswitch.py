"""Kill switch: create a file named KILL in a run dir or any parent of it (or call engage()) and every engine stops at its next check.

Checked by: SelfTrainer.run_cycle (before and during training steps), Orchestrator.run (every step),
A2ANode.handle (every request), EnvironmentManager (every operation). Removing the file re-enables
them; that is a human action. Running code_exec children are bounded by their own timeout.
"""
import os, time

KILL_FILE = "KILL"


class Killed(Exception):
    pass


def path(run_dir: str) -> str:
    return os.path.join(os.path.abspath(run_dir), KILL_FILE)


def engage(run_dir: str, reason: str = "manual") -> str:
    os.makedirs(run_dir, exist_ok=True)
    p = path(run_dir)
    with open(p, "w") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {reason}\n")
    return p


def engaged_path(run_dir: str):
    """A KILL file in run_dir or ANY ancestor directory counts, so one file at the top of a
    run tree stops every engine below it."""
    d = os.path.abspath(run_dir)
    while True:
        if os.path.exists(os.path.join(d, KILL_FILE)):
            return os.path.join(d, KILL_FILE)
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def is_engaged(run_dir: str) -> bool:
    return engaged_path(run_dir) is not None


def check(run_dir: str):
    p = engaged_path(run_dir)
    if p:
        raise Killed(f"kill switch engaged: {p}")
