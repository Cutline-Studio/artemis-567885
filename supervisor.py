"""ArtemisSelfFixingSupervisor (+ matrix auto-correct) for the expand engine.

One supervised cycle:
  1. plan candidate mutation paths from the last cycle's metrics (expand.plan_mutations)
  2. for each candidate (max_attempts):
       a. back up config.json, runs/checkpoint.pt and every top-level *.py to .backup_checkpoint/
       b. apply the mutation to config.json (never to Python source)
       c. ast.parse every top-level *.py
       d. run `expand.py run-cycle` in a subprocess with a timeout
       e. on non-zero exit / timeout / NaN loss / output-shape mismatch:
            - parse stderr for the failing file:line and PyTorch shape errors,
              log to runs/supervisor_log.jsonl and runs/runtime_errors.log
            - if it is a shape error that maps to a config dimension, realign
              config.json to the checkpoint (the trained weights) and do ONE
              validation re-run; keep it if it passes
            - otherwise roll back everything from the backup and try the next
              (different) mutation path
  3. the no-op path is always last; if every path fails the supervisor stops.

CLI:  python supervisor.py --cycles 10 [--timeout 300] [--max-attempts 3] [--root R]
"""
import argparse, ast, copy, glob, json, math, os, re, shutil, subprocess, sys, time
from typing import Callable, List, Optional

import expand

SHAPE_PATTERNS = {
    "matmul": re.compile(r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)"),
    "broadcast": re.compile(r"The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)"),
    "view": re.compile(r"shape '\[([-\d, ]+)\]' is invalid for input of size (\d+)"),
    "view_stride": re.compile(r"view size is not compatible with input tensor's size and stride"),
    "state_dict": re.compile(r"size mismatch for ([\w.]+): copying a param with shape torch\.Size\(\[([\d, ]*)\]\) "
                             r"from checkpoint, the shape in current model is torch\.Size\(\[([\d, ]*)\]\)"),
    "divisible": re.compile(r"d_model \((\d+)\) must be divisible by n_heads \((\d+)\)"),
    "output_shape": re.compile(r"OutputShapeMismatch: output \(([\d, ]+)\) != input \(([\d, ]+)\)"),
    "shrink": re.compile(r"ShrinkNotSupported"),
}
NAN_PATTERN = re.compile(r"NaNLoss")
FRAME_PATTERN = re.compile(r'File "([^"]+)", line (\d+)')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def parse_failure(stderr: str) -> dict:
    frames = FRAME_PATTERN.findall(stderr or "")
    last_line = (stderr or "").strip().splitlines()[-1] if (stderr or "").strip() else ""
    shape = []
    for name, pat in SHAPE_PATTERNS.items():
        for m in pat.finditer(stderr or ""):
            shape.append({"kind": name, "groups": list(m.groups())})
    return {"file": frames[-1][0] if frames else None, "line": int(frames[-1][1]) if frames else None,
            "error": last_line[:500], "shape_errors": shape[:5], "nan": bool(NAN_PATTERN.search(stderr or ""))}


class ArtemisSelfFixingSupervisor:
    def __init__(self, root: str, timeout: float = 300.0, max_attempts: int = 3,
                 inject: Optional[Callable[[str, int, int], None]] = None, python: str = sys.executable):
        self.root = os.path.abspath(root)
        self.timeout, self.max_attempts, self.inject, self.python = timeout, max_attempts, inject, python
        self.backup_dir = os.path.join(self.root, ".backup_checkpoint")
        self.p = expand.paths(self.root)
        os.makedirs(os.path.join(self.root, "runs"), exist_ok=True)
        self.sup_log = os.path.join(self.root, "runs", "supervisor_log.jsonl")
        self.err_log = os.path.join(self.root, "runs", "runtime_errors.log")

    # ------------------------------------------------------------ backup / rollback
    def _py_files(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.root, "*.py")))

    def backup(self):
        if os.path.exists(self.backup_dir):
            shutil.rmtree(self.backup_dir)
        os.makedirs(self.backup_dir)
        manifest = {"config": False, "ckpt": False, "py": []}
        shutil.copy2(self.p["config"], os.path.join(self.backup_dir, "config.json")); manifest["config"] = True
        if os.path.exists(self.p["ckpt"]):
            shutil.copy2(self.p["ckpt"], os.path.join(self.backup_dir, "checkpoint.pt")); manifest["ckpt"] = True
        for f in self._py_files():
            shutil.copy2(f, os.path.join(self.backup_dir, os.path.basename(f))); manifest["py"].append(os.path.basename(f))
        with open(os.path.join(self.backup_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh)

    def rollback(self) -> dict:
        with open(os.path.join(self.backup_dir, "manifest.json")) as fh:
            manifest = json.load(fh)
        shutil.copy2(os.path.join(self.backup_dir, "config.json"), self.p["config"])
        if manifest["ckpt"]:
            shutil.copy2(os.path.join(self.backup_dir, "checkpoint.pt"), self.p["ckpt"])
        elif os.path.exists(self.p["ckpt"]):
            os.remove(self.p["ckpt"])
        current = {os.path.basename(f) for f in self._py_files()}
        for name in manifest["py"]:
            shutil.copy2(os.path.join(self.backup_dir, name), os.path.join(self.root, name))
        for extra in current - set(manifest["py"]):
            os.remove(os.path.join(self.root, extra))  # .py files created by the failed mutation
        # the failed cycle may have appended a partial line before dying; keep the log consistent
        return {"restored": ["config.json"] + (["runs/checkpoint.pt"] if manifest["ckpt"] else []) + manifest["py"]}

    def backup_config(self) -> dict:
        with open(os.path.join(self.backup_dir, "config.json")) as fh:
            return json.load(fh)

    # ------------------------------------------------------------ checks
    def syntax_check(self) -> Optional[dict]:
        for f in self._py_files():
            try:
                with open(f) as fh:
                    ast.parse(fh.read(), filename=f)
            except SyntaxError as e:
                return {"file": f, "line": e.lineno, "error": f"SyntaxError: {e.msg}"}
        return None

    def run_subprocess(self) -> dict:
        cmd = [self.python, os.path.join(self.root, "expand.py"), "run-cycle", "--root", self.root]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            return {"ok": False, "reason": "timeout", "stderr": f"TimeoutExpired after {self.timeout}s",
                    "seconds": round(time.time() - t0, 2)}
        res = {"ok": r.returncode == 0, "returncode": r.returncode, "stderr": r.stderr[-8000:],
               "seconds": round(time.time() - t0, 2)}
        if r.returncode != 0:
            res["reason"] = "nonzero_exit"
            return res
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            res.update(ok=False, reason="no_result")
            return res
        out = json.loads(line[-1][len("RESULT "):])
        res["result"] = out
        if not out.get("output_shape_ok") or any(not math.isfinite(out[k]) for k in ("loss", "aux")):
            res.update(ok=False, reason="nan_or_shape")
        return res

    # ------------------------------------------------------------ auto-correct
    def auto_correct(self, failure: dict) -> Optional[dict]:
        """Realign config.json to the trained checkpoint for shape errors. Returns the fix or None."""
        if not failure["shape_errors"]:
            return None
        cfg = expand.load_config(self.root)
        before = self.backup_config()
        changed = {k: (before.get(k), cfg.get(k)) for k in expand.DIM_KEYS + ("num_layers", "num_experts")
                   if before.get(k) != cfg.get(k)}
        ref = None
        if os.path.exists(self.p["ckpt"]):
            import torch
            ref = torch.load(self.p["ckpt"], weights_only=False)["arch"]
            ref_src = "checkpoint"
        else:
            ref, ref_src = before, "pre-mutation config"
        fix = {}
        for k in expand.DIM_KEYS:
            if ref.get(k) is not None and cfg.get(k) != ref[k]:
                fix[k] = (cfg.get(k), ref[k])
                cfg[k] = ref[k]
        if ref_src == "checkpoint":
            if cfg["num_layers"] < ref["num_layers"]:
                fix["num_layers"] = (cfg["num_layers"], ref["num_layers"]); cfg["num_layers"] = ref["num_layers"]
            cur, rne = list(cfg["num_experts"]), ref["num_experts"]
            ne = [max(cur[i] if i < len(cur) else cfg["base_num_experts"], rne[i] if i < len(rne) else 0)
                  for i in range(cfg["num_layers"])]
            if ne != cfg["num_experts"]:
                fix["num_experts"] = (cfg["num_experts"], ne); cfg["num_experts"] = ne
        if cfg["d_model"] % cfg["n_heads"] != 0:
            h = max(d for d in range(1, cfg["n_heads"] + 1) if cfg["d_model"] % d == 0)
            fix["n_heads"] = (cfg["n_heads"], h); cfg["n_heads"] = h
        if not fix:
            return None
        save = copy.deepcopy(cfg)
        expand.save_config(self.root, save)
        return {"realigned": {k: {"from": v[0], "to": v[1]} for k, v in fix.items()}, "reference": ref_src,
                "changed_this_cycle": {k: {"from": v[0], "to": v[1]} for k, v in changed.items()},
                "weights_action": ("kept checkpoint weights; config dimensions reverted to the checkpoint "
                                   "(a dimension change cannot load trained weights as-is)") if ref_src == "checkpoint"
                else "no checkpoint yet; config realigned, model will be freshly initialised"}

    # ------------------------------------------------------------ logging
    def log(self, event: dict):
        event = {"time": now(), **event}
        with open(self.sup_log, "a") as fh:
            fh.write(json.dumps(event) + "\n")
        return event

    def log_runtime_error(self, cycle: int, attempt: int, mutation: str, text: str):
        with open(self.err_log, "a") as fh:
            fh.write(f"=== {now()} cycle={cycle} attempt={attempt} mutation={mutation} ===\n{text.rstrip()}\n")

    # ------------------------------------------------------------ main loop
    def supervise_cycle(self) -> dict:
        cfg = expand.load_config(self.root)
        cycle = cfg["cycle"]
        cands = expand.plan_mutations(cfg)
        tried = []
        for attempt, mut in enumerate(cands[:self.max_attempts] if len(cands) <= self.max_attempts
                                      else cands[:self.max_attempts - 1] + [cands[-1]]):
            self.backup()
            expand.save_config(self.root, expand.apply_mutation(expand.load_config(self.root), mut))
            if self.inject:
                self.inject(self.root, cycle, attempt)
            tried.append(mut["name"])
            syn = self.syntax_check()
            if syn:
                self.log_runtime_error(cycle, attempt, mut["name"], f'File "{syn["file"]}", line {syn["line"]}\n{syn["error"]}')
                rb = self.rollback()
                self.log({"event": "rollback", "cycle": cycle, "attempt": attempt, "mutation": mut["name"],
                          "reason": "syntax_error", "failure": syn, **rb})
                continue
            res = self.run_subprocess()
            if res["ok"]:
                return self.log({"event": "cycle_ok", "cycle": cycle, "attempt": attempt, "mutation": mut,
                                 "result": res["result"], "seconds": res["seconds"]})
            failure = parse_failure(res.get("stderr", ""))
            failure["reason"] = res.get("reason")
            self.log_runtime_error(cycle, attempt, mut["name"], res.get("stderr", ""))
            self.log({"event": "failure", "cycle": cycle, "attempt": attempt, "mutation": mut["name"], "failure": failure})
            fix = self.auto_correct(failure) if res.get("reason") != "timeout" else None
            if fix:
                self.log({"event": "autocorrect", "cycle": cycle, "attempt": attempt, **fix})
                val = self.run_subprocess()
                if val["ok"]:
                    return self.log({"event": "cycle_ok", "cycle": cycle, "attempt": attempt, "mutation": mut,
                                     "autocorrected": True, "result": val["result"], "seconds": val["seconds"]})
                vf = parse_failure(val.get("stderr", ""))
                self.log_runtime_error(cycle, attempt, mut["name"] + "+autocorrect", val.get("stderr", ""))
                self.log({"event": "autocorrect_failed", "cycle": cycle, "attempt": attempt, "failure": vf})
            rb = self.rollback()
            self.log({"event": "rollback", "cycle": cycle, "attempt": attempt, "mutation": mut["name"],
                      "reason": failure.get("reason"), **rb})
        return self.log({"event": "cycle_failed", "cycle": cycle, "tried": tried})

    def run(self, cycles: int) -> List[dict]:
        out = []
        for _ in range(cycles):
            ev = self.supervise_cycle()
            out.append(ev)
            if ev["event"] == "cycle_failed":
                break
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-attempts", type=int, default=3)
    a = ap.parse_args(argv)
    if not os.path.exists(expand.paths(a.root)["config"]):
        expand.init_config(a.root)
    sup = ArtemisSelfFixingSupervisor(a.root, timeout=a.timeout, max_attempts=a.max_attempts)
    sup.last_event = None
    for ev in sup.run(a.cycles):
        sup.last_event = ev
        r = ev.get("result", {})
        print(json.dumps({"event": ev["event"], "cycle": ev["cycle"], "attempt": ev.get("attempt"),
                          "mutation": (ev.get("mutation") or {}).get("name") if isinstance(ev.get("mutation"), dict) else None,
                          "autocorrected": ev.get("autocorrected", False), **{k: r.get(k) for k in
                          ("loss", "aux", "dropped", "capacity_factor", "alpha", "num_experts", "num_layers",
                           "grad_x_norm", "grad_gate_norm")}}))
    return 1 if sup.last_event and sup.last_event["event"] == "cycle_failed" else 0


if __name__ == "__main__":
    sys.exit(main())
