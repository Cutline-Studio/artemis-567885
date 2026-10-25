"""Autonomous orchestrator: picks the next action from state with a simple policy, logs every decision.

SAFETY BOUNDS (required; enforced in code, not just documented):
  1. Local engines only: actions are a fixed whitelist dispatched to SelfTrainer, SkillStore,
     CodeExecEngine methods. No subprocess/network calls of its own; code runs only via code_exec's sandbox.
  2. Budget: max_steps and max_seconds; the loop stops when either is hit.
  3. No network, no money, no GPU rental: there is no action that does any of these. Actions tagged
     costs_money=True are refused unless approve_spend=True (default False), and even then
     only an explicit human-supplied handler may run them (none is shipped).
  4. No file deletion outside its own run_dir: the orchestrator deletes nothing; every engine
     writes only under run_dir (asserted at construction).
  6. Standing rules: before every action the orchestrator checks rule-skills in the skills store
     (instruction learning); a matching "block" rule or an unconfirmed "require_confirmation"
     rule stops the action.
  5. Kill switch: a KILL file in run_dir (or any parent dir) stops the loop before the next step.
"""
import json, os, time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .self_train import SelfTrainer
from .skills import SkillStore
from .code_exec import CodeExecEngine
from .. import killswitch


@dataclass
class Action:
    name: str
    costs_money: bool = False


ACTIONS: Dict[str, Action] = {a.name: a for a in [
    Action("train_cycle"), Action("evaluate"), Action("force_grow"), Action("code_task"),
    Action("use_skill"), Action("record_skill"), Action("stop"), Action("author_doc"),
    Action("request_gpu", costs_money=True),     # placeholder: always refused without approval
]}


class Orchestrator:
    def __init__(self, run_dir: str, trainer: SelfTrainer, skills: SkillStore, code: CodeExecEngine,
                 max_steps: int = 10, max_seconds: float = 60.0, approve_spend: bool = False,
                 spend_handlers: Optional[Dict[str, Callable]] = None, author=None):
        self.run_dir = os.path.abspath(run_dir)
        for eng_dir in (trainer.run_dir, os.path.dirname(skills.path), code.run_dir):
            if not os.path.abspath(eng_dir).startswith(self.run_dir):
                raise ValueError(f"engine dir {eng_dir} is outside the orchestrator run_dir {self.run_dir}")
        self.trainer, self.skills, self.code = trainer, skills, code
        self.max_steps, self.max_seconds = max_steps, max_seconds
        self.approve_spend = approve_spend
        self.spend_handlers = spend_handlers or {}
        self.confirmed: set = set()          # action names a human confirmed for this run
        self.code_tasks: List[dict] = []     # {"task", "propose", "tests", "skill_name"}
        self.author = author                 # optional SelfAuthor
        self.author_tasks: List[dict] = []   # {"name", "content", "title"}
        self.log_path = os.path.join(self.run_dir, "decisions.jsonl")
        self.steps = 0
        self.ok_cycles_since_skill = 0

    # ---------------------------------------------------------------- policy
    def decide(self) -> (str, str):
        hist = self.trainer.state["history"]
        if self.code_tasks:
            return "code_task", f"{len(self.code_tasks)} code task(s) queued"
        if self.author_tasks and self.author is not None:
            return "author_doc", f"{len(self.author_tasks)} doc(s) queued for self_author"
        if len(hist) >= 2 and all(h["status"] == "rolled_back" for h in hist[-2:]):
            return "stop", "two consecutive rollbacks; stopping for human review"
        if self.ok_cycles_since_skill >= 2 and not self.skills.get("train-and-evaluate"):
            return "record_skill", "2 clean cycles; recording the train+eval procedure as a skill"
        best = self.skills.lookup("train evaluate cycle", k=1)
        if best and best[0]["stats"]["score"] >= 0.5 and best[0]["kind"] == "steps":
            return "use_skill", f"reusing skill {best[0]['name']} (score {best[0]['stats']['score']})"
        return "train_cycle", "default: train one cycle"

    # ---------------------------------------------------------------- dispatch
    def _exec_steps(self, skill: dict) -> bool:
        ok = True
        for st in skill["body"]:
            if st["action"] not in ("train_cycle", "evaluate"):
                return False                        # skills may only call whitelisted local steps
            ok = self.act(st["action"], via_skill=skill["name"])["ok"] and ok
        return ok

    def act(self, name: str, via_skill: Optional[str] = None, **kw) -> dict:
        if name not in ACTIONS:
            return {"ok": False, "detail": f"unknown action {name!r} refused"}
        a = ACTIONS[name]
        blocked = self.skills.check_rules(name, confirmed=name in self.confirmed)   # standing rules first
        if blocked:
            return {"ok": False, "detail": blocked}
        if a.costs_money:
            if not self.approve_spend or name not in self.spend_handlers:
                return {"ok": False, "detail": f"{name} costs money: refused (approve_spend={self.approve_spend})"}
            return {"ok": bool(self.spend_handlers[name](**kw)), "detail": "human-approved handler ran"}
        if name == "train_cycle":
            r = self.trainer.run_cycle()
            self.ok_cycles_since_skill += r["status"] == "ok"
            return {"ok": r["status"] == "ok", "detail": r}
        if name == "evaluate":
            ev = self.trainer.evaluate(self.trainer.load_model())
            return {"ok": ev == ev, "detail": {"eval_loss": ev}}
        if name == "force_grow":
            m = self.trainer.load_model()
            g = self.trainer.plan_growth(m.cfg)
            self.trainer.state["pending_growth"] = g
            self.trainer._save_state()
            return {"ok": g is not None, "detail": {"pending_growth": g}}
        if name == "code_task":
            t = self.code_tasks.pop(0)
            r = self.code.iterate(t["task"], t["propose"], t.get("tests", ""), t.get("max_iters", 4),
                                  skills=self.skills, trainer=self.trainer, skill_name=t.get("skill_name"))
            return {"ok": r["passed"], "detail": {k: r[k] for k in ("task", "passed", "iterations")}
                    | ({"skill_saved": r["skill_saved"]} if "skill_saved" in r else {})}
        if name == "author_doc":
            t = self.author_tasks.pop(0)
            r = self.author.create_doc(t["name"], t["content"], t.get("title"))
            return {"ok": bool(r.get("created")), "detail": r}
        if name == "record_skill":
            self.skills.acquire("train-and-evaluate", "Train one dense cycle, then evaluate",
                                "train evaluate cycle when loss is still improving", "steps",
                                [{"action": "train_cycle"}, {"action": "evaluate"}], source="autonomous")
            self.ok_cycles_since_skill = 0
            return {"ok": True, "detail": "recorded skill train-and-evaluate"}
        if name == "use_skill":
            sk = self.skills.lookup("train evaluate cycle", k=1)[0]
            ok = self.skills.reuse(sk["name"], self._exec_steps)
            return {"ok": ok, "detail": {"skill": sk["name"]}}
        if name == "stop":
            return {"ok": True, "detail": "stop"}
        return {"ok": False, "detail": "unhandled"}

    def _log(self, rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # ---------------------------------------------------------------- loop
    def run(self) -> List[dict]:
        t0, out = time.time(), []
        while self.steps < self.max_steps:
            if killswitch.is_engaged(self.run_dir):
                self._log({"step": self.steps, "action": "stop", "reason": "kill switch engaged"})
                break
            if time.time() - t0 > self.max_seconds:
                self._log({"step": self.steps, "action": "stop", "reason": "time budget exhausted"})
                break
            name, why = self.decide()
            res = self.act(name)
            rec = {"step": self.steps, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "action": name,
                   "reason": why, "ok": res["ok"], "detail": res["detail"]}
            self._log(rec); out.append(rec)
            self.steps += 1
            if name == "stop":
                break
        return out
