"""Skills engine: a JSON-on-disk registry of named, reusable procedures.

A skill = {name, description, trigger, kind: "steps"|"code"|"callable", body, stats}.
  steps    : list of engine actions, e.g. [{"action": "train_cycle"}, {"action": "evaluate"}]
  code     : Python source; reused by running it through engines/code_exec (sandboxed)
  callable : name registered in-process via SkillStore.register_callable (not persisted as code)
Stats: uses, successes, failures, score (Laplace-smoothed success rate), last_used.
"""
import json, os, re, time
from typing import Callable, Dict, List, Optional

_NAME = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


class SkillStore:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.skills: Dict[str, dict] = {}
        self._callables: Dict[str, Callable] = {}
        if os.path.exists(self.path):
            self.skills = json.load(open(self.path))["skills"]

    def _save(self):
        tmp = self.path + ".tmp"
        json.dump({"version": 1, "skills": self.skills}, open(tmp, "w"), indent=2)
        os.replace(tmp, self.path)

    # ---- acquire / store -----------------------------------------------------
    def acquire(self, name: str, description: str, trigger: str, kind: str, body,
                source: str = "manual", overwrite: bool = False) -> dict:
        if not _NAME.match(name):
            raise ValueError(f"bad skill name {name!r}")
        if kind not in ("steps", "code", "callable", "rule"):
            raise ValueError(f"bad kind {kind!r}")
        if name in self.skills and not overwrite:
            raise KeyError(f"skill {name!r} exists")
        self.skills[name] = {"name": name, "description": description, "trigger": trigger, "kind": kind,
                             "body": body, "source": source, "created": time.time(),
                             "stats": {"uses": 0, "successes": 0, "failures": 0, "score": 0.5, "last_used": None}}
        self._save()
        return self.skills[name]

    store = acquire

    # ---- standing rules (instruction learning) -------------------------------
    def add_rule(self, name: str, description: str, applies_to: List[str], effect: str = "require_confirmation",
                 overwrite: bool = False) -> dict:
        """A standing instruction the orchestrator checks before EVERY action.
        applies_to: glob patterns over action names; effect: "block" | "require_confirmation"."""
        if effect not in ("block", "require_confirmation"):
            raise ValueError(effect)
        return self.acquire(name, description, " ".join(applies_to), "rule",
                            {"applies_to": list(applies_to), "effect": effect}, source="instruction",
                            overwrite=overwrite)

    def rules(self) -> List[dict]:
        return [s for s in self.skills.values() if s["kind"] == "rule"]

    def check_rules(self, action: str, confirmed: bool = False) -> Optional[str]:
        """Return the reason an action is blocked by a standing rule, or None if allowed."""
        import fnmatch
        for r in self.rules():
            if any(fnmatch.fnmatchcase(action, p) for p in r["body"]["applies_to"]):
                if r["body"]["effect"] == "block":
                    return f"blocked by rule {r['name']}: {r['description']}"
                if not confirmed:
                    return f"needs human confirmation (rule {r['name']}: {r['description']})"
        return None

    def register_callable(self, name: str, fn: Callable):
        self._callables[name] = fn

    # ---- lookup ----------------------------------------------------------------
    def get(self, name: str) -> Optional[dict]:
        return self.skills.get(name)

    def lookup(self, query: str, k: int = 3) -> List[dict]:
        """Rank by keyword overlap with name/description/trigger, then by score."""
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked = []
        for s in self.skills.values():
            words = set(re.findall(r"[a-z0-9]+", f"{s['name']} {s['description']} {s['trigger']}".lower()))
            overlap = len(q & words)
            if overlap:
                ranked.append((overlap, s["stats"]["score"], s))
        ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [s for _, _, s in ranked[:k]]

    # ---- reuse + scoring -------------------------------------------------------
    def reuse(self, name: str, executor: Callable[[dict], bool]) -> bool:
        """Run a skill through `executor(skill) -> success` (the orchestrator supplies it)
        and update its stats. Callable skills run in-process."""
        s = self.skills[name]
        try:
            if s["kind"] == "callable":
                ok = bool(self._callables[s["body"]]())
            else:
                ok = bool(executor(s))
        except Exception:
            ok = False
        self.record_use(name, ok)
        return ok

    def record_use(self, name: str, success: bool):
        st = self.skills[name]["stats"]
        st["uses"] += 1
        st["successes" if success else "failures"] += 1
        st["score"] = round((st["successes"] + 1) / (st["uses"] + 2), 4)   # Laplace-smoothed
        st["last_used"] = time.time()
        self._save()


# ---------------------------------------------------------------- dictionary lookup tool
DEFAULT_DICTIONARY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "data", "dictionary", "dictionary.jsonl")


class Dictionary:
    """Exact-meaning lookup over dictionary.jsonl (built by data/build_dictionary.py).
    Builds an in-memory {word: [byte offsets]} index on first use; records are read on demand."""

    def __init__(self, path: str = DEFAULT_DICTIONARY):
        self.path = path
        self._index: Optional[Dict[str, List[int]]] = None

    def _build(self):
        idx: Dict[str, List[int]] = {}
        with open(self.path, "rb") as f:
            off = 0
            for line in f:
                w = json.loads(line)["word"]
                idx.setdefault(w, []).append(off)
                off += len(line)
        self._index = idx

    def define(self, word: str) -> List[dict]:
        if self._index is None:
            self._build()
        key = re.sub(r"\s+", " ", word.replace("_", " ")).strip().lower()
        out = []
        with open(self.path, "rb") as f:
            for off in self._index.get(key, []):
                f.seek(off)
                out.append(json.loads(f.readline()))
        return out


_DICT: Optional[Dictionary] = None


def define(word: str, path: Optional[str] = None) -> List[dict]:
    """Return every dictionary record for `word` (all parts of speech, all sources); [] if unknown."""
    global _DICT
    if path is not None:
        return Dictionary(path).define(word)
    if _DICT is None:
        _DICT = Dictionary()
    return _DICT.define(word)
