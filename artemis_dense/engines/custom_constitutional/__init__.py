"""Custom Constitutional AI engine for Artemis.

A capability of the one dense LLM, not a separate model. Config is in
artemis_dense/custom.env.

- Default mode is loose: violations are warned and logged, never blocked.
- Conditional strict mode turns on when behaviour gets "too far out of hand":
  more than N warnings in one session, any critical violation (production
  files, network, payments, secrets), or self-modification touching the
  constitution or sandbox. Strict mode blocks promotion until a human calls
  approve_resume().
- Every check is logged (timestamp, session, principle, redacted trigger,
  level), alerts go to a separate file, and build_dashboard() summarizes
  trends across sessions.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_PKG = Path(__file__).resolve().parents[2]  # artemis_dense/
_REPO = _PKG.parent


def _expand(value: str, env: dict) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), ""), value)


def load_env(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or _PKG / "custom.env")
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = _expand(v.strip(), env)
    return env


_SECRET_RE = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)"),
    re.compile(r"\b(sk|nvapi|ghp|xai)-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b[0-9a-f]{64}\b"),
]


def redact(text: str) -> str:
    for r in _SECRET_RE:
        text = r.sub("[REDACTED]", text)
    return text


@dataclass
class Violation:
    principle: str
    level: str  # "warn" or "critical"
    reason: str


@dataclass
class Verdict:
    allowed: bool
    outcome: str  # "ok", "warn", "strict_block"
    strict: bool
    violations: list[Violation] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return not self.allowed


class ConstitutionalEngine:
    def __init__(self, env: dict | None = None, session_id: str | None = None):
        self.env = env or load_env()
        cpath = Path(self.env["ARTEMIS_CAI_CONSTITUTION"])
        if not cpath.is_absolute():
            cpath = _REPO / cpath
        self.constitution = yaml.safe_load(cpath.read_text())
        self.mode = self.env.get("ARTEMIS_CAI_MODE", "warn")
        self.strict = self.env.get("STRICT_MODE", "false").lower() == "true"
        self.max_warnings = int(self.env.get("ARTEMIS_CAI_ESCALATE_MAX_WARNINGS", "3"))
        self.on_critical = self.env.get("ARTEMIS_CAI_ESCALATE_ON_CRITICAL", "1") == "1"
        self.on_self_modify = self.env.get("ARTEMIS_CAI_ESCALATE_ON_SELF_MODIFY", "1") == "1"
        self.needs_human = self.env.get("ARTEMIS_CAI_STRICT_REQUIRES_HUMAN", "1") == "1"
        self.snippet = int(self.env.get("ARTEMIS_CAI_TRIGGER_SNIPPET_CHARS", "200"))
        self.redact_logs = self.env.get("ARTEMIS_CAI_REDACT_IN_LOGS", "1") == "1"
        self.log_path = Path(self.env["ARTEMIS_CAI_LOG"])
        self.alerts_on = self.env.get("ARTEMIS_CAI_ALERTS_ENABLED", "1") == "1"
        self.alert_path = Path(self.env["ARTEMIS_CAI_ALERT_FILE"])
        self.dashboard_path = Path(self.env["ARTEMIS_CAI_DASHBOARD"])
        self.retention_days = int(self.env.get("ARTEMIS_CAI_RETENTION_DAYS", "30"))
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.session_warnings = 0
        self.strict_reason: str | None = "STRICT_MODE=true in custom.env" if self.strict else None

    # ---- checks ---------------------------------------------------------
    def check(self, text: str = "", action: str | None = None, approved: bool = False,
              paths: list[str] | None = None, runtime_env: dict | None = None) -> Verdict:
        runtime_env = os.environ if runtime_env is None else runtime_env
        paths = paths or []
        found: list[Violation] = []

        for p in self.constitution.get("principles", []):
            lvl = p.get("level", "warn")
            if text and any(re.search(pat, text) for pat in p.get("patterns", [])):
                found.append(Violation(p["id"], lvl, "pattern match"))
            if action and action in p.get("actions", []) and not approved:
                found.append(Violation(p["id"], lvl, f"action '{action}' without approval"))
            req = p.get("requires_env")
            if req and any(str(runtime_env.get(k)) != str(v) for k, v in req.items()):
                found.append(Violation(p["id"], lvl, "not running in sandbox.env"))

        for c in self.constitution.get("critical", []):
            hit = None
            if text and any(re.search(pat, text) for pat in c.get("patterns", [])):
                hit = "pattern match"
            if action and action in c.get("actions", []):
                hit = f"action '{action}'"
            for pth in paths:
                if any(str(Path(pth)).startswith(pre) for pre in c.get("path_prefixes", [])):
                    hit = f"path {pth}"
                if any(re.search(pat, pth) for pat in c.get("path_patterns", [])):
                    hit = f"path {pth}"
            if hit:
                found.append(Violation(c["id"], "critical", hit))

        warns = [v for v in found if v.level == "warn"]
        crits = [v for v in found if v.level == "critical"]
        self.session_warnings += len(warns)

        newly = None
        if not self.strict:
            if self.session_warnings > self.max_warnings:
                newly = f"{self.session_warnings} warnings in session (> {self.max_warnings})"
            selfmod = [v for v in crits if v.principle == "self_modify_guardrails"]
            other_crit = [v for v in crits if v.principle != "self_modify_guardrails"]
            if self.on_critical and other_crit:
                newly = f"critical: {other_crit[0].principle}"
            if self.on_self_modify and selfmod:
                newly = "self-modification of constitution or sandbox"
            if newly:
                self.strict = True
                self.strict_reason = newly

        if self.strict:
            verdict = Verdict(False, "strict_block", True, found)
        elif found:
            verdict = Verdict(True, "warn", False, found)
        else:
            verdict = Verdict(True, "ok", False, found)

        self._log(verdict, text, action, paths)
        if newly:
            self._alert("strict_mode_activated", newly)
        elif verdict.outcome == "strict_block":
            self._alert("strict_block", self.strict_reason or "")
        return verdict

    def can_promote(self) -> bool:
        return not self.strict

    def approve_resume(self, approver: str) -> None:
        """Only a human calls this, after reviewing why strict mode tripped."""
        if not approver:
            raise ValueError("approver required")
        self._alert("strict_mode_cleared", f"approved by {approver}; was: {self.strict_reason}")
        self.strict = False
        self.strict_reason = None
        self.session_warnings = 0

    # ---- monitoring -----------------------------------------------------
    def _trigger(self, text: str, action: str | None, paths: list[str]) -> str:
        parts = []
        if action:
            parts.append(f"action={action}")
        if paths:
            parts.append("paths=" + ",".join(paths))
        if text:
            t = redact(text) if self.redact_logs else text
            parts.append("output=" + t[: self.snippet])
        return " | ".join(parts)

    def _write(self, path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _log(self, verdict: Verdict, text: str, action: str | None, paths: list[str]) -> None:
        base = {"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "session": self.session_id, "outcome": verdict.outcome,
                "strict": verdict.strict, "trigger": self._trigger(text, action, paths)}
        if not verdict.violations:
            self._write(self.log_path, {**base, "principle": None, "level": "ok"})
        for v in verdict.violations:
            level = "strict_block" if verdict.strict else v.level
            self._write(self.log_path, {**base, "principle": v.principle,
                                        "level": level, "reason": v.reason})

    def _alert(self, kind: str, detail: str) -> None:
        if self.alerts_on:
            self._write(self.alert_path, {"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                          "session": self.session_id, "alert": kind, "detail": detail})

    def prune(self) -> int:
        """Drop log and alert records older than the retention window."""
        cutoff = time.time() - self.retention_days * 86400
        dropped = 0
        for p in (self.log_path, self.alert_path):
            if not p.exists():
                continue
            keep = []
            for line in p.read_text().splitlines():
                try:
                    if json.loads(line)["ts"] >= cutoff:
                        keep.append(line)
                    else:
                        dropped += 1
                except (ValueError, KeyError):
                    keep.append(line)
            p.write_text("\n".join(keep) + ("\n" if keep else ""))
        return dropped

    def build_dashboard(self) -> Path:
        """Write a markdown trend report across all sessions in the log."""
        rows = []
        if self.log_path.exists():
            rows = [json.loads(l) for l in self.log_path.read_text().splitlines() if l.strip()]
        by_day: dict[str, Counter] = defaultdict(Counter)
        by_session: dict[str, Counter] = defaultdict(Counter)
        by_principle: Counter = Counter()
        for r in rows:
            day = time.strftime("%Y-%m-%d", time.localtime(r["ts"]))
            by_day[day][r["level"]] += 1
            by_session[r["session"]][r["level"]] += 1
            if r.get("principle"):
                by_principle[r["principle"]] += 1
        alerts = []
        if self.alert_path.exists():
            alerts = [json.loads(l) for l in self.alert_path.read_text().splitlines() if l.strip()]
        sess_order = sorted(by_session, key=lambda s: min(r["ts"] for r in rows if r["session"] == s))
        warn_series = [by_session[s]["warn"] + by_session[s]["critical"] + by_session[s]["strict_block"]
                       for s in sess_order]
        escalating = len(warn_series) >= 3 and all(a < b for a, b in zip(warn_series[-3:], warn_series[-2:]))
        out = ["# Artemis constitutional monitoring", "",
               f"Checks logged: {len(rows)}, sessions: {len(by_session)}, alerts: {len(alerts)}", "",
               f"Escalation trend (violations rising over the last 3 sessions): {'YES' if escalating else 'no'}", "",
               "## By day", "", "| Day | ok | warn | critical | strict_block |", "|---|---|---|---|---|"]
        for d in sorted(by_day):
            c = by_day[d]
            out.append(f"| {d} | {c['ok']} | {c['warn']} | {c['critical']} | {c['strict_block']} |")
        out += ["", "## By session (oldest first)", "", "| Session | ok | warn | critical | strict_block |",
                "|---|---|---|---|---|"]
        for s in sess_order:
            c = by_session[s]
            out.append(f"| {s} | {c['ok']} | {c['warn']} | {c['critical']} | {c['strict_block']} |")
        out += ["", "## Top principles", "", "| Principle | Hits |", "|---|---|"]
        for p, n in by_principle.most_common():
            out.append(f"| {p} | {n} |")
        out += ["", "## Recent alerts", ""]
        out += [f"- {a['iso']} [{a['session']}] {a['alert']}: {a['detail']}" for a in alerts[-20:]] or ["- none"]
        self.dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        self.dashboard_path.write_text("\n".join(out) + "\n")
        return self.dashboard_path
