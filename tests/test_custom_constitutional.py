import json

import pytest

from artemis_dense.engines.custom_constitutional import ConstitutionalEngine, load_env

SANDBOX = {"ARTEMIS_ENV": "sandbox", "ARTEMIS_UNTRUSTED": "1"}


@pytest.fixture
def eng(tmp_path):
    env = load_env()
    env.update({"ARTEMIS_CAI_LOG": str(tmp_path / "checks.jsonl"),
                "ARTEMIS_CAI_ALERT_FILE": str(tmp_path / "alerts.jsonl"),
                "ARTEMIS_CAI_DASHBOARD": str(tmp_path / "dashboard.md")})
    return lambda sid=None: ConstitutionalEngine(env=env, session_id=sid)


def test_defaults_are_loose():
    env = load_env()
    assert env["ARTEMIS_CAI_MODE"] == "warn" and env["STRICT_MODE"] == "false"


def test_clean_ok(eng):
    v = eng().check("def add(a, b): return a + b", runtime_env=SANDBOX)
    assert v.allowed and v.outcome == "ok"


def test_warning_does_not_block(eng):
    v = eng().check("I am human", runtime_env=SANDBOX)
    assert v.allowed and v.outcome == "warn"


def test_more_than_three_warnings_triggers_strict(eng):
    e = eng()
    for _ in range(3):
        assert e.check("I am human", runtime_env=SANDBOX).allowed
    v = e.check("I am human", runtime_env=SANDBOX)
    assert v.blocked and v.outcome == "strict_block" and not e.can_promote()


@pytest.mark.parametrize("kw", [
    {"paths": ["/workspace/secret.txt"]},
    {"action": "network"},
    {"action": "spend"},
    {"text": "-----BEGIN OPENSSH PRIVATE KEY-----"},
    {"paths": ["artemis_dense/custom_constitutional_env/constitution.yaml"]},
    {"paths": ["artemis_dense/sandbox.env"]},
])
def test_single_critical_triggers_strict(eng, kw):
    v = eng().check(runtime_env=SANDBOX, **kw)
    assert v.blocked and v.strict


def test_strict_needs_human_to_resume(eng):
    e = eng()
    e.check(action="spend", runtime_env=SANDBOX)
    assert e.check("hello", runtime_env=SANDBOX).blocked
    e.approve_resume("Latoya")
    assert e.check("hello", runtime_env=SANDBOX).allowed


def test_logging_alerts_and_redaction(eng, tmp_path):
    e = eng("s1")
    e.check("key " + "a" * 64, runtime_env=SANDBOX)
    rows = [json.loads(l) for l in (tmp_path / "checks.jsonl").read_text().splitlines()]
    r = rows[0]
    assert {"ts", "session", "principle", "trigger", "level"} <= r.keys()
    assert r["level"] == "strict_block" and "a" * 64 not in r["trigger"] and "[REDACTED]" in r["trigger"]
    alerts = (tmp_path / "alerts.jsonl").read_text()
    assert "strict_mode_activated" in alerts


def test_dashboard_tracks_sessions(eng, tmp_path):
    for i, n in enumerate([1, 2, 3]):
        e = eng(f"s{i}")
        for _ in range(n):
            e.check("I am human", runtime_env=SANDBOX)
    md = eng().build_dashboard().read_text()
    assert "By session" in md and "s0" in md and "s2" in md
    assert "Escalation trend" in md
