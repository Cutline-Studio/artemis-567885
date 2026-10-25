"""sandbox.env loader + constitutional promotion gate wiring."""
import os, subprocess
import pytest

from artemis_dense.sandbox import SandboxEnv, SandboxViolation, parse_env
from artemis_dense.promotion import PromotionGate, PromotionBlocked
from artemis_dense.engines.code_exec import CodeExecEngine
from artemis_dense.engines.custom_constitutional import ConstitutionalEngine, load_env


@pytest.fixture
def sb(tmp_path):
    s = SandboxEnv.load(ARTEMIS_SANDBOX_ROOT=str(tmp_path / "sbx"))
    with s.session():
        yield s


def gate_for(sb, tmp_path):
    env = load_env()
    d = str(tmp_path / "cai")
    env.update(ARTEMIS_CAI_LOG=d + "/checks.jsonl", ARTEMIS_CAI_ALERT_FILE=d + "/alerts.jsonl",
               ARTEMIS_CAI_DASHBOARD=d + "/dashboard.md")
    return PromotionGate(sandbox=sb, engine=ConstitutionalEngine(env))


def test_loader_parses_and_expands():
    v = parse_env(open(os.path.join(os.path.dirname(__file__), "..", "artemis_dense", "sandbox.env")).read())
    assert v["ARTEMIS_UNTRUSTED"] == "1" and v["ARTEMIS_WORKDIR"] == "/tmp/artemis-sandbox/work"
    assert v["ARTEMIS_SANDBOX_LABEL"].startswith("UNTRUSTED")
    with pytest.raises(SandboxViolation):
        SandboxEnv.load(ARTEMIS_SANDBOX_ROOT="/workspace/x")


def test_limits_and_denials(sb):
    L = sb.limits()
    assert L.use_bwrap is True and L.mem_bytes <= 2048 * 1024 * 1024 and L.nproc == 32
    for cap in ("push", "payments", "email", "deploy", "gpu_rental"):
        with pytest.raises(SandboxViolation):
            sb.deny(cap)
    with pytest.raises(SandboxViolation):
        sb.check_path("/workspace/anything")
    with pytest.raises(SandboxViolation):
        CodeExecEngine("/tmp/not-in-sandbox-run", sandbox=sb)


def test_sandboxed_run_cannot_read_workspace_or_reach_network(sb):
    ex = CodeExecEngine(sb.work_dir("exec"), sandbox=sb)
    ex.limits.ro_paths = ("/workspace", "/home/box")          # even if someone asks, denied paths aren't bound
    r = ex.run("import os\nprint(os.path.exists('/workspace'), os.path.exists('/home/box'), os.path.exists('/cursor'))\n"
               "print(os.environ.get('ARTEMIS_UNTRUSTED'), os.environ.get('ARTEMIS_NETWORK'))\n"
               "import _socket\ns = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)\ns.settimeout(3)\n"
               "try:\n    s.connect(('1.1.1.1', 80)); print('NET OK')\nexcept OSError as e:\n    print('NET BLOCKED', e)\n")
    assert r.sandbox == "bwrap+rlimits", r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "False False False"
    assert lines[1] == "1 off"
    assert lines[2].startswith("NET BLOCKED")
    events = [l for l in open(sb.vars["ARTEMIS_AUDIT_LOG"])]
    assert any('"code_exec_run"' in e for e in events)


def test_wipe_on_exit(tmp_path):
    s = SandboxEnv.load(ARTEMIS_SANDBOX_ROOT=str(tmp_path / "w"))
    with s.session():
        open(os.path.join(s.work_dir(), "f"), "w").write("x")
    assert not os.path.exists(s.root)


def test_gate_loose_then_strict_on_self_modify(sb, tmp_path):
    g = gate_for(sb, tmp_path)
    staged = os.path.join(sb.work_dir("s"), "a.md")
    open(staged, "w").write("x")
    v = g.require("doc", text="I am human", artifact_path=staged, paths=["artemis_dense/a.md"])
    assert v.outcome == "warn"                                 # loose: warn only, still promotable
    with pytest.raises(PromotionBlocked):
        g.require("doc", text="ok", artifact_path=staged, paths=["artemis_dense/sandbox.env"])
    with pytest.raises(PromotionBlocked):                      # stays strict for the session
        g.require("doc", text="ok", artifact_path=staged, paths=["artemis_dense/b.md"])
    g.engine.approve_resume("Latoya")
    assert g.require("doc", text="ok", artifact_path=staged, paths=["artemis_dense/b.md"]).outcome == "ok"


def _repo(tmp_path):
    import shutil, sys
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = tmp_path / "repo"
    (r / "artemis_dense").mkdir(parents=True)
    shutil.copytree(os.path.join(REPO, "scripts"), r / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    for c in (["init", "-q"], ["config", "user.name", "t"], ["config", "user.email", "t@t"]):
        subprocess.run(["git", "-C", str(r), *c], check=True)
    return str(r)


def test_self_author_promotion_gate(sb, tmp_path):
    from artemis_dense.engines.self_author import SelfAuthor
    repo = _repo(tmp_path)
    a = SelfAuthor(repo, str(tmp_path / "run"), gate=gate_for(sb, tmp_path))
    assert a.create_doc("Fine", "# Fine\n")["created"]
    with pytest.raises(PromotionBlocked):                      # writing sandbox.env trips strict mode
        a.write_file("artemis_dense/sandbox.env", "ARTEMIS_NETWORK=on\n")
    assert not os.path.exists(os.path.join(repo, "artemis_dense", "sandbox.env"))
    with pytest.raises(PromotionBlocked):                      # and nothing else promotes until approve_resume
        a.create_doc("After", "# After\n")
    a.gate.engine.approve_resume("Latoya")
    assert a.create_doc("After", "# After\n")["created"]


def test_self_train_promotion(sb, tmp_path):
    from artemis_dense.config import DenseConfig
    from artemis_dense.engines.self_train import SelfTrainer, TrainConfig
    from tests.test_dense_engines import TINY, TC
    t = SelfTrainer(sb.work_dir("train"), DenseConfig(**TINY), TrainConfig(**TC))
    t.run_cycle()
    g = gate_for(sb, tmp_path)
    with pytest.raises(PermissionError):
        t.promote(str(tmp_path / "promoted"), gate=g)          # needs human approval
    out = t.promote(str(tmp_path / "promoted"), approved=True, gate=g)
    assert os.path.exists(out["promoted"])


def test_watch_practice_promotion(sb, tmp_path):
    from artemis_dense.engines.watch import practice
    from artemis_dense.engines.skills import SkillStore
    from artemis_dense.engines.tools import ToolRouter
    ex = CodeExecEngine(sb.work_dir("exec2"), sandbox=sb)
    router = ToolRouter(sb.work_dir("router"), permissions={"code_exec.run"}, code_exec=ex)
    demo = [{"obs": "say hi", "action": {"type": "command", "value": "echo hi"}, "outcome": "hi"}]
    skills = SkillStore(str(tmp_path / "skills.json"))
    score = practice(demo, lambda s: "echo hi", router, skills=skills, skill_name="say-hi", gate=gate_for(sb, tmp_path))
    assert score["step_match_rate"] == 1.0 and score["promotion"]["allowed"]
    assert skills.get("say-hi")["source"] == "watch"
