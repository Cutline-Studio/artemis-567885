import json, math, os, time
import torch, pytest
from artemis_dense import DenseConfig, ArtemisDense, killswitch
from artemis_dense.engines.self_train import SelfTrainer, TrainConfig
from artemis_dense.engines.skills import SkillStore
from artemis_dense.engines.code_exec import CodeExecEngine, SandboxLimits
from artemis_dense.engines.autonomous import Orchestrator

TINY = dict(vocab_size=256, context_len=32, n_layers=1, d_model=32, n_heads=4, n_kv_heads=4, d_ff=64,
            caps={"num_layers_max": 2, "d_ff_max": 96})
TC = dict(steps_per_cycle=4, batch_size=4, seq_len=16, d_ff_step=32)


# ---------------------------------------------------------------- growth
def test_add_layer_preserves_outputs():
    torch.manual_seed(0)
    m = ArtemisDense(DenseConfig(**TINY)).eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        a, _ = m(x); m.add_layer(); b, _ = m(x)
    assert m.num_layers == 2 and torch.allclose(a, b, atol=1e-6)


def test_widen_ffn_preserves_outputs():
    torch.manual_seed(0)
    m = ArtemisDense(DenseConfig(**TINY)).eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        a, _ = m(x); m.widen_ffn(96); b, _ = m(x)
    assert m.cfg.d_ff == 96 and torch.allclose(a, b, atol=1e-6)


def test_self_train_grows_within_caps(tmp_path):
    t = SelfTrainer(str(tmp_path), DenseConfig(**TINY), TrainConfig(**TC, stall_rel_improvement=1.1))  # always "stalled"
    recs = [t.run_cycle() for _ in range(4)]
    assert all(r["status"] == "ok" for r in recs)
    assert [r["n_layers"] for r in recs] == [1, 2, 2, 2]
    assert [r["d_ff"] for r in recs] == [64, 64, 96, 96]       # layers first, then width, then caps hold
    assert recs[-1]["planned_growth"] is None
    assert os.path.exists(tmp_path / "ckpt.pt")


def test_rollback_on_injected_nan(tmp_path):
    t = SelfTrainer(str(tmp_path), DenseConfig(**TINY), TrainConfig(**TC))
    ok = t.run_cycle()
    good = torch.load(tmp_path / "ckpt.pt", weights_only=False)["model"]
    t.inject = lambda m, c, s, loss: loss * float("nan") if (c == 1 and s == 2) else loss
    r = t.run_cycle()
    assert ok["status"] == "ok" and r["status"] == "rolled_back" and "non-finite" in r["reason"]
    after = torch.load(tmp_path / "ckpt.pt", weights_only=False)["model"]
    assert all(torch.equal(good[k], after[k]) for k in good)    # checkpoint untouched
    assert t.run_cycle()["status"] == "ok"                      # recovers next cycle


def test_rollback_on_regression(tmp_path):
    t = SelfTrainer(str(tmp_path), DenseConfig(**TINY), TrainConfig(**TC, regression_tol=0.0))
    t.run_cycle()
    t.state["best_eval"] = 1e-6                                  # anything is a regression now
    r = t.run_cycle()
    assert r["status"] == "rolled_back" and "regression" in r["reason"]


def test_kill_switch_stops_trainer(tmp_path):
    t = SelfTrainer(str(tmp_path), DenseConfig(**TINY), TrainConfig(**TC))
    killswitch.engage(str(tmp_path), "test")
    with pytest.raises(killswitch.Killed):
        t.run_cycle()


# ---------------------------------------------------------------- skills
def test_skill_store_roundtrip_and_reuse(tmp_path):
    p = str(tmp_path / "skills.json")
    s = SkillStore(p)
    s.acquire("sum-list", "Sum a list of ints", "sum numbers list", "steps", [{"action": "evaluate"}])
    s2 = SkillStore(p)                                           # reload from disk
    assert s2.get("sum-list")["description"] == "Sum a list of ints"
    assert s2.lookup("please sum these numbers")[0]["name"] == "sum-list"
    assert s2.reuse("sum-list", lambda sk: True) is True
    assert s2.reuse("sum-list", lambda sk: False) is False
    st = SkillStore(p).get("sum-list")["stats"]
    assert (st["uses"], st["successes"], st["failures"], st["score"]) == (2, 1, 1, 0.5)
    s2.register_callable("always", lambda: True)
    s2.acquire("cb", "callable skill", "cb", "callable", "always")
    assert s2.reuse("cb", lambda sk: False) is True
    with pytest.raises(KeyError):
        s2.acquire("cb", "dup", "x", "steps", [])
    with pytest.raises(ValueError):
        s2.acquire("../evil", "bad", "x", "steps", [])


# ---------------------------------------------------------------- code_exec
@pytest.fixture(params=["auto", "subprocess"])
def ce(request, tmp_path):
    return CodeExecEngine(str(tmp_path), SandboxLimits(timeout_s=2, cpu_s=2,
                                                       use_bwrap=None if request.param == "auto" else False))


def test_hello_world_and_streams(ce):
    r = ce.run("import sys\nprint('hello world')\nprint('oops', file=sys.stderr)")
    assert r.ok and r.stdout.strip() == "hello world" and r.stderr.strip() == "oops"


def test_timeout_infinite_loop(ce):
    t0 = time.time()
    r = ce.run("while True:\n    pass")
    assert not r.ok and (r.timed_out or r.returncode < 0) and time.time() - t0 < 5


def test_network_blocked(ce):
    r = ce.run("import socket\ns = socket.create_connection(('1.1.1.1', 80), timeout=2)\nprint('connected')")
    assert not r.ok and "connected" not in r.stdout


def test_secret_env_not_visible(ce, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-should-not-leak")
    r = ce.run("import os\nprint(sorted(os.environ))\nprint(os.environ.get('VAST_API_KEY'))")
    assert r.ok and "should-not-leak" not in r.stdout and "VAST_API_KEY" not in r.stdout
    assert "GITHUB_TOKEN" not in r.stdout


def test_output_cap(tmp_path):
    ce = CodeExecEngine(str(tmp_path), SandboxLimits(max_output=100))
    r = ce.run("print('x' * 10000)")
    assert r.truncated and len(r.stdout) == 100


def test_bwrap_hides_host_fs(tmp_path):
    ce = CodeExecEngine(str(tmp_path))
    if not ce.bwrap:
        pytest.skip("bubblewrap not available")
    r = ce.run("import os\nprint(os.path.exists('/home'), os.path.exists('/workspace'))")
    assert r.ok and r.stdout.strip() == "False False" and r.sandbox == "bwrap+rlimits"


def test_iterate_fixes_failing_snippet(tmp_path):
    ce = CodeExecEngine(str(tmp_path))
    skills = SkillStore(str(tmp_path / "skills.json"))
    trainer = SelfTrainer(str(tmp_path / "train"), DenseConfig(**TINY), TrainConfig(**TC))
    attempts = ["def add(a, b):\n    return a - b\n",            # deliberately wrong
                "def add(a, b):\n    return a + b\n"]           # scripted fix after seeing the failure

    def model(task, history):
        if history:
            assert "AssertionError" in history[-1]["stderr"]    # it observed the failure
        return attempts[len(history)]
    out = ce.iterate("write add(a, b)", model, tests="assert add(2, 3) == 5\nprint('PASS')",
                     max_iters=3, skills=skills, trainer=trainer, skill_name="add-two-numbers")
    assert out["passed"] and out["iterations"] == 2 and "PASS" in out["stdout"]
    assert skills.get("add-two-numbers")["kind"] == "code"
    sig = [json.loads(l) for l in open(out["signal_path"])]
    assert sig[-1]["passed"] is True and sig[-1]["iterations"] == 2


# ---------------------------------------------------------------- orchestrator
def _orch(tmp_path, **kw):
    run = str(tmp_path)
    t = SelfTrainer(os.path.join(run, "train"), DenseConfig(**TINY), TrainConfig(**TC))
    return Orchestrator(run, t, SkillStore(os.path.join(run, "skills", "skills.json")),
                        CodeExecEngine(run), **kw)


def test_orchestrator_runs_n_steps_within_budget(tmp_path):
    o = _orch(tmp_path, max_steps=5, max_seconds=60)
    o.code_tasks.append({"task": "print 42", "propose": lambda t, h: "print(42)", "tests": "",
                         "skill_name": "print-42"})
    t0 = time.time()
    recs = o.run()
    assert len(recs) == 5 and time.time() - t0 < 60
    acts = [r["action"] for r in recs]
    assert acts[0] == "code_task" and "record_skill" in acts and acts[-1] == "use_skill"
    log = [json.loads(l) for l in open(tmp_path / "decisions.jsonl")]
    assert len(log) == 5 and all("reason" in r for r in log)


def test_orchestrator_refuses_money_without_approval(tmp_path):
    o = _orch(tmp_path)
    r = o.act("request_gpu")
    assert not r["ok"] and "refused" in r["detail"]
    assert not o.act("delete_everything")["ok"]


def test_orchestrator_time_budget_and_kill_switch(tmp_path):
    o = _orch(tmp_path, max_steps=100, max_seconds=0.0)
    assert o.run() == []
    o2 = _orch(tmp_path, max_steps=100)
    killswitch.engage(str(tmp_path))
    recs = o2.run()
    assert recs == [] and "kill switch" in open(tmp_path / "decisions.jsonl").read()


def test_orchestrator_rejects_engine_outside_run_dir(tmp_path):
    t = SelfTrainer(str(tmp_path / "a" / "train"), DenseConfig(**TINY), TrainConfig(**TC))
    with pytest.raises(ValueError):
        Orchestrator(str(tmp_path / "b"), t, SkillStore(str(tmp_path / "b" / "s.json")), CodeExecEngine(str(tmp_path / "b")))
