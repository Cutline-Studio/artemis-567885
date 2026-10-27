import json, os, shutil
import pytest
import torch
import expand
import supervisor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = {"steps_per_cycle": 15, "batch_size": 4, "seq_len": 8, "eval_batch_size": 4}


@pytest.fixture
def repo(tmp_path):
    for f in ("moe.py", "transformer.py", "model.py", "positional.py", "expand.py", "supervisor.py", "sandbox.py"):
        shutil.copy(os.path.join(ROOT, f), tmp_path / f)
    expand.init_config(str(tmp_path), overrides=FAST)
    return tmp_path


def events(repo):
    return [json.loads(l) for l in open(repo / "runs" / "supervisor_log.jsonl")]


def sup(repo, inject=None):
    return supervisor.ArtemisSelfFixingSupervisor(str(repo), timeout=120, max_attempts=3, inject=inject)


def test_parse_failure_regexes():
    err = ('Traceback (most recent call last):\n  File "/x/moe.py", line 12, in forward\n'
           'RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x16 and 512x4)\n')
    f = supervisor.parse_failure(err)
    assert f["file"] == "/x/moe.py" and f["line"] == 12
    assert f["shape_errors"][0] == {"kind": "matmul", "groups": ["128", "16", "512", "4"]}
    f = supervisor.parse_failure("RuntimeError: The size of tensor a (16) must match the size of tensor b (512) at non-singleton dimension 2")
    assert f["shape_errors"][0]["kind"] == "broadcast"
    f = supervisor.parse_failure("RuntimeError: shape '[4, 8, 3, 5]' is invalid for input of size 512")
    assert f["shape_errors"][0]["kind"] == "view"


def test_syntax_error_mutation_rolls_back_then_next_path_succeeds(repo):
    original = (repo / "model.py").read_text()

    def inject(root, cycle, attempt):
        if cycle == 0 and attempt == 0:
            with open(os.path.join(root, "model.py"), "a") as fh:
                fh.write("\ndef broken(:\n")

    s = sup(repo, inject)
    # cycle 0 has only the no-op path; give it a second path by forcing two attempts
    expand.plan_mutations.__globals__  # noqa: keep reference
    orig_plan = expand.plan_mutations
    expand.plan_mutations = lambda cfg: [{"name": "bad", "changes": {}, "reasons": []},
                                         {"name": "none", "changes": {}, "reasons": []}]
    try:
        ev = s.supervise_cycle()
    finally:
        expand.plan_mutations = orig_plan
    assert ev["event"] == "cycle_ok" and ev["attempt"] == 1
    assert (repo / "model.py").read_text() == original
    evs = events(repo)
    rb = [e for e in evs if e["event"] == "rollback"]
    assert rb and rb[0]["reason"] == "syntax_error" and rb[0]["failure"]["file"].endswith("model.py")
    assert "SyntaxError" in (repo / "runs" / "runtime_errors.log").read_text()
    assert s.supervise_cycle()["event"] == "cycle_ok"  # next cycle also fine


def test_output_shape_mismatch_rolls_back_and_tries_different_path(repo):
    s = sup(repo)
    assert s.supervise_cycle()["event"] == "cycle_ok"  # cycle 0 creates a checkpoint
    original = (repo / "model.py").read_text()

    def inject(root, cycle, attempt):
        if attempt == 0:
            p = os.path.join(root, "model.py")
            src = open(p).read().replace("        return x, aux_total, dropped_total",
                                         "        return x[..., :-1], aux_total, dropped_total")
            open(p, "w").write(src)

    s.inject = inject
    ev = s.supervise_cycle()
    assert ev["event"] == "cycle_ok" and ev["attempt"] >= 1
    assert (repo / "model.py").read_text() == original
    evs = events(repo)
    fail = [e for e in evs if e["event"] == "failure"][0]
    assert any(se["kind"] == "output_shape" for se in fail["failure"]["shape_errors"])
    assert [e for e in evs if e["event"] == "rollback"]
    tried = {e["mutation"] for e in evs if e["event"] in ("failure", "rollback")}
    assert ev["mutation"]["name"] not in tried  # a different path succeeded


def test_nan_loss_rolls_back(repo):
    s = sup(repo)
    assert s.supervise_cycle()["event"] == "cycle_ok"

    def inject(root, cycle, attempt):
        if attempt == 0:
            ck_path = os.path.join(root, "runs", "checkpoint.pt")
            ck = torch.load(ck_path, weights_only=False)
            ck["model"]["blocks.0.attn.q_linear.weight"].fill_(float("nan"))  # poisoned weights
            torch.save(ck, ck_path)

    s.inject = inject
    ev = s.supervise_cycle()
    assert ev["event"] == "cycle_ok" and ev["attempt"] >= 1
    ck = torch.load(repo / "runs" / "checkpoint.pt", weights_only=False)
    assert all(torch.isfinite(v).all() for v in ck["model"].values())  # rolled back to clean weights
    fail = [e for e in events(repo) if e["event"] == "failure"][0]
    assert fail["failure"]["nan"]


def test_d_model_drift_is_detected_autocorrected_validated_and_continues(repo):
    s = sup(repo)
    assert s.supervise_cycle()["event"] == "cycle_ok"  # checkpoint trained at d_model=16
    ck_before = torch.load(repo / "runs" / "checkpoint.pt", weights_only=False)["arch"]
    assert ck_before["d_model"] == 16

    def inject(root, cycle, attempt):
        if cycle == 1 and attempt == 0:
            c = expand.load_config(root); c["d_model"] = 512; expand.save_config(root, c)

    s.inject = inject
    ev = s.supervise_cycle()
    assert ev["event"] == "cycle_ok" and ev.get("autocorrected") is True
    assert expand.load_config(str(repo))["d_model"] == 16
    evs = events(repo)
    fail = [e for e in evs if e["event"] == "failure"][0]
    assert any(se["kind"] == "state_dict" for se in fail["failure"]["shape_errors"])
    ac = [e for e in evs if e["event"] == "autocorrect"][0]
    assert ac["realigned"]["d_model"] == {"from": 512, "to": 16}
    assert ac["reference"] == "checkpoint" and "kept checkpoint weights" in ac["weights_action"]
    log = (repo / "runs" / "runtime_errors.log").read_text()
    assert "cycle=1 attempt=0" in log and "size mismatch" in log
    s.inject = None
    assert s.supervise_cycle()["event"] == "cycle_ok"  # continues normally
    assert expand.load_config(str(repo))["cycle"] == 3


def test_heads_not_dividing_d_model_is_realigned(repo):
    s = sup(repo)
    assert s.supervise_cycle()["event"] == "cycle_ok"

    def inject(root, cycle, attempt):
        if attempt == 0:
            c = expand.load_config(root); c["n_heads"] = 3; expand.save_config(root, c)

    s.inject = inject
    ev = s.supervise_cycle()
    assert ev["event"] == "cycle_ok" and ev.get("autocorrected") is True
    assert expand.load_config(str(repo))["n_heads"] == 2


def test_layer_addition_failure_is_rolled_back(repo):
    s = sup(repo)
    assert s.supervise_cycle()["event"] == "cycle_ok"
    orig_plan = expand.plan_mutations

    def plan(cfg):
        return [{"name": "add_layer", "changes": {"num_layers": 2, "num_experts": [4, 4]}, "reasons": []},
                {"name": "none", "changes": {}, "reasons": []}]

    def inject(root, cycle, attempt):
        if attempt == 0:  # sabotage add_layer so the new block has a mismatched width
            p = os.path.join(root, "model.py")
            src = open(p).read().replace("blk = MoETransformerBlock(self.d_model,", "blk = MoETransformerBlock(self.d_model * 2,")
            open(p, "w").write(src)

    s.inject = inject
    expand.plan_mutations = plan
    try:
        ev = s.supervise_cycle()
    finally:
        expand.plan_mutations = orig_plan
    assert ev["event"] == "cycle_ok" and ev["mutation"]["name"] == "none"
    assert expand.load_config(str(repo))["num_layers"] == 1
    assert torch.load(repo / "runs" / "checkpoint.pt", weights_only=False)["arch"]["num_layers"] == 1
