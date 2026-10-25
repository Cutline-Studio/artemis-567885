"""Watch (imitation), instruction rules, mistakes (error journal + DPO pairs), dictionary define,
new_md scaffolding and the self_author engine. CPU only."""
import json, os, shutil, subprocess, sys
import pytest, torch
from artemis_dense import DenseConfig, ArtemisDense, ImagePatchEncoder
from artemis_dense.engines import watch
from artemis_dense.engines.watch import redact, redact_image, ingest_terminal, ingest_action_log, imitate, match_rate, practice
from artemis_dense.engines.mistakes import MistakeJournal, dpo_loss
from artemis_dense.engines.skills import SkillStore, define
from artemis_dense.engines.code_exec import CodeExecEngine, SandboxLimits
from artemis_dense.engines.tools import ToolRouter
from artemis_dense.engines.self_train import SelfTrainer, TrainConfig, text_batch_fn
from artemis_dense.engines.autonomous import Orchestrator
from artemis_dense.data.instructions import extract

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = dict(vocab_size=256, context_len=128, n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128)
DEMO = """\
$ echo hello
hello
$ pwd
/work
$ ls
main.py
$ echo done
done
"""


# ---------------------------------------------------------------- watch
def test_redaction_secrets_and_pii():
    s = ("password=hunter2 token: abc123 ghp_" + "A" * 36 + " card 4111 1111 1111 1111 "
         "mail me at someone@example.com or 555-123-4567 and Bearer abcdefghijkl; order 1234567890123")
    r = redact(s)
    for leaked in ("hunter2", "abc123", "ghp_AAAA", "4111 1111", "someone@example.com", "555-123-4567", "abcdefghijkl"):
        assert leaked not in r, leaked
    assert "1234567890123" in r                       # not a Luhn-valid card number: kept
    img = torch.ones(3, 32, 32)
    out = redact_image(img, [(0, 0, 16, 8)])
    assert out[:, :8, :16].sum() == 0 and out[:, 8:, :].sum() > 0


def test_ingest_terminal_and_action_log():
    demo = ingest_terminal(DEMO)
    assert [s["action"]["value"] for s in demo] == ["echo hello", "pwd", "ls", "echo done"]
    assert demo[0]["outcome"] == "hello"
    log = ingest_action_log('{"type": "key", "value": "ctrl+s", "observation": "editor password=x"}\n'
                            '{"type": "tool", "value": "skills.lookup"}')
    assert log[0]["observation"]["text"] == "editor password=[REDACTED]" and log[1]["action"]["type"] == "tool"


def test_video_frontend_or_planned(tmp_path):
    if not shutil.which("ffmpeg"):
        with pytest.raises(NotImplementedError):
            watch.sample_frames("x.mp4")
        return
    vid = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=64x64:rate=5",
                    "-pix_fmt", "yuv420p", str(vid)], check=True, timeout=60)
    frames = watch.sample_frames(str(vid), every_s=1.0, size=32)
    assert frames.shape[1:] == (3, 32, 32) and 2 <= frames.shape[0] <= 4
    toks = watch.frames_to_tokens(frames, ImagePatchEncoder(64, img_size=32, patch=8))
    assert toks.shape == (1, frames.shape[0] * 16, 64)


def test_watch_learns_to_imitate_terminal_demo(capsys):
    torch.manual_seed(567885)
    demo = ingest_terminal(DEMO)
    m = ArtemisDense(DenseConfig(**TINY)).eval()
    before = match_rate(m, demo)
    losses = imitate(m, demo, steps=150, lr=3e-3, seed=567885)
    after = match_rate(m, demo)
    print(f"WATCH seed=567885 steps=150 loss[0]={losses[0]} loss[49]={losses[49]} loss[99]={losses[99]} "
          f"loss[-1]={losses[-1]} match_before={before} match_after={after}")
    assert losses[-1] < 0.2 * losses[0]
    assert before == 0.0 and after == 1.0


def test_practice_goes_through_router_permissions(tmp_path):
    demo = ingest_terminal(DEMO)[:2]
    ce = CodeExecEngine(str(tmp_path), SandboxLimits(timeout_s=5))
    policy = lambda s: s["action"]["value"]                      # perfect imitator
    trainer = SelfTrainer(str(tmp_path / "t"), DenseConfig(**TINY), TrainConfig(steps_per_cycle=1))
    ok = practice(demo, policy, ToolRouter(str(tmp_path), {"code_exec.run"}, code_exec=ce), trainer)
    assert ok["step_match_rate"] == 1.0 and ok["steps"][0]["outcome_match"] is True
    assert os.path.exists(tmp_path / "t" / "exec_signal.jsonl")
    denied = practice(demo, policy, ToolRouter(str(tmp_path), set(), code_exec=ce))   # no permission
    assert all(not s["ran"] and "not permitted" in s["denied"] for s in denied["steps"])


# ---------------------------------------------------------------- instructions
def test_instruction_extraction():
    md = ("x\n<!-- example -->\n**User:** What is GQA?\n**Artemis:** Grouped-query attention shares key/value heads.\n\n"
          "<!-- example -->\n**User:** (a question)\n**Artemis:** (an answer)\n")
    recs = extract(md, "t.md")
    assert recs == [{"instruction": "What is GQA?", "input": "",
                     "output": "Grouped-query attention shares key/value heads.", "source": "t.md"}]


def test_standing_rule_blocks_action(tmp_path):
    run = str(tmp_path)
    skills = SkillStore(os.path.join(run, "skills.json"))
    skills.add_rule("confirm-before-spend", "Confirm with Latoya before any spend", ["request_gpu", "*spend*"])
    skills.add_rule("no-growth-today", "Do not grow the model", ["force_grow"], effect="block")
    t = SelfTrainer(os.path.join(run, "train"), DenseConfig(**TINY), TrainConfig(steps_per_cycle=1))
    o = Orchestrator(run, t, skills, CodeExecEngine(run), approve_spend=True,
                     spend_handlers={"request_gpu": lambda: True})
    r = o.act("request_gpu")
    assert not r["ok"] and "confirmation" in r["detail"]          # blocked by the stored rule
    o.confirmed.add("request_gpu")
    assert o.act("request_gpu")["ok"]                             # confirmed + approved handler
    assert "blocked by rule" in o.act("force_grow")["detail"]


# ---------------------------------------------------------------- mistakes
def test_mistake_journal_caps_and_pairs(tmp_path):
    j = MistakeJournal(str(tmp_path))
    a = j.record("test_failure", "def add(a,b):", "return a-b", "AssertionError", "    return a+b",
                 test="assert add(2,3)==5")
    b = j.record("permission_denial", "user: rent a GPU now", "renting GPU", "costs money: refused",
                 "I need your approval before renting a GPU.")
    c = j.record("permission_denial", "user: rent a GPU now", "renting", "refused", "Just set approve_spend=True")
    d = j.record("user_correction", "Q: capital of France?", "Lyon", "user said wrong", "Paris")
    e = j.record("sandbox_error", "x", "y", "z", "use_bwrap=False to get network")
    assert [x["status"] for x in (a, b, c, d, e)] == ["fixed", "fixed", "quarantined", "fixed", "quarantined"]
    pairs = j.preference_pairs()
    assert {p["chosen"] for p in pairs} == {"    return a+b", "I need your approval before renting a GPU.", "Paris"}
    assert not any("approve_spend" in s.decode() or "use_bwrap" in s.decode() for s in j.corrective_samples())
    assert MistakeJournal(str(tmp_path)).entries[2]["status"] == "quarantined"   # persisted


def test_regression_check_reruns_fix(tmp_path):
    j = MistakeJournal(str(tmp_path))
    j.record("test_failure", "ctx", "def add(a,b): return a-b", "AssertionError", "def add(a,b): return a+b",
             test="assert add(2,3)==5")
    j.record("test_failure", "ctx", "x", "err", "def sub(a,b): return a+b", test="assert sub(5,3)==2")
    res = j.regression_check(CodeExecEngine(str(tmp_path)))
    assert [r["ok"] for r in res] == [True, False]
    assert [e["status"] for e in j.entries] == ["verified", "regressed"]


def test_mistake_replay_lowers_loss_and_dpo(tmp_path, capsys):
    torch.manual_seed(567885)
    j = MistakeJournal(str(tmp_path))
    for i in range(4):
        j.record("user_correction", f"Q{i}: what is 2+{i}?", f"{3 + i}", "wrong", f"A{i}: {2 + i}")
    m = ArtemisDense(DenseConfig(**TINY))
    samples = j.corrective_samples()
    T = max(len(s) for s in samples)
    X = torch.tensor([list(s.ljust(T, b" ")) for s in samples])
    with torch.no_grad():
        before = m(X[:, :-1], X[:, 1:])[1].item()
    batch_fn = j.replay_batch_fn(text_batch_fn(), weight=0.9)
    assert j.max_replay_fraction == 0.5                      # weight is capped
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(40):                                      # corrective smoke steps (CPU, seconds)
        _, loss = m(X[:, :-1], X[:, 1:]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        after = m(X[:, :-1], X[:, 1:])[1].item()
    print(f"MISTAKES seed=567885 steps=40 loss_before={before:.4f} loss_after={after:.4f}")
    assert after < 0.5 * before
    x, y = batch_fn(torch.Generator().manual_seed(0), 8, 32)
    assert x.shape == (8, 32)
    ref = ArtemisDense(DenseConfig(**TINY)).eval()
    assert torch.isfinite(dpo_loss(m, ref, j.preference_pairs()[:2]))


# ---------------------------------------------------------------- dictionary
DICT = os.path.join(REPO, "artemis_dense", "data", "dictionary", "dictionary.jsonl")


def test_define_small_fixture(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"word": "artemis", "pos": "noun", "definitions": ["Greek goddess of the hunt"],
                             "examples": [], "synonyms": [], "source": "test"}) + "\n")
    assert define("Artemis", str(p))[0]["definitions"] == ["Greek goddess of the hunt"]
    assert define("nope", str(p)) == []


@pytest.mark.skipif(not os.path.exists(DICT), reason="dictionary.jsonl not built (see Dictionary.md)")
def test_define_real_dictionary():
    recs = define("dictionary")
    assert {r["source"] for r in recs} >= {"gcide-0.54", "wordnet-3.1"}
    assert any("words" in d.lower() for r in recs for d in r["definitions"])
    r = ToolRouter("/tmp/_router_define", {"skills.define"})
    assert r.dispatch({"name": "skills.define", "args": {"word": "serendipity"}})[0]["pos"] == "noun"
