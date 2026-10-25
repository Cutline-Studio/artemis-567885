"""Tests for the five approved upgrades: scale ladder, tool heads, longer memory (GQA, RoPE
scaling, KV cache), self-scoring sample selection, multimodal encoders. CPU only, seconds."""
import json
import pytest, torch
from artemis_dense import (DenseConfig, ArtemisDense, expected_param_count, SCALES, TOOLS,
                           ImagePatchEncoder, AudioLogMelEncoder, log_mel)
from artemis_dense.engines.self_train import SelfTrainer, TrainConfig
from artemis_dense.engines.tools import ToolRouter, ToolDenied, parse_tool_call
from artemis_dense.engines.code_exec import CodeExecEngine, SandboxLimits
from artemis_dense.engines.skills import SkillStore
from artemis_dense.engines.a2a import (Identity, A2ANode, A2AClient, LoopbackTransport, OrgTrust, make_card,
                                       EnvironmentManager, AuditLog, EnvDenied)

TINY = dict(vocab_size=256, context_len=64, n_layers=2, d_model=32, n_heads=4, n_kv_heads=4, d_ff=64)
EXPECTED = {"tiny": 885_888, "small": 6_885_632, "medium": 36_319_744}


# ---------------------------------------------------------------- 1. scale ladder
@pytest.mark.parametrize("name", list(SCALES))
def test_scale_ladder_param_counts(name):
    cfg = DenseConfig.preset(name)
    m = ArtemisDense(cfg)
    assert m.num_params() == expected_param_count(cfg) == EXPECTED[name]


def test_scale_ladder_ordering_and_small_is_default():
    assert DenseConfig.preset("small") == DenseConfig()
    assert EXPECTED["tiny"] < EXPECTED["small"] < EXPECTED["medium"]


def test_tiny_preset_forward_cpu():
    m = ArtemisDense(DenseConfig.preset("tiny"))
    logits, _ = m(torch.randint(0, 256, (1, 64)))
    assert logits.shape == (1, 64, 256)


# ---------------------------------------------------------------- 3. longer memory
def test_gqa_equals_mha_when_kv_heads_equal_heads():
    torch.manual_seed(0)
    a = ArtemisDense(DenseConfig(**TINY)).eval()
    b = ArtemisDense(DenseConfig(**{**TINY, "n_kv_heads": 4})).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (2, 20))
    with torch.no_grad():
        assert torch.allclose(a(x)[0], b(x)[0])


def test_gqa_matches_mha_with_duplicated_kv_weights():
    """A GQA model (2 kv heads) equals an MHA model whose k/v heads are the GQA heads repeated."""
    torch.manual_seed(0)
    g = ArtemisDense(DenseConfig(**{**TINY, "n_kv_heads": 2})).eval()
    m = ArtemisDense(DenseConfig(**TINY)).eval()
    sd = g.state_dict()
    hd = TINY["d_model"] // TINY["n_heads"]
    for k, v in list(sd.items()):
        if k.endswith("attn.wk.weight") or k.endswith("attn.wv.weight"):
            sd[k] = v.view(2, hd, -1).repeat_interleave(2, 0).reshape(4 * hd, -1)
    m.load_state_dict(sd)
    x = torch.randint(0, 256, (2, 20))
    with torch.no_grad():
        assert torch.allclose(g(x)[0], m(x)[0], atol=1e-5)
    assert g.num_params() < m.num_params()


def test_gqa_causal_mask():
    m = ArtemisDense(DenseConfig(**{**TINY, "n_kv_heads": 1})).eval()
    x = torch.randint(0, 256, (1, 24)); x2 = x.clone(); x2[0, 10:] = (x2[0, 10:] + 7) % 256
    with torch.no_grad():
        assert torch.allclose(m(x)[0][0, :10], m(x2)[0][0, :10], atol=1e-6)


@pytest.mark.parametrize("kv", [4, 2, 1])
def test_kv_cache_matches_full_forward(kv):
    torch.manual_seed(0)
    m = ArtemisDense(DenseConfig(**{**TINY, "n_kv_heads": kv})).eval()
    x = torch.randint(0, 256, (2, 30))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache()
        a, _ = m(x[:, :12], cache=cache)                 # prefill
        steps = [a]
        for t in range(12, 30):                          # one token at a time
            s, _ = m(x[:, t:t + 1], cache=cache)
            steps.append(s)
        inc = torch.cat(steps, 1)
    assert cache.length == 30
    assert torch.allclose(full, inc, atol=1e-5)


def test_generate_with_and_without_cache_identical():
    torch.manual_seed(0)
    m = ArtemisDense(DenseConfig(**TINY)).eval()
    p = torch.randint(0, 256, (2, 5))
    assert torch.equal(m.generate(p, 20, use_cache=True), m.generate(p, 20, use_cache=False))


def test_rope_scaling_extends_context():
    base = DenseConfig(**TINY)
    long = DenseConfig(**{**TINY, "context_len": 256, "rope_scaling": 4.0})
    a, b = ArtemisDense(base), ArtemisDense(long)
    assert a.num_params() == b.num_params()             # scaling adds no parameters
    # position p in the scaled model uses the angle of position p/4 in the base model
    assert torch.allclose(b.rope_cos[40], a.rope_cos[10]) and torch.allclose(b.rope_sin[40], a.rope_sin[10])
    with pytest.raises(ValueError):
        a(torch.zeros(1, 65, dtype=torch.long))
    assert b(torch.zeros(1, 200, dtype=torch.long))[0].shape == (1, 200, 256)
    with pytest.raises(ValueError):
        DenseConfig(**{**TINY, "rope_scaling": 0.5})


# ---------------------------------------------------------------- 2. tool heads + routing
def test_tool_head_shape_and_params():
    cfg = DenseConfig(**TINY, tool_head=True, value_head=True)
    m = ArtemisDense(cfg)
    assert m.num_params() == expected_param_count(cfg)
    assert m.tool_logits(torch.randint(0, 256, (3, 9))).shape == (3, len(TOOLS))
    assert all(t in TOOLS for t in m.tool_call(torch.randint(0, 256, (3, 9))))


def test_parse_tool_call():
    assert parse_tool_call('ok <tool>{"name": "skills.lookup", "args": {"query": "x"}}</tool>') == \
        {"name": "skills.lookup", "args": {"query": "x"}}
    assert parse_tool_call("<tool>{not json}</tool>") is None
    assert parse_tool_call('<tool>{"name": "self_train.run_cycle"}</tool>') is None   # not a tool


@pytest.fixture
def router(tmp_path):
    skills = SkillStore(str(tmp_path / "skills.json"))
    skills.acquire("sum-list", "Sum numbers", "sum numbers", "steps", [])
    return ToolRouter(str(tmp_path), permissions={"skills.lookup", "code_exec.run"},
                      code_exec=CodeExecEngine(str(tmp_path), SandboxLimits(timeout_s=3)), skills=skills)


def test_router_default_deny_and_logging(router, tmp_path):
    assert router.dispatch({"name": "skills.lookup", "args": {"query": "sum"}})[0]["name"] == "sum-list"
    with pytest.raises(ToolDenied):
        router.dispatch({"name": "skills.acquire", "args": {"name": "x", "description": "d", "trigger": "t"}})
    with pytest.raises(ToolDenied):
        router.dispatch({"name": "self_train.run_cycle", "args": {}})
    log = [json.loads(l) for l in open(tmp_path / "tool_calls.jsonl")]
    assert [e["allowed"] for e in log] == [True, False, False]


def test_router_code_exec_is_sandboxed(router, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "should-not-leak")
    r = router.dispatch({"name": "code_exec.run", "args": {"code":
        "import os, socket\nprint(os.environ.get('VAST_API_KEY'))\n"
        "try:\n    socket.create_connection(('1.1.1.1', 80), timeout=2); print('NET')\nexcept OSError:\n    print('NONET')"}})
    assert r["ok"] and "should-not-leak" not in r["stdout"] and "NONET" in r["stdout"] and "rlimits" in r["sandbox"]


def test_router_env_calls_go_through_grants(tmp_path):
    me = Identity.generate("cutline", "artemis")
    envs = EnvironmentManager(me, str(tmp_path), AuditLog(str(tmp_path / "a.jsonl")), CodeExecEngine(str(tmp_path)))
    env = envs.define("ws", "workspace", str(tmp_path / "ws"))
    open(f"{env.root}/a.txt", "w").write("A")
    sid = envs.enter("local/artemis", "ws", envs.grant("local/artemis", "ws", ["read"]))
    r = ToolRouter(str(tmp_path), permissions={"env.read", "env.write"}, envs=envs)
    assert r.dispatch({"name": "env.read", "args": {"session": sid, "path": "a.txt"}}) == "A"
    with pytest.raises(EnvDenied):     # router permits env.write, but the grant doesn't: the gate wins
        r.dispatch({"name": "env.write", "args": {"session": sid, "path": "b.txt", "content": "x"}})
    with pytest.raises(EnvDenied):
        r.dispatch({"name": "env.read", "args": {"session": sid, "path": "../a.jsonl"}})


def test_router_a2a_goes_through_remote_gate(tmp_path):
    a, b = Identity.generate("cutline", "artemis"), Identity.generate("peer", "agent")
    node = A2ANode(b, str(tmp_path / "peer"), {"cutline": OrgTrust("cutline", a.public_key_b64, "basic")})
    net = LoopbackTransport(); net.register(node)
    cli = A2AClient(a, net, make_card(a, "artemis", "x", []))
    cli.handshake("peer", ["artemis.echo", "artemis.skills.lookup", "self_train.run_cycle"])
    r = ToolRouter(str(tmp_path), permissions={"a2a.send_task"}, a2a_client=cli)
    ok = r.dispatch({"name": "a2a.send_task", "args": {"org": "peer", "capability": "artemis.echo", "input": {"v": 1}}})
    assert ok["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    no = r.dispatch({"name": "a2a.send_task", "args": {"org": "peer", "capability": "self_train.run_cycle"}})
    assert no["task"]["status"]["state"] == "TASK_STATE_REJECTED"


# ---------------------------------------------------------------- 4. self-scoring + sample selection
def test_sequence_logprob_and_value_head():
    m = ArtemisDense(DenseConfig(**TINY, value_head=True))
    x = torch.randint(0, 256, (3, 16))
    lp = m.sequence_logprob(x)
    assert lp.shape == (3,) and (lp <= 0).all()
    assert m.score(x).shape == (3,)
    with pytest.raises(RuntimeError):
        ArtemisDense(DenseConfig(**TINY)).score(x)


def test_select_hard_picks_lowest_logprob():
    torch.manual_seed(0)
    m = ArtemisDense(DenseConfig(**TINY))
    easy = torch.zeros(4, 17, dtype=torch.long)                         # constant sequence
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(30):                                                 # tiny smoke fit (CPU, <1 s)
        _, loss = m(easy[:, :-1], easy[:, 1:]); opt.zero_grad(); loss.backward(); opt.step()
    hard = torch.randint(1, 256, (4, 17))
    pool = torch.cat([easy, hard])
    x, y, s = SelfTrainer.select_hard(m, pool[:, :-1], pool[:, 1:], 4)
    assert (x != 0).any(1).all()                                        # all 4 picks are the random ones
    assert s.shape == (4,)


def test_self_train_cycle_with_hard_selection(tmp_path):
    t = SelfTrainer(str(tmp_path), DenseConfig(**TINY),
                    TrainConfig(steps_per_cycle=3, batch_size=2, seq_len=16, sample_selection="hard", pool_factor=3))
    assert t.run_cycle()["status"] == "ok"


# ---------------------------------------------------------------- 5. multimodal
def test_image_encoder_shape():
    enc = ImagePatchEncoder(32, img_size=32, patch=8)
    assert enc(torch.randn(2, 3, 32, 32)).shape == (2, 16, 32)


def test_log_mel_and_audio_encoder_shape():
    mel = log_mel(torch.randn(2, 16000))                                # 1 s at 16 kHz
    assert mel.shape == (2, 80, 101) and torch.isfinite(mel).all()
    assert AudioLogMelEncoder(32)(mel).shape == (2, 51, 32)


def test_multimodal_forward_and_causality():
    torch.manual_seed(0)
    m = ArtemisDense(DenseConfig(**{**TINY, "context_len": 128})).eval()
    m.attach_encoder("image", ImagePatchEncoder(32, img_size=32, patch=8))
    m.attach_encoder("audio", AudioLogMelEncoder(32, n_mels=80))
    img, mel, txt = torch.randn(2, 3, 32, 32), torch.randn(2, 80, 40), torch.randint(0, 256, (2, 12))
    logits, loss = m.forward_multimodal(txt, txt, image=img, audio=mel)
    assert logits.shape == (2, 12, 256) and torch.isfinite(loss)
    loss.backward()
    assert m.encoders["image"].proj.weight.grad is not None and m.encoders["audio"].conv1.weight.grad is not None
    with torch.no_grad():                                                # the image changes text logits ...
        a, _ = m.forward_multimodal(txt, image=img, audio=mel)
        b, _ = m.forward_multimodal(txt, image=img + 1, audio=mel)
    assert not torch.allclose(a, b)
    assert m.num_params() == expected_param_count(m.cfg)                 # encoders counted separately
    assert m.num_params(include_encoders=True) > m.num_params()
    with pytest.raises(ValueError):
        m.attach_encoder("bad", ImagePatchEncoder(16))
