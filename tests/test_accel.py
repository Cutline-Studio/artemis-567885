"""Tests for artemis_dense/accel: one or more CPU tests per accelerated-learning method (1-20).
Approval-needed methods are tested for REFUSAL (and for their offline code paths), never run."""
import json, os

import pytest, torch
import torch.nn.functional as F

from artemis_dense.config import DenseConfig
from artemis_dense.model import ArtemisDense
from artemis_dense.accel import METHODS, PRESETS, AccelConfig
from artemis_dense.accel import data as D, device as dev, distill, growth, lora, pretrained, replay, schedule, selfplay

TINY = dict(vocab_size=256, context_len=64, n_layers=2, d_model=32, n_heads=4, n_kv_heads=4, d_ff=64)


def tiny(seed=0, **o):
    torch.manual_seed(seed)
    return ArtemisDense(DenseConfig(**{**TINY, **o}))


def batch(B=4, T=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, (B, T + 1), generator=g)
    return x[:, :-1], x[:, 1:]


def text_batch(T=32, B=4):
    s = (b"the cat sat on the mat. " * 200)
    d = torch.tensor(list(s))
    ix = torch.arange(B) * 7
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


def test_registry_has_all_20_methods_with_status():
    ids = {int(m[0]) for m in METHODS}
    assert ids == set(range(1, 21))
    assert {m[3] for m in METHODS} <= {"built+tested", "approval-needed", "planned"}
    assert {m[0] for m in METHODS if m[3] == "approval-needed"} == {1, 5, 9}
    for name in PRESETS:
        AccelConfig.preset(name)


# 1 ------------------------------------------------------------------------------------------
def test_1_gpu_refused_without_approval():
    assert dev.resolve_device("cpu").type == "cpu"
    for d in ("cuda", "auto"):
        with pytest.raises(dev.ApprovalRequired):
            dev.resolve_device(d, gpu_approved=False)


# 2 ------------------------------------------------------------------------------------------
def test_2_bf16_autocast_cpu():
    m = tiny(); x, y = batch()
    with dev.autocast("bf16", torch.device("cpu")):
        logits, loss = m(x, y)
    assert logits.dtype == torch.bfloat16 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


# 3 ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["math", "flash", "efficient"])
def test_3_attention_backends_match_default(kind):
    m = tiny().eval(); x, _ = batch()
    ref, _ = m(x)
    try:
        with dev.attention_ctx(kind):
            out, _ = m(x)
    except RuntimeError:
        pytest.skip(f"{kind} kernel not available on this build")
    assert torch.allclose(ref, out, atol=1e-4)


# 4 ------------------------------------------------------------------------------------------
def test_4_compile_flag_and_fallback():
    m = tiny()
    assert dev.maybe_compile(m, False) == (m, "off")
    def bad(_):
        raise RuntimeError("no compiler")
    fm, st = dev.maybe_compile(m, True, probe=bad)
    assert fm is m and st.startswith("fallback-eager")


# 5 ------------------------------------------------------------------------------------------
def test_5_ddp_refused_and_launcher_command():
    with pytest.raises(dev.ApprovalRequired):
        dev.setup_ddp(tiny(), approved=False)
    cmd = dev.torchrun_command(2, ["--preset", "gpu_full", "--run", "x y"])
    assert cmd.startswith("torchrun --standalone --nproc_per_node=2 -m artemis_dense.accel.trainer")
    assert "'x y'" in cmd
    with pytest.raises(dev.ApprovalRequired):
        from artemis_dense.accel.trainer import run
        run(AccelConfig.preset("baseline", ddp=True), 0.01, "t", "/tmp/accel_t")


# 6 ------------------------------------------------------------------------------------------
def test_6_byte_codec_roundtrip():
    c = D.get_codec("byte")
    assert c.decode(c.encode("Artemis é")) == "Artemis é"


def test_6_bpe_hook(tmp_path):
    pytest.importorskip("tokenizers")
    try:
        p = D.train_bpe("hello artemis world. " * 300 + "<|user|>hi<|end|>", str(tmp_path), vocab_size=320)
    except ImportError:
        pytest.skip("artemis_dense/tokenizer.py not present yet")
    c = D.get_codec("bpe", p)
    s = "hello artemis world."
    assert c.decode(c.encode(s)) == s and len(c.encode(s)) < len(s.encode())
    assert len(c.encode("<|user|>")) == 1


# 7 ------------------------------------------------------------------------------------------
def test_7_packing_fills_every_slot():
    docs = [list(range(1, n + 1)) for n in (5, 17, 9, 40, 3)]
    rows = D.pack_documents(docs, T=15, sep=[0])
    assert rows.shape[1] == 16 and rows.numel() <= sum(len(d) + 1 for d in docs)
    recs = [([1] * n, [2] * n) for n in (10, 20, 5, 30)]
    X, Y, fill = D.pack_sft(recs, T=32)
    padded_fill = sum(n for n in (10, 20, 5, 30)) / (4 * 32)
    assert fill > padded_fill and X.shape == Y.shape


# 8 ------------------------------------------------------------------------------------------
def test_8_seq_len_warmup_schedule():
    Ts = [D.seq_len_at(f / 100, 256, 64, 0.3) for f in range(101)]
    assert Ts[0] == 64 and Ts[-1] == 256 and all(a <= b for a, b in zip(Ts, Ts[1:]))
    assert D.seq_len_at(0.5, 256, 64, 0.0) == 256


# 9 ------------------------------------------------------------------------------------------
def _hf_dir(tmp_path, m):
    cfg = m.cfg
    sd = {"model.embed_tokens.weight": m.embed.weight, "model.norm.weight": m.norm.weight}
    for i, b in enumerate(m.blocks):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = pretrained.interleaved_to_hf(b.attn.wq.weight, cfg.n_heads)
        sd[p + "self_attn.k_proj.weight"] = pretrained.interleaved_to_hf(b.attn.wk.weight, cfg.n_kv_heads)
        sd[p + "self_attn.v_proj.weight"] = b.attn.wv.weight
        sd[p + "self_attn.o_proj.weight"] = b.attn.wo.weight
        sd[p + "mlp.gate_proj.weight"] = b.ffn.w_gate.weight
        sd[p + "mlp.up_proj.weight"] = b.ffn.w_up.weight
        sd[p + "mlp.down_proj.weight"] = b.ffn.w_down.weight
        sd[p + "input_layernorm.weight"] = b.attn_norm.weight
        sd[p + "post_attention_layernorm.weight"] = b.ffn_norm.weight
    torch.save({k: v.detach().clone() for k, v in sd.items()}, tmp_path / "pytorch_model.bin")
    json.dump({"hidden_size": cfg.d_model, "num_attention_heads": cfg.n_heads, "num_key_value_heads": cfg.n_kv_heads,
               "vocab_size": cfg.vocab_size, "num_hidden_layers": cfg.n_layers, "intermediate_size": cfg.d_ff,
               "max_position_embeddings": cfg.context_len, "rope_theta": cfg.rope_theta,
               "rms_norm_eps": cfg.norm_eps, "tie_word_embeddings": True}, open(tmp_path / "config.json", "w"))


def test_9_pretrained_refused_without_approval_and_license(tmp_path):
    with pytest.raises(pretrained.ApprovalRequired):
        pretrained.load_local(str(tmp_path))
    with pytest.raises(pretrained.ApprovalRequired):
        pretrained.load_local(str(tmp_path), approved=True, license_ack="whatever")
    assert set(pretrained.LICENSES) == {"llama3", "qwen2.5", "mistral"}


def test_9_local_hf_layout_loads_function_identically(tmp_path):
    src = tiny(3, n_kv_heads=2).eval()
    _hf_dir(tmp_path, src)
    m = pretrained.load_local(str(tmp_path), approved=True, license_ack="mistral", context_len=64).eval()
    x, _ = batch()
    assert torch.allclose(src(x)[0], m(x)[0], atol=1e-5)
    assert m.trained_info["from_scratch"] is False


# 10 -----------------------------------------------------------------------------------------
def test_10_kd_loss_zero_for_identical_and_distillation_reduces_kl():
    l = torch.randn(2, 5, 256)
    assert distill.kd_loss(l, l).abs() < 1e-5
    teacher = tiny(1, n_layers=4, d_model=64, d_ff=128).eval()        # bigger random-init local teacher
    student = tiny(2)
    x, y = batch(8, 32, 5)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-3)
    with torch.no_grad():
        k0 = distill.kd_loss(student(x)[0], teacher(x)[0]).item()
    for _ in range(40):
        loss, _ = distill.distill_step_loss(student, teacher, x, y, alpha=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        k1 = distill.kd_loss(student(x)[0], teacher(x)[0]).item()
    assert k1 < 0.5 * k0
    with pytest.raises(RuntimeError):
        distill.teacher_generate()


# 11 -----------------------------------------------------------------------------------------
def test_11_growth_is_function_preserving_and_training_continues():
    m = tiny(4); x, y = text_batch()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(20):
        loss = m(x, y)[1]; opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        before = m(x, y)[1].item()
    ev = growth.grow(m, add_layers=1, new_d_ff=96, generator=torch.Generator().manual_seed(0))
    assert ev["layers"] == [2, 3] and ev["d_ff"] == [64, 96]
    with torch.no_grad():
        after = m(x, y)[1].item()
    assert abs(before - after) < 1e-5
    opt = growth.rebuild_optimizer(opt, m)
    assert len(opt.state) > 0                      # surviving params kept their Adam state
    for _ in range(20):
        loss = m(x, y)[1]; opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < after


# 12 -----------------------------------------------------------------------------------------
def test_12_lora_trains_only_adapters_and_merges():
    m = tiny(5).eval(); x, y = text_batch()
    ref = m(x)[0].detach()
    n = lora.apply_lora(m, r=4, alpha=8, targets=("wq", "wv"))
    assert n == 4 and torch.allclose(m(x)[0], ref, atol=1e-6)
    ps = lora.lora_parameters(m)
    assert {id(p) for p in m.parameters() if p.requires_grad} == {id(p) for p in ps}
    base0 = m.blocks[0].ffn.w_up.weight.clone()
    opt = torch.optim.AdamW(ps, lr=1e-2)
    l0 = m(x, y)[1].item()
    m.train()
    for _ in range(30):
        loss = m(x, y)[1]; opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < l0 and torch.equal(base0, m.blocks[0].ffn.w_up.weight)
    m.eval(); out = m(x)[0].detach()
    assert lora.merge_lora(m) == 4
    assert torch.allclose(m(x)[0], out, atol=1e-4)


# 13 -----------------------------------------------------------------------------------------
def test_13_quality_filter():
    good = "Artemis is one dense transformer. Every token uses every weight, and the sandbox runs tests."
    junk = ["$$$ ### @@@ %%% ^^^ &&& *** ((( )))", "spam line\n" * 30, "ok"]
    assert D.quality_filter([good] + junk) == [good]


# 14 -----------------------------------------------------------------------------------------
def test_14_exact_and_near_dedup():
    a = "the quick brown fox jumps over the lazy dog near the river bank on a sunny day " * 3
    near = a.replace("sunny", "bright", 1)
    other = "completely different text about rotary position embeddings and grouped query attention " * 3
    out = D.dedup([a, a, " ".join(a.split()), near, other], threshold=0.7)
    assert out == [a, other]
    assert D.jaccard_est(D.minhash(a), D.minhash(near)) > 0.7 > D.jaccard_est(D.minhash(a), D.minhash(other))


# 15 -----------------------------------------------------------------------------------------
def test_15_synthetic_problems_verified_and_val_excluded():
    probs = D.synthetic_problems(12, 4, seed=3)
    assert probs and all('verified="true"' in p for p in probs)
    q = probs[0].split("Q: ")[1].split("\n")[0]
    assert all(q not in p for p in D.synthetic_problems(12, 4, seed=3, exclude=f"Q: {q}\n"))


# 16 -----------------------------------------------------------------------------------------
def test_16_curriculum_order_and_pacing():
    docs = ["x" * 5, "a fairly long and varied sentence, with 123 numbers & symbols! " * 5, "short text"]
    order = D.curriculum_order(docs)
    assert order[0] == "x" * 5 and order[-1] == docs[1]
    c = [D.competence(f / 10, 0.3, 0.6) for f in range(11)]
    assert abs(c[0] - 0.3) < 1e-9 and c[-1] == 1.0 and all(a <= b for a, b in zip(c, c[1:]))


# 17 -----------------------------------------------------------------------------------------
def test_17_scaling_size_prefers_small_model_when_tokens_are_scarce():
    few = schedule.choose_size({"tiny": 1e3, "small": 3e2}, 60)
    many = schedule.choose_size({"tiny": 1e9, "small": 5e8}, 3600)
    assert few["choice"] == "tiny" and many["choice"] == "small"
    assert "estimate" in few["note"]


# 18 -----------------------------------------------------------------------------------------
def test_18_warmup_cosine_and_lr_range_test():
    lrs = [schedule.lr_at(f / 100, 1e-3, 0.1) for f in range(101)]
    assert max(lrs) == pytest.approx(1e-3, rel=1e-2) and lrs[0] < 1e-4 and lrs[-1] == pytest.approx(1e-4)
    m = tiny(6)
    r = schedule.lr_range_test(m, lambda: text_batch(), steps=20)
    assert 1e-4 <= r["suggested_lr"] <= 1e-2 and r["steps"] >= 6


# 19 -----------------------------------------------------------------------------------------
def test_19_mistake_first_replay(tmp_path):
    from artemis_dense.engines.mistakes import MistakeJournal
    j = MistakeJournal(str(tmp_path))
    j.record("user_correction", "Q: 2+2?", "5", "wrong", fix="A: 4")
    rp = replay.MistakeReplay(capacity=2, replay_frac=0.5, journal_dir=str(tmp_path))
    x, y = batch(4, 16)
    rp.observe(x, y, torch.tensor([0.1, 5.0, 3.0, 0.2]))
    assert sorted(h[0] for h in rp.heap) == [3.0, 5.0]
    mx, my, used = rp.mix(torch.zeros_like(x), torch.zeros_like(y))
    assert used == 2
    assert bytes(mx[0].tolist()).startswith(b"Q: 2+2?\nA: 4")       # recorded mistake replayed first
    assert any(torch.equal(mx[1], x[i]) for i in (1, 2))             # then the hardest window
    m = tiny(); logits, _ = m(x)
    assert replay.per_sample_loss(logits, y).shape == (4,)


# 20 -----------------------------------------------------------------------------------------
class _Oracle:
    """Fake model that answers correctly on even sample seeds, wrongly on odd ones."""
    def __init__(self, answer):
        self.answer, self.n = answer, 0
    def eval(self):
        return self
    def generate(self, ids, max_new, temperature=0.0, generator=None):
        self.n += 1
        a = self.answer if self.n % 2 else "wrong"
        return torch.cat([ids, torch.tensor([list((a + "\n").encode())])], 1)
    def sequence_logprob(self, t):
        return torch.tensor([-1.0])


def test_20_self_play_keeps_only_verified_best():
    p = {"q": "Integrate 2*x + 1 from 0 to 2", "a": "6"}
    r = selfplay.best_of_n(_Oracle("6"), D.get_codec("byte"), [p], n=4)
    assert r["tried"] == 4 and r["verified_correct"] == 2 and len(r["kept"]) == 1
    assert r["kept"][0].endswith("A: 6\n</problem>\n")
    r2 = selfplay.best_of_n(tiny(), D.get_codec("byte"), [p], n=2, max_new=4)
    assert r2["tried"] == 2 and len(r2["kept"]) == r2["verified_correct"] == 0


# trainer integration ---------------------------------------------------------------------------
def test_trainer_fast_preset_smoke(tmp_path):
    corpus = tmp_path / "c.txt"
    body = "".join(f'<doc file="d{i}.md">\nArtemis doc {i}: one dense model, sandbox first, approvals always.\n</doc>\n'
                   for i in range(1200))
    corpus.write_text(body)
    from artemis_dense.accel.trainer import run
    cfg = AccelConfig.preset("fast", tokenizer="byte", scaling_size=False, size="tiny", context_len=64, batch=4,
                             synthetic_math=6, synthetic_physics=2, self_play_prompts=1, self_play_n=1,
                             precision="auto", attention="auto", compile=False)
    s = run(cfg, 0.15, "smoke", str(tmp_path / "out"), str(corpus), threads=1, eval_every_s=1000, log_fn=lambda *_: None)
    assert s["steps"] > 0 and s["final"]["bits_per_byte"] < s["before"]["bits_per_byte"]
    assert "13_quality_filter" in s["status"] and "keep_best" in s
    assert s["final_layers"] in (2, 3, 4)          # keep_best may restore a pre-growth snapshot
    with pytest.raises(dev.ApprovalRequired):
        run(AccelConfig.preset("gpu_full"), 0.01, "g", str(tmp_path / "g"), str(corpus))
