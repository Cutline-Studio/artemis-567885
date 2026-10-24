"""Stage 6: byte-level training loop (train_bytes.py). Small and CPU-fast."""
import json, math
import pytest
import torch
import train_bytes as tb
from model import ArtemisModel

TEXT = (b"the quick brown fox jumps over the lazy dog. " * 40
        + b"pack my box with five dozen liquor jugs. " * 40)


def args(tmp_path, **kw):
    a = tb.build_parser().parse_args([])
    base = dict(d_model=32, n_heads=4, d_ff=64, experts=4, layers=2, seq_len=32, batch=8, lr=3e-3,
                warmup=5, max_steps=40, time_budget=0, eval_every=10, eval_windows=16, sample_len=20,
                prompt=["the "], out_dir=str(tmp_path), run_name="t", seed=1, threads=1)
    base.update(kw)
    for k, v in base.items():
        setattr(a, k, v)
    return a


def test_loop_runs_and_writes_curve(tmp_path):
    s = tb.train(args(tmp_path, max_steps=20), TEXT[:3000], TEXT[3000:], log=lambda *_: None)
    assert s["steps"] == 20
    d = tmp_path / "t"
    rows = [json.loads(l) for l in (d / "curve.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 10, 20]
    assert all(math.isfinite(r["heldout_bpb"]) for r in rows)
    assert (d / "curve.csv").read_text().splitlines()[0].startswith("step,elapsed_s")
    assert json.loads((d / "summary.json").read_text())["steps"] == 20
    assert (d / "samples.txt").exists()


def test_loss_decreases_on_tiny_text(tmp_path):
    s = tb.train(args(tmp_path, max_steps=60, eval_every=20), TEXT, TEXT, log=lambda *_: None)
    assert s["final_heldout_bpb"] < s["initial_heldout_bpb"] - 2.0
    assert s["final_heldout_bpb"] < s["unigram_bpb_eval_windows"]


def test_time_budget_stops(tmp_path):
    s = tb.train(args(tmp_path, max_steps=0, time_budget=1.0, eval_every=10_000), TEXT, TEXT,
                 log=lambda *_: None)
    assert s["steps"] > 0 and s["train_time_s"] < 10


def test_seed_control_reproducible(tmp_path):
    a = tb.train(args(tmp_path, max_steps=10, run_name="a"), TEXT, TEXT, log=lambda *_: None)
    b = tb.train(args(tmp_path, max_steps=10, run_name="b"), TEXT, TEXT, log=lambda *_: None)
    assert a["final_heldout_bpb"] == b["final_heldout_bpb"]
    assert a["samples"][0]["text"] == b["samples"][0]["text"]


def test_eval_bpb_finite_and_uniform_at_init():
    torch.manual_seed(0)
    m = ArtemisModel(32, 4, 64, 4, 2, max_len=33)
    w = tb.eval_windows(tb.to_tensor(TEXT), 33, 8)
    v = tb.evaluate_bpb(m, w, batch=3)
    assert math.isfinite(v) and 6.0 < v < 11.0      # random init ~ log2(256) = 8
    with torch.no_grad():                            # zero head -> exactly uniform -> 8 bpb
        m.head.weight.zero_(); m.head.bias.zero_()
    assert tb.evaluate_bpb(m, w) == pytest.approx(8.0, abs=1e-5)


def test_unigram_known_distribution():
    # p(a)=1/2, p(b)=1/4, p(c)=1/4 -> entropy 1.5 bits
    train = b"aabc" * 100
    assert tb.unigram_bpb(train, train, smoothing=0) == pytest.approx(1.5, abs=1e-12)
    assert tb.unigram_bpb(train, b"a" * 10, smoothing=0) == pytest.approx(1.0)
    assert tb.unigram_bpb(train, b"cc", smoothing=0) == pytest.approx(2.0)
    assert tb.unigram_bpb(train, b"z", smoothing=0) == math.inf
    # Laplace smoothing: unseen byte gets (0+1)/(400+256)
    assert tb.unigram_bpb(train, b"z", smoothing=1.0) == pytest.approx(math.log2(656))
    assert tb.unigram_bpb(b"\x00\xff" * 8, bytes(range(256)) , smoothing=0) == math.inf


def test_eval_windows_heldout_disjoint_and_deterministic():
    data = tb.to_tensor(bytes(range(256)) * 4)
    w = tb.eval_windows(data, 16, 10)
    assert w.shape == (10, 16) and torch.equal(w, tb.eval_windows(data, 16, 10))
    assert w.dtype == torch.long
    # non-overlapping: window i starts at a multiple of 16 and starts are strictly increasing by >= 16
    st = [int(r[0]) for r in w] ; pos = [i * (1024 // 16 // 10) * 16 for i in range(10)]
    assert st == [p % 256 for p in pos]


def test_text_split_contiguous(tmp_path):
    p = tmp_path / "book.txt"; p.write_bytes(bytes(range(100)))
    tr, ho, _ = tb.load_data(None, str(p), heldout_frac=0.1)
    assert tr == bytes(range(90)) and ho == bytes(range(90, 100))


def test_sampling_valid_bytes_and_seed_deterministic():
    torch.manual_seed(0)
    m = ArtemisModel(32, 4, 64, 4, 2, max_len=16)
    a = tb.sample(m, b"ab", 40, temperature=1.0, top_k=10, seed=3)
    b = tb.sample(m, b"ab", 40, temperature=1.0, top_k=10, seed=3)
    c = tb.sample(m, b"ab", 40, temperature=1.0, top_k=10, seed=4)
    assert isinstance(a, bytes) and len(a) == 42 and a[:2] == b"ab"
    assert all(0 <= x <= 255 for x in a)
    assert a == b and a != c
    g1 = tb.sample(m, b"ab", 10, temperature=0, seed=1)   # greedy ignores the seed
    assert g1 == tb.sample(m, b"ab", 10, temperature=0, seed=2)


def test_top_k_one_equals_greedy():
    torch.manual_seed(0)
    m = ArtemisModel(32, 4, 64, 4, 2, max_len=16)
    assert tb.sample(m, b"x", 12, temperature=1.0, top_k=1, seed=5) == tb.sample(m, b"x", 12, temperature=0)


def test_lr_schedule():
    assert tb.lr_at(0, 0, 1.0, 10) == pytest.approx(0.1)
    assert tb.lr_at(10, 0.0, 1.0, 10) == pytest.approx(1.0)
    assert tb.lr_at(10, 1.0, 1.0, 10) == pytest.approx(0.1)
