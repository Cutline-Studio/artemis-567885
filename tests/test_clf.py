"""Stage 9(c) byte-level CLF tests. Offline: a local http.server, never the network."""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch
import torch.nn.functional as F

import clf
import webstream as ws
from fourier import FourierFeatures

TEXT = (b"header junk\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        + b"It is a truth universally acknowledged, that a single man. " * 40
        + b"*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicense\n")


def small(pos_mode="doc", **kw):
    torch.manual_seed(0)
    cfg = dict(d_model=32, n_heads=4, d_ff=64, num_experts=4, num_layers=2, top_k=2,
               pos_mode=pos_mode, position_frequencies=12)
    cfg.update(kw)
    return clf.ByteCLF(**cfg)


# ---------------------------------------------------------------- local server
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.server.hits += 1
        rng = self.headers.get("Range")
        a, b = rng[6:].split("-")
        a, b = int(a), min(int(b), len(TEXT) - 1)
        if a >= len(TEXT):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(TEXT)}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        part = TEXT[a:b + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(TEXT)}")
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.hits = 0
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}/book.txt"
    srv.shutdown()
    srv.server_close()


def test_fetch_document_local_server_and_cache(server, tmp_path):
    srv, url = server
    f1 = ws.RangeFetcher(cache_dir=str(tmp_path), min_interval=0.0, log=lambda m: None)
    data, info = clf.fetch_document(url, chunk_size=700, fetcher=f1, log=lambda m: None)
    assert data == TEXT
    assert info["n_bytes"] == len(TEXT) and info["synthetic"] is False
    assert info["sources"] == {"network": info["chunks"]}
    hits = srv.hits
    f2 = ws.RangeFetcher(cache_dir=str(tmp_path), min_interval=0.0, log=lambda m: None)
    data2, info2 = clf.fetch_document(url, chunk_size=700, fetcher=f2, log=lambda m: None)
    assert data2 == TEXT and f2.network_requests == 0 and srv.hits == hits
    assert info2["sha256"] == info["sha256"]


def test_fetch_document_fails_loudly(tmp_path):
    f = ws.RangeFetcher(cache_dir=None, min_interval=0.0, log=lambda m: None, timeout=2)
    with pytest.raises(ws.WebStreamError):
        clf.fetch_document("http://127.0.0.1:9/nothing.txt", fetcher=f, log=lambda m: None)


def test_gutenberg_body_and_split():
    a, b = clf.gutenberg_body(TEXT)
    assert TEXT[a:].startswith(b"It is a truth") and TEXT[b:].startswith(b"*** END OF")
    parts = clf.split_document(TEXT, 0.1, (a, b))
    (to, tb), (ho, hb) = parts["train"], parts["heldout"]
    assert to == a and ho == a + len(tb) and tb + hb == TEXT[a:b]
    assert len(hb) == (b - a) - int((b - a) * 0.9)
    assert TEXT[to:to + len(tb)] == tb and TEXT[ho:ho + len(hb)] == hb
    assert clf.gutenberg_body(b"no markers") == (0, 10)
    with pytest.raises(ValueError):
        clf.split_document(TEXT, 0.0)


def test_windows_and_offsets():
    data = torch.arange(100, dtype=torch.uint8)
    w, o = clf.tiled_windows(data, 1000, 30)
    assert w.shape == (3, 30) and o.tolist() == [1000, 1030, 1060]
    assert torch.equal(w[1], torch.arange(30, 60))
    g = torch.Generator().manual_seed(1)
    rw, ro = clf.random_windows(data, 500, 8, 20, g)
    assert rw.shape == (8, 20) and rw.dtype == torch.int64
    for r, off in zip(rw, ro):        # window content equals data at its offset
        assert torch.equal(r, torch.arange(int(off) - 500, int(off) - 480))


# ---------------------------------------------------------------- model
@pytest.mark.parametrize("mode", clf.POS_MODES)
def test_shapes(mode):
    m = small(mode)
    ids = torch.randint(0, 256, (3, 17))
    logits, aux, dropped = m(ids, torch.tensor([0, 100, 5000]))
    assert logits.shape == (3, 17, 256) and aux.dim() == 0 and dropped == 0
    assert m.causal_safe


def test_bad_inputs():
    m = small()
    with pytest.raises(ValueError):
        m(torch.randint(0, 256, (17,)))
    with pytest.raises(TypeError):
        m(torch.rand(2, 5))
    with pytest.raises(ValueError):
        m(torch.full((1, 4), 256))
    with pytest.raises(ValueError):
        m(torch.zeros(2, 4, dtype=torch.long), torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError):
        clf.ByteCLF(pos_mode="bogus")


@pytest.mark.parametrize("mode", clf.POS_MODES)
@pytest.mark.parametrize("value_scalar", [False, True])
def test_causality_future_bytes_do_not_change_past_logits(mode, value_scalar):
    m = small(mode, value_scalar=value_scalar).eval()
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 256, (2, 24), generator=g)
    off = torch.tensor([10, 7777])
    base = m(ids, off)[0]
    for t in (0, 5, 11, 22):
        pert = ids.clone()
        pert[:, t + 1:] = torch.randint(0, 256, pert[:, t + 1:].shape, generator=g)
        out = m(pert, off)[0]
        assert torch.allclose(out[:, :t + 1], base[:, :t + 1], atol=1e-5, rtol=0)
        assert not torch.allclose(out[:, t + 1:], base[:, t + 1:])


def test_causality_across_batch_rows():
    m = small().eval()
    ids = torch.randint(0, 256, (2, 16))
    off = torch.tensor([0, 50])
    base = m(ids, off)[0]
    pert = ids.clone()
    pert[1] = torch.randint(0, 256, (16,))
    assert torch.allclose(m(pert, off)[0][0], base[0], atol=1e-5)


def test_gradients_flow_to_every_parameter():
    m = small("doc", value_scalar=True)
    ids = torch.randint(0, 256, (4, 20))
    logits, aux, _ = m(ids, torch.tensor([0, 10, 20, 30]))
    (clf.next_byte_nats(logits, ids) + 0.01 * aux).backward()
    names = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert names == []
    for n in ("lm.embed.weight", "pos_proj.weight", "value_proj.weight", "lm.head.weight"):
        assert dict(m.named_parameters())[n].grad.abs().sum() > 0, n


def test_one_step_reduces_loss():
    m = small("doc")
    ids = torch.randint(0, 256, (4, 20))
    off = torch.zeros(4, dtype=torch.long)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    l0 = None
    for _ in range(20):
        loss = clf.next_byte_nats(m(ids, off)[0], ids)
        l0 = float(loss.detach()) if l0 is None else l0
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < l0


def test_no_position_ablation_path():
    m = small("none").eval()
    assert m.gamma is None and m.pos_proj is None
    assert not any("pos_proj" in n or "gamma" in n for n, _ in m.named_parameters())
    ids = torch.randint(0, 256, (2, 12))
    a = m(ids, torch.tensor([0, 0]))[0]
    b = m(ids, torch.tensor([123456, 999]))[0]
    c = m(ids)[0]
    assert torch.equal(a, b) and torch.equal(a, c)          # offsets are ignored
    # embedding is exactly the byte embedding
    assert torch.equal(m.embed(ids), m.lm.embed(ids))


def test_doc_mode_depends_on_offset_window_mode_does_not():
    ids = torch.randint(0, 256, (1, 12))
    d = small("doc").eval()
    with torch.no_grad():
        d.pos_proj.weight.normal_()
    assert not torch.allclose(d(ids, torch.tensor([0]))[0], d(ids, torch.tensor([4096]))[0])
    w = small("window").eval()
    assert torch.equal(w(ids, torch.tensor([0]))[0], w(ids, torch.tensor([4096]))[0])


def test_position_features_match_fourier_and_bytefeaturizer():
    m = small("doc", position_frequencies=12, position_scale=2.0 ** 20)
    off = torch.tensor([678231])
    g = m.position_features(5, off)
    t = (torch.arange(5, dtype=torch.float64) + 678231) / 2 ** 20
    ref = FourierFeatures(1, 12)(t.unsqueeze(-1)).float()
    assert torch.allclose(g[0], ref, atol=1e-6)
    bf = ws.ByteFeaturizer(value_frequencies=1, position_frequencies=12, dtype=torch.float64)
    feats = bf(torch.zeros(5, dtype=torch.uint8), offset=678231)["features"][:, 2:]
    assert torch.allclose(g[0], feats.float(), atol=1e-6)


def test_position_precision_float64():
    """Fine bands at large offsets: neighbouring bytes must get distinct features."""
    m = small("doc", position_frequencies=20)
    g = m.position_features(2, torch.tensor([700001]))
    assert (g[0, 0] - g[0, 1]).abs().max() > 0.1


# ---------------------------------------------------------------- metrics
def test_bpb_hand_computed():
    # uniform over 256 -> exactly 8 bits
    assert clf.bpb_from_logits(torch.zeros(5, 256), torch.randint(0, 256, (5,))) == pytest.approx(8.0)
    # p(target) = 1/2 for row 0, 1/4 for row 1 -> mean bits (1 + 2)/2 = 1.5
    logits = torch.full((2, 256), -1e9, dtype=torch.float64)
    logits[0, 65] = 0.0; logits[0, 66] = 0.0                       # {A: 1/2, B: 1/2}
    logits[1, 10] = math.log(3.0); logits[1, 32] = 0.0             # {\n: 3/4, ' ': 1/4}
    assert clf.bpb_from_logits(logits, torch.tensor([65, 32])) == pytest.approx(1.5, abs=1e-9)


def test_evaluate_bpb_matches_manual():
    m = small("doc")
    w = torch.randint(0, 256, (5, 9))
    off = torch.arange(5) * 9
    got = clf.evaluate_bpb(m, w, off, batch=2)
    m.eval()
    logits = m(w, off)[0]
    ref = clf.bpb_from_logits(logits[:, :-1], w[:, 1:])
    assert got == pytest.approx(ref, abs=1e-5)


def test_unigram_and_entropy_hand_computed():
    assert clf.byte_entropy_bits(b"abab") == pytest.approx(1.0)
    assert clf.byte_entropy_bits(b"aaaa") == pytest.approx(0.0)
    assert clf.byte_entropy_bits(b"aabc") == pytest.approx(1.5)
    assert clf.byte_entropy_bits(bytes(range(256))) == pytest.approx(8.0)
    # unigram with no smoothing on the train set == entropy
    assert clf.unigram_bpb(b"aabc", b"aabc", smoothing=0) == pytest.approx(1.5)


def test_train_variant_tiny_end_to_end():
    class A:  # tiny config
        d_model, n_heads, d_ff, experts, layers, top_k = 16, 2, 32, 2, 1, 1
        position_frequencies, position_scale, value_scalar, dispatch = 8, 2.0 ** 12, False, "loop"
        seq_len, batch, steps, lr, warmup, weight_decay, grad_clip, aux_alpha = 16, 4, 6, 3e-3, 2, 0.0, 1.0, 0.01
        eval_every, eval_windows, eval_batch, seed = 3, 4, 2, 0
    a, b = clf.gutenberg_body(TEXT)
    parts = clf.split_document(TEXT, 0.2, (a, b))
    r1 = clf.train_variant("doc", parts["train"], parts["heldout"], A, log=lambda m: None)
    r2 = clf.train_variant("doc", parts["train"], parts["heldout"], A, log=lambda m: None)
    assert r1["final_heldout_bpb"] == r2["final_heldout_bpb"]   # deterministic
    assert [c["step"] for c in r1["curve"]] == [0, 3, 6]
    assert r1["eval_windows"] == 4 and math.isfinite(r1["final_heldout_bpb"])
    json.dumps(r1)
    rn = clf.train_variant("none", parts["train"], parts["heldout"], A, log=lambda m: None)
    assert rn["params"] < r1["params"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    torch.manual_seed(0)
    m = clf.ByteCLF(d_model=32, n_heads=2, d_ff=64, num_experts=4, num_layers=2, pos_mode="doc",
                dispatch="vectorized").eval()
    ids = torch.randint(0, 256, (3, 17))
    off = torch.tensor([0, 1000, 500000])
    with torch.no_grad():
        a = m(ids, off)[0]
        b = m.to("cuda")(ids.cuda(), off.cuda())[0].cpu()
    assert b.shape == a.shape
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
