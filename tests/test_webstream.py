"""Stage 9b webstream tests against a local http.server running in a thread."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

import webstream as ws
from fourier import FourierFeatures

DOC = bytes((i * 7 + 3) % 256 for i in range(5000))  # every offset distinguishable


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send_range(self, body):
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        a, b = rng[6:].split("-")
        a, b = int(a), (int(b) if b else len(body) - 1)
        if a >= len(body):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(body)}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        b = min(b, len(body) - 1)
        part = body[a:b + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(body)}")
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)

    def do_GET(self):
        srv = self.server
        srv.log.append({"path": self.path, "range": self.headers.get("Range"),
                        "ua": self.headers.get("User-Agent"), "t": time.monotonic()})
        p = self.path
        if p == "/doc.txt":
            self._send_range(DOC)
        elif p == "/norange.txt":           # ignores Range
            self.send_response(200)
            self.send_header("Content-Length", str(len(DOC)))
            self.end_headers()
            self.wfile.write(DOC)
        elif p.startswith("/limited"):       # 429 first, then OK
            srv.hits[p] = srv.hits.get(p, 0) + 1
            if srv.hits[p] <= srv.fail_times:
                self.send_response(429)
                self.send_header("Retry-After", srv.retry_after)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send_range(DOC)
        elif p == "/always429":
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif p == "/badrange":
            self.send_response(206)
            self.send_header("Content-Range", "bytes 0-9/5000")
            self.send_header("Content-Length", "10")
            self.end_headers()
            self.wfile.write(DOC[:10])
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.log, srv.hits, srv.fail_times, srv.retry_after = [], {}, 1, "1"
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield srv
    srv.shutdown()
    srv.server_close()


def F(tmp_path=None, **kw):
    kw.setdefault("min_interval", 0.0)
    kw.setdefault("log", lambda m: None)
    return ws.RangeFetcher(cache_dir=str(tmp_path / "cache") if tmp_path else None, **kw)


# ------------------------------------------------------------------ ranges
@pytest.mark.parametrize("start,end", [(0, 0), (0, 99), (123, 456), (4000, 4999), (4990, 5100)])
def test_correct_byte_ranges(server, start, end):
    f = F()
    r = f.fetch(server.base + "/doc.txt", start, end)
    assert r.status == 206 and r.range_honored is True and r.source == "network"
    assert r.data == DOC[start:end + 1]
    assert r.total_length == len(DOC)
    assert server.log[-1]["range"] == f"bytes={start}-{end}"
    assert r.eof == (end >= len(DOC))
    assert not r.synthetic


def test_user_agent_identifies(server):
    F().fetch(server.base + "/doc.txt", 0, 9)
    ua = server.log[-1]["ua"]
    assert ua == ws.USER_AGENT and "Artemis-567885" in ua and "github.com" in ua


def test_416_past_end_is_eof(server):
    r = F().fetch(server.base + "/doc.txt", 6000, 6099)
    assert r.status == 416 and r.data == b"" and r.eof and r.total_length == len(DOC)


def test_bad_range_rejected():
    with pytest.raises(ValueError):
        F().fetch("http://x", 10, 5)


# ------------------------------------------------------------------ 206 vs 200
def test_200_server_ignored_range_is_sliced(server):
    logs = []
    f = F(log=logs.append)
    r = f.fetch(server.base + "/norange.txt", 100, 199)
    assert r.status == 200 and r.range_honored is False
    assert r.data == DOC[100:200] and len(r.data) == 100
    assert server.log[-1]["range"] == "bytes=100-199"   # we did send Range
    assert any("ignored Range" in m for m in logs)


def test_200_read_is_limited(server):
    f = F(max_ignored_range_bytes=1000)
    assert f.fetch(server.base + "/norange.txt", 0, 999).data == DOC[:1000]
    with pytest.raises(ws.WebStreamError, match="ignored Range"):
        f.fetch(server.base + "/norange.txt", 1000, 1999)


def test_206_with_wrong_content_range_fails(server):
    with pytest.raises(ws.WebStreamError, match="Content-Range"):
        F().fetch(server.base + "/badrange", 100, 199)


# ------------------------------------------------------------------ cache
def test_cache_hit_avoids_network(server, tmp_path):
    f = F(tmp_path)
    url = server.base + "/doc.txt"
    r1 = f.fetch(url, 10, 59)
    n = len(server.log)
    r2 = f.fetch(url, 10, 59)
    assert len(server.log) == n and f.network_requests == 1
    assert r2.source == "cache" and r2.data == r1.data == DOC[10:60] and r2.status == 206
    # a new fetcher on the same cache dir also hits
    f2 = F(tmp_path)
    assert f2.fetch(url, 10, 59).source == "cache" and f2.network_requests == 0
    assert len(server.log) == n


def test_cache_keyed_by_url_and_range(server, tmp_path):
    f = F(tmp_path)
    url = server.base + "/doc.txt"
    f.fetch(url, 0, 9)
    f.fetch(url, 0, 10)                       # different range -> network
    assert f.network_requests == 2
    assert ws.cache_key(url, 0, 9) != ws.cache_key(url, 0, 10)
    assert ws.cache_key(url, 0, 9) != ws.cache_key(url + "2", 0, 9)
    assert len([p for p in os.listdir(tmp_path / "cache") if p.endswith(".bin")]) == 2


def test_truncated_cache_refetches(server, tmp_path):
    f = F(tmp_path)
    url = server.base + "/doc.txt"
    f.fetch(url, 0, 99)
    bin_p, _ = f._cache_paths(url, 0, 99)
    with open(bin_p, "wb") as fh:
        fh.write(b"xx")
    r = f.fetch(url, 0, 99)
    assert r.source == "network" and r.data == DOC[:100] and f.network_requests == 2


def test_use_cache_false_goes_to_network(server, tmp_path):
    f = F(tmp_path)
    url = server.base + "/doc.txt"
    f.fetch(url, 0, 9)
    assert f.fetch(url, 0, 9, use_cache=False).source == "network"
    assert f.network_requests == 2


# ------------------------------------------------------------------ rate limit
def test_default_rate_limit_at_least_one_second():
    assert ws.DEFAULT_MIN_INTERVAL >= 1.0
    assert ws.RangeFetcher(log=lambda m: None).min_interval >= 1.0


def test_rate_limit_spacing_real_time(server):
    f = F(min_interval=0.3)
    url = server.base + "/doc.txt"
    for i in range(4):
        f.fetch(url, i * 10, i * 10 + 9)
    ts = [e["t"] for e in server.log]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert len(gaps) == 3 and min(gaps) >= 0.29, gaps


def test_rate_limit_one_second_default_with_fake_clock(server):
    now = [100.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    f = ws.RangeFetcher(log=lambda m: None, sleep=sleep, clock=lambda: now[0])
    url = server.base + "/doc.txt"
    f.fetch(url, 0, 9)
    now[0] += 0.25
    f.fetch(url, 10, 19)
    f.fetch(url, 20, 29)
    assert slept == pytest.approx([0.75, 1.0])
    assert [b - a for a, b in zip(f.request_times, f.request_times[1:])] == pytest.approx([1.0, 1.0])


def test_cache_hits_do_not_wait(server, tmp_path):
    slept = []
    f = ws.RangeFetcher(cache_dir=str(tmp_path / "c"), log=lambda m: None, sleep=slept.append)
    url = server.base + "/doc.txt"
    f.fetch(url, 0, 9)
    for _ in range(5):
        f.fetch(url, 0, 9)
    assert slept == [] and f.network_requests == 1


# ------------------------------------------------------------------ 429 / Retry-After
def test_429_retry_after_seconds_honoured(server):
    logs = []
    f = F(min_interval=0.0, log=logs.append)
    t0 = time.monotonic()
    r = f.fetch(server.base + "/limited1", 0, 49)
    dt = time.monotonic() - t0
    assert r.data == DOC[:50] and r.status == 206 and r.retries == 1
    assert dt >= 0.95
    gap = server.log[1]["t"] - server.log[0]["t"]
    assert gap >= 0.95 and len(server.log) == 2
    assert any("429" in m and "waiting 1.00s" in m for m in logs)


def test_429_retry_after_http_date(server):
    server.retry_after = "Wed, 21 Oct 2015 07:28:00 GMT"   # in the past -> wait 0 (but min_interval)
    slept = []
    f = ws.RangeFetcher(log=lambda m: None, min_interval=1.0, sleep=slept.append)
    r = f.fetch(server.base + "/limited2", 0, 9)
    assert r.data == DOC[:10]
    assert slept and slept[0] >= 0.0    # min interval still applied from first request


def test_parse_retry_after():
    assert ws._parse_retry_after("5") == 5.0
    assert ws._parse_retry_after(None) is None
    assert ws._parse_retry_after("garbage") is None
    assert ws._parse_retry_after("Wed, 21 Oct 2015 07:28:10 GMT",
                                 now=1445412480.0) == pytest.approx(10.0)


def test_retry_wait_uses_max_of_retry_after_and_interval(server):
    server.retry_after = "3"
    slept = []
    now = [0.0]

    def sleep(s):
        slept.append(s)
        now[0] += s
    f = ws.RangeFetcher(log=lambda m: None, min_interval=1.0, sleep=sleep, clock=lambda: now[0])
    f.fetch(server.base + "/limited3", 0, 9)
    assert slept == pytest.approx([3.0])


def test_persistent_429_fails_loudly(server):
    f = F(max_retries=2)
    with pytest.raises(ws.WebStreamError, match="gave up"):
        f.fetch(server.base + "/always429", 0, 9)
    assert len(server.log) == 3


def test_retry_after_too_long_fails(server):
    server.retry_after = "9999"
    with pytest.raises(ws.WebStreamError, match="max_retry_after"):
        F().fetch(server.base + "/limited4", 0, 9)


# ------------------------------------------------------------------ loud failure / synthetic
def test_404_fails_loudly_by_default(server):
    with pytest.raises(ws.WebStreamError, match="404"):
        ws.fetch_bytes(server.base + "/missing.txt", 0, 32, F())


def test_connection_refused_fails_loudly():
    with pytest.raises(ws.WebStreamError):
        ws.fetch_bytes("http://127.0.0.1:9/x", 0, 32, F(timeout=2))


def test_synthetic_fallback_is_labelled(server, tmp_path):
    logs = []
    f = F(tmp_path, log=logs.append)
    r = ws.fetch_bytes(server.base + "/missing.txt", 0, 32, f, fallback="synthetic")
    assert r.synthetic and r.source == "SYNTHETIC" and r.status is None
    assert "SYNTHETIC" in r.label and r.summary()["synthetic"] is True
    assert r.data == ws.synthetic_bytes(32)
    assert any(m.startswith(ws.SYNTHETIC_TAG) and "NOT internet data" in m for m in logs)
    assert not os.listdir(tmp_path / "cache")   # synthetic never cached


def test_real_fetch_not_labelled_synthetic(server):
    r = ws.fetch_bytes(server.base + "/doc.txt", 0, 32, F(), fallback="synthetic")
    assert not r.synthetic and r.source == "network" and "SYNTHETIC" not in r.label


def test_bad_fallback_value():
    with pytest.raises(ValueError):
        ws.fetch_bytes("http://x", 0, 1, F(), fallback="silent")


def test_cli_synthetic_output_labelled(server, tmp_path, capsys):
    log = tmp_path / "out.jsonl"
    rec = ws.main(["--url", server.base + "/missing.txt", "--length", "16", "--no-cache",
                   "--fallback", "synthetic", "--log", str(log)])
    out = capsys.readouterr().out
    assert out.startswith(ws.SYNTHETIC_TAG) and rec["synthetic"] and "SYNTHETIC" in rec["label"]
    assert json.loads(log.read_text())["source"] == "SYNTHETIC"


def test_cli_default_fails_loudly(server):
    with pytest.raises(ws.WebStreamError):
        ws.main(["--url", server.base + "/missing.txt", "--length", "16", "--no-cache"])


# ------------------------------------------------------------------ resumable stream
def test_stream_reads_whole_doc_in_chunks(server):
    s = ws.WebByteStream(server.base + "/doc.txt", chunk_size=1000, fetcher=F())
    chunks = list(s)
    assert b"".join(c.data for c in chunks) == DOC
    assert [c.start for c in chunks] == [0, 1000, 2000, 3000, 4000] and s.done


def test_stream_resumes_from_state(server, tmp_path):
    url = server.base + "/doc.txt"
    st = str(tmp_path / "state.json")
    s1 = ws.WebByteStream(url, chunk_size=700, fetcher=F(), state_path=st)
    got = [s1.next_chunk().data, s1.next_chunk().data]
    assert json.load(open(st))["offset"] == 1400
    s2 = ws.WebByteStream(url, chunk_size=700, fetcher=F(), state_path=st)
    assert s2.offset == 1400
    got += [c.data for c in s2]
    assert b"".join(got) == DOC
    assert json.load(open(st))["done"] is True
    s3 = ws.WebByteStream(url, chunk_size=700, fetcher=F(), state_path=st)
    assert s3.next_chunk() is None
    ranges = [e["range"] for e in server.log]
    assert ranges.count("bytes=0-699") == 1     # nothing refetched after resume


def test_stream_state_url_mismatch(server, tmp_path):
    st = str(tmp_path / "s.json")
    ws.WebByteStream(server.base + "/doc.txt", 10, fetcher=F(), state_path=st).next_chunk()
    with pytest.raises(ValueError):
        ws.WebByteStream(server.base + "/other.txt", 10, fetcher=F(), state_path=st)


def test_stream_start_end(server):
    s = ws.WebByteStream(server.base + "/doc.txt", chunk_size=64, start=100, end=300, fetcher=F())
    assert b"".join(c.data for c in s) == DOC[100:300]


def test_stream_exact_multiple_ends_via_total(server):
    s = ws.WebByteStream(server.base + "/doc.txt", chunk_size=2500, fetcher=F())
    assert len(list(s)) == 2 and s.done
    assert len(server.log) == 2     # no extra request past the end


def test_stream_synthetic_counted(server, tmp_path):
    st = str(tmp_path / "s.json")
    s = ws.WebByteStream(server.base + "/missing.txt", 16, end=48, fetcher=F(), state_path=st,
                         fallback="synthetic")
    chunks = list(s)
    assert all(c.synthetic for c in chunks) and s.synthetic_chunks == 3
    assert json.load(open(st))["synthetic_chunks"] == 3


# ------------------------------------------------------------------ features
def test_featurizer_per_element(server):
    r = F().fetch(server.base + "/doc.txt", 1000, 1031)
    fz = ws.ByteFeaturizer(value_frequencies=4, position_frequencies=6, position_scale=5000)
    out = fz(r.as_tensor(), offset=r.start)
    assert out["ids"].dtype == torch.long and out["ids"].tolist() == list(DOC[1000:1032])
    assert out["features"].shape == (32, fz.out_dim) and fz.out_dim == 2 * 4 + 2 * 6
    vf = FourierFeatures(1, 4)(torch.tensor(list(DOC[1000:1032]), dtype=torch.float32)
                               .unsqueeze(-1) / 255)
    pf = FourierFeatures(1, 6)(((torch.arange(32, dtype=torch.float64) + 1000) / 5000)
                               .float().unsqueeze(-1))
    assert torch.allclose(out["features"], torch.cat([vf, pf], -1))


def test_featurizer_value_and_position_independent():
    fz = ws.ByteFeaturizer(value_frequencies=3, position_frequencies=5)
    a = fz(torch.tensor([65, 65, 66], dtype=torch.uint8), offset=0)["features"]
    vd = 2 * 3
    # same byte value -> same value features; different position -> different pos features
    assert torch.equal(a[0, :vd], a[1, :vd]) and not torch.equal(a[0, vd:], a[1, vd:])
    # same position, different values -> same pos features
    b = fz(torch.tensor([66, 0, 0], dtype=torch.uint8), offset=0)["features"]
    assert torch.equal(a[0, vd:], b[0, vd:]) and not torch.equal(a[0, :vd], b[0, :vd])
    # a byte's features do not depend on its neighbours (no window mixing)
    c = fz(torch.tensor([65, 200, 7], dtype=torch.uint8), offset=0)["features"]
    assert torch.equal(a[0], c[0])


def test_featurizer_offset_and_batch():
    fz = ws.ByteFeaturizer()
    x = torch.randint(0, 256, (2, 10), dtype=torch.uint8)
    out = fz(x, offset=50)
    assert out["features"].shape == (2, 10, fz.out_dim)
    whole = fz(torch.zeros(60, dtype=torch.uint8), offset=0)["position"]
    assert torch.equal(out["position"][0], whole[50:60])
    assert ((out["features"] >= -1) & (out["features"] <= 1)).all()


def test_featurizer_rejects_non_bytes():
    with pytest.raises(TypeError):
        ws.ByteFeaturizer()(torch.tensor([1, 2, 3]))


def test_featurizer_no_params_and_global_rng_untouched():
    s = torch.random.get_rng_state()
    fz = ws.ByteFeaturizer()
    assert list(fz.parameters()) == []
    assert torch.equal(s, torch.random.get_rng_state())


def test_empty_result_tensor():
    r = ws.FetchResult("u", 0, 9, b"", 416, "network", eof=True)
    assert r.as_tensor().shape == (0,) and r.as_tensor().dtype == torch.uint8
