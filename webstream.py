"""Stream real text bytes over HTTP Range requests (PLAN stage 9b).

Ported from `docs/reference/clf_stream_original.py` with these fixes:

1. No silent synthetic fallback. A failed fetch raises `WebStreamError`. Synthetic bytes
   are only produced when the caller passes `fallback="synthetic"` explicitly, and then
   every result carries `synthetic=True`, `source="SYNTHETIC"`, and every log line is
   prefixed with ``[SYNTHETIC]``.
2. A real ``Range: bytes=start-end`` header is sent and HTTP 206 is required. The
   Content-Range of a 206 is checked against the requested start. A 200 means the server
   ignored Range: the body is read only up to ``end + 1`` bytes (bounded by
   `max_ignored_range_bytes`) and sliced explicitly; the result has `range_honored=False`.
   416 means the start is past the end of the resource (end of stream).
3. Working source: a Project Gutenberg plain-text book (the default smoke URL; the old
   w3.org URL returns 404).
4. Per-element features: `ByteFeaturizer` encodes each byte value and its own absolute
   position separately with `fourier.FourierFeatures` (stage 9a), giving one vector per
   byte, so the stream plugs into the stage 9c byte-level model. Raw byte ids (0..255)
   are returned too, for an `nn.Embedding(256, d)` front end. No tokenizer anywhere.
5. torch instead of NumPy.

Politeness: an identifying User-Agent, >= 1 s between network requests (also between
retries), 429/503 handled by honouring Retry-After (seconds or HTTP-date) with a bounded
number of retries. Cache: one file per (URL, start, end) under `cache_dir`, so a repeated
request makes no network call. Resume: `WebByteStream` persists its next offset to a JSON
state file after each chunk.

Nothing else in the repo imports this module yet; no existing default changed.
"""
from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import torch

from fourier import FourierFeatures

USER_AGENT = ("Artemis-567885-webstream/0.1 (research byte-stream prototype; "
              "+https://github.com/lpittman-blip/artemis-567885)")
DEFAULT_MIN_INTERVAL = 1.0
DEFAULT_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"  # Pride and Prejudice
SYNTHETIC_TAG = "[SYNTHETIC]"


class WebStreamError(RuntimeError):
    """Raised when real bytes could not be fetched (never silently replaced)."""


@dataclass
class FetchResult:
    url: str
    start: int
    end: int                      # inclusive end that was requested
    data: bytes
    status: Optional[int]         # HTTP status of the network response (None for synthetic)
    source: str                   # "network" | "cache" | "SYNTHETIC"
    range_honored: Optional[bool] = None   # True for 206, False for 200 (server ignored Range)
    total_length: Optional[int] = None     # from Content-Range when known
    elapsed_s: float = 0.0
    synthetic: bool = False
    eof: bool = False             # 416 or fewer bytes than requested
    retries: int = 0

    @property
    def label(self) -> str:
        return "SYNTHETIC (not internet data)" if self.synthetic else f"real bytes ({self.source})"

    def as_tensor(self) -> torch.Tensor:
        return torch.frombuffer(bytearray(self.data), dtype=torch.uint8) if self.data \
            else torch.empty(0, dtype=torch.uint8)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("data")
        d["n_bytes"] = len(self.data)
        d["label"] = self.label
        return d


def _parse_retry_after(value: Optional[str], now: Optional[float] = None) -> Optional[float]:
    """Retry-After as delta-seconds or HTTP-date -> seconds to wait (>= 0), else None."""
    if value is None:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    now = time.time() if now is None else now
    return max(0.0, dt.timestamp() - now)


def _parse_content_range(value: Optional[str]):
    """'bytes a-b/total' -> (a, b, total or None); 'bytes */total' -> (None, None, total);
    None if missing/unparseable."""
    if not value:
        return None
    try:
        unit, rest = value.strip().split(" ", 1)
        if unit.lower() != "bytes":
            return None
        rng, total = rest.split("/", 1)
        if rng.strip() == "*":        # unsatisfied-range form of a 416: 'bytes */total'
            return None, None, (None if total.strip() == "*" else int(total))
        a, b = rng.split("-", 1)
        return int(a), int(b), (None if total.strip() == "*" else int(total))
    except ValueError:
        return None


def cache_key(url: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{url}\nbytes={start}-{end}".encode()).hexdigest()


class RangeFetcher:
    """Polite HTTP Range fetcher with a disk cache.

    `sleep` and `clock` are injectable for tests; defaults are real time.
    """

    def __init__(self, cache_dir: Optional[str] = None, user_agent: str = USER_AGENT,
                 min_interval: float = DEFAULT_MIN_INTERVAL, timeout: float = 20.0,
                 max_retries: int = 3, max_retry_after: float = 120.0,
                 max_ignored_range_bytes: int = 8 * 1024 * 1024,
                 log: Optional[Callable[[str], None]] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retry_after = max_retry_after
        self.max_ignored_range_bytes = max_ignored_range_bytes
        self.log = log if log is not None else (lambda msg: print(msg, file=sys.stderr))
        self._sleep = sleep
        self._clock = clock
        self._last_request: Optional[float] = None
        self._lock = threading.Lock()
        self.network_requests = 0      # number of HTTP requests actually sent
        self.request_times: list = []  # clock() at each send (for tests / logs)

    # ---------------------------------------------------------------- cache
    def _cache_paths(self, url, start, end):
        k = cache_key(url, start, end)
        return os.path.join(self.cache_dir, k + ".bin"), os.path.join(self.cache_dir, k + ".json")

    def _cache_get(self, url, start, end) -> Optional[FetchResult]:
        if not self.cache_dir:
            return None
        bin_p, meta_p = self._cache_paths(url, start, end)
        if not (os.path.exists(bin_p) and os.path.exists(meta_p)):
            return None
        with open(meta_p) as f:
            meta = json.load(f)
        if meta.get("url") != url or meta.get("start") != start or meta.get("end") != end:
            return None  # hash collision guard
        with open(bin_p, "rb") as f:
            data = f.read()
        if len(data) != meta.get("n_bytes"):
            return None  # truncated cache file: refetch
        return FetchResult(url=url, start=start, end=end, data=data, status=meta.get("status"),
                           source="cache", range_honored=meta.get("range_honored"),
                           total_length=meta.get("total_length"), eof=meta.get("eof", False))

    def _cache_put(self, r: FetchResult):
        if not self.cache_dir or r.synthetic:
            return
        bin_p, meta_p = self._cache_paths(r.url, r.start, r.end)
        tmp = bin_p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(r.data)
        os.replace(tmp, bin_p)
        meta = {"url": r.url, "start": r.start, "end": r.end, "n_bytes": len(r.data),
                "status": r.status, "range_honored": r.range_honored,
                "total_length": r.total_length, "eof": r.eof,
                "sha256": hashlib.sha256(r.data).hexdigest(), "fetched_at": time.time()}
        with open(meta_p + ".tmp", "w") as f:
            json.dump(meta, f)
        os.replace(meta_p + ".tmp", meta_p)

    # ---------------------------------------------------------------- rate limit
    def _wait_turn(self, extra: float = 0.0):
        now = self._clock()
        wait = 0.0
        if self._last_request is not None:
            wait = self._last_request + self.min_interval - now
        wait = max(wait, extra)
        if wait > 0:
            self._sleep(wait)
        self._last_request = self._clock()
        self.request_times.append(self._last_request)
        self.network_requests += 1

    # ---------------------------------------------------------------- fetch
    def fetch(self, url: str, start: int, end: int, use_cache: bool = True) -> FetchResult:
        """Fetch bytes [start, end] (inclusive). Raises WebStreamError on failure."""
        if start < 0 or end < start:
            raise ValueError(f"bad range {start}-{end}")
        if use_cache:
            hit = self._cache_get(url, start, end)
            if hit is not None:
                self.log(f"[webstream] cache hit {url} bytes={start}-{end} ({len(hit.data)} B)")
                return hit
        with self._lock:
            r = self._fetch_network(url, start, end)
        if use_cache:
            self._cache_put(r)
        return r

    def _fetch_network(self, url, start, end) -> FetchResult:
        want = end - start + 1
        retry_wait = 0.0
        last_err = None
        for attempt in range(self.max_retries + 1):
            self._wait_turn(retry_wait)
            req = urllib.request.Request(url, headers={
                "User-Agent": self.user_agent, "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity"})
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = resp.status
                    if status == 206:
                        cr = _parse_content_range(resp.headers.get("Content-Range"))
                        if cr is None or cr[0] != start:
                            raise WebStreamError(
                                f"206 with Content-Range {resp.headers.get('Content-Range')!r} "
                                f"does not match requested start {start} for {url}")
                        data = resp.read(want)
                        total = cr[2]
                        honored = True
                    elif status == 200:
                        if end + 1 > self.max_ignored_range_bytes:
                            raise WebStreamError(
                                f"server ignored Range (200) for {url}; slicing bytes {start}-{end} "
                                f"would need reading {end + 1} B > max_ignored_range_bytes="
                                f"{self.max_ignored_range_bytes}")
                        body = resp.read(end + 1)        # never read past what we need
                        data = body[start:end + 1]
                        cl = resp.headers.get("Content-Length")
                        total = int(cl) if cl and cl.isdigit() else None
                        honored = False
                        self.log(f"[webstream] WARNING server ignored Range (HTTP 200) for {url}; "
                                 f"read {len(body)} B and sliced {start}-{end} explicitly")
                    else:
                        raise WebStreamError(f"unexpected HTTP {status} for {url}")
                elapsed = time.monotonic() - t0
                eof = len(data) < want
                r = FetchResult(url=url, start=start, end=end, data=data, status=status,
                                source="network", range_honored=honored, total_length=total,
                                elapsed_s=elapsed, eof=eof, retries=attempt)
                self.log(f"[webstream] GET {url} Range bytes={start}-{end} -> {status} "
                         f"{len(data)} B in {elapsed:.3f}s")
                return r
            except urllib.error.HTTPError as e:
                if e.code == 416:
                    cr = _parse_content_range(e.headers.get("Content-Range"))
                    self.log(f"[webstream] GET {url} Range bytes={start}-{end} -> 416 (past end)")
                    return FetchResult(url=url, start=start, end=end, data=b"", status=416,
                                       source="network", range_honored=True,
                                       total_length=cr[2] if cr else None,
                                       elapsed_s=time.monotonic() - t0, eof=True, retries=attempt)
                if e.code in (429, 503):
                    ra = _parse_retry_after(e.headers.get("Retry-After"))
                    if ra is None:
                        ra = self.min_interval * (2 ** attempt)
                    if ra > self.max_retry_after:
                        raise WebStreamError(f"HTTP {e.code} for {url}: Retry-After {ra:.0f}s exceeds "
                                             f"max_retry_after={self.max_retry_after}s") from e
                    retry_wait = ra
                    last_err = e
                    self.log(f"[webstream] HTTP {e.code} for {url}; waiting {ra:.2f}s "
                             f"(attempt {attempt + 1}/{self.max_retries + 1})")
                    continue
                raise WebStreamError(f"HTTP {e.code} for {url} bytes={start}-{end}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise WebStreamError(f"network error for {url} bytes={start}-{end}: {e}") from e
        raise WebStreamError(f"gave up on {url} after {self.max_retries + 1} attempts "
                             f"(last: HTTP {getattr(last_err, 'code', '?')})")


def synthetic_bytes(n: int, offset: int = 0) -> bytes:
    """The original wrapper's sine bytes. Only for explicit, labelled fallback/testing."""
    return bytes(int(127 + 127 * math.sin((offset + i) * 0.1)) for i in range(n))


def fetch_bytes(url: str, start: int, length: int, fetcher: Optional[RangeFetcher] = None,
                fallback: str = "error") -> FetchResult:
    """Fetch `length` real bytes at `start`. fallback="error" (default) re-raises failures;
    fallback="synthetic" returns sine bytes labelled SYNTHETIC in the result and the log."""
    if fallback not in ("error", "synthetic"):
        raise ValueError("fallback must be 'error' or 'synthetic'")
    fetcher = fetcher or RangeFetcher()
    try:
        return fetcher.fetch(url, start, start + length - 1)
    except WebStreamError as e:
        if fallback == "error":
            raise
        fetcher.log(f"{SYNTHETIC_TAG} fetch failed ({e}); returning {length} SYNTHETIC sine bytes "
                    f"for {url} bytes={start}-{start + length - 1} — NOT internet data")
        return FetchResult(url=url, start=start, end=start + length - 1,
                           data=synthetic_bytes(length, start), status=None, source="SYNTHETIC",
                           synthetic=True)


class WebByteStream:
    """Iterate fixed-size chunks of a URL with resumable offsets.

    The next offset is written to `state_path` (JSON) after every chunk, so a new
    WebByteStream with the same state_path resumes where the last one stopped.
    """

    def __init__(self, url: str, chunk_size: int = 4096, start: int = 0,
                 end: Optional[int] = None, fetcher: Optional[RangeFetcher] = None,
                 state_path: Optional[str] = None, fallback: str = "error"):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.url = url
        self.chunk_size = chunk_size
        self.end = end                  # optional exclusive stop offset
        self.fetcher = fetcher or RangeFetcher()
        self.state_path = state_path
        self.fallback = fallback
        self.offset = start
        self.done = False
        self.synthetic_chunks = 0
        if state_path and os.path.exists(state_path):
            with open(state_path) as f:
                st = json.load(f)
            if st.get("url") != url:
                raise ValueError(f"state file {state_path} is for {st.get('url')!r}, not {url!r}")
            self.offset = int(st["offset"])
            self.done = bool(st.get("done", False))

    def _save(self):
        if not self.state_path:
            return
        d = os.path.dirname(self.state_path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"url": self.url, "offset": self.offset, "done": self.done,
                       "chunk_size": self.chunk_size,
                       "synthetic_chunks": self.synthetic_chunks}, f)
        os.replace(tmp, self.state_path)

    def next_chunk(self) -> Optional[FetchResult]:
        if self.done:
            return None
        n = self.chunk_size
        if self.end is not None:
            n = min(n, self.end - self.offset)
            if n <= 0:
                self.done = True
                self._save()
                return None
        r = fetch_bytes(self.url, self.offset, n, self.fetcher, self.fallback)
        if r.synthetic:
            self.synthetic_chunks += 1
        if not r.data:
            self.done = True
            self._save()
            return None
        self.offset += len(r.data)
        if r.eof or (r.total_length is not None and self.offset >= r.total_length):
            self.done = True
        self._save()
        return r

    def __iter__(self):
        while True:
            r = self.next_chunk()
            if r is None:
                return
            yield r


class ByteFeaturizer(torch.nn.Module):
    """Per-element features for a byte chunk: each byte value and its own absolute
    position are encoded separately with FourierFeatures (stage 9a) and concatenated.

    value v in 0..255  -> FourierFeatures(1, value_frequencies)(v / 255)
    position p (bytes) -> FourierFeatures(1, position_frequencies)(p / position_scale)

    forward(chunk uint8 [T] or [B, T], offset int) -> dict with
      ids      LongTensor [..., T]  raw byte ids (for an nn.Embedding(256, d) front end)
      value    [..., T]             v / 255
      position [..., T]             t = absolute offset / position_scale
      features [..., T, out_dim]    [gamma_v(value), gamma_p(position)]
    """

    def __init__(self, value_frequencies: int = 8, position_frequencies: int = 16,
                 position_scale: float = 2.0 ** 20, dtype=torch.float32):
        super().__init__()
        if position_scale <= 0:
            raise ValueError("position_scale must be > 0")
        self.value_ff = FourierFeatures(1, value_frequencies)
        self.pos_ff = FourierFeatures(1, position_frequencies)
        self.position_scale = float(position_scale)
        self.dtype = dtype

    @property
    def out_dim(self) -> int:
        return self.value_ff.out_dim + self.pos_ff.out_dim

    def forward(self, chunk: torch.Tensor, offset: int = 0) -> dict:
        if chunk.dtype != torch.uint8:
            raise TypeError(f"expected uint8 bytes, got {chunk.dtype}")
        ids = chunk.long()
        T = chunk.size(-1)
        value = chunk.to(self.dtype) / 255.0
        pos = (torch.arange(T, dtype=torch.float64) + offset) / self.position_scale
        pos = pos.to(self.dtype).expand(chunk.shape)
        feats = torch.cat([self.value_ff(value.unsqueeze(-1)), self.pos_ff(pos.unsqueeze(-1))], -1)
        return {"ids": ids, "value": value, "position": pos, "features": feats}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 9b: fetch a byte range over HTTP Range")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--cache-dir", default="runs/stage9b/cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--log", default=None, help="append a JSON line with the measured result")
    ap.add_argument("--fallback", choices=["error", "synthetic"], default="error")
    a = ap.parse_args(argv)
    fetcher = RangeFetcher(cache_dir=None if a.no_cache else a.cache_dir)
    t0 = time.monotonic()
    r = fetch_bytes(a.url, a.start, a.length, fetcher, a.fallback)
    wall = time.monotonic() - t0
    feats = ByteFeaturizer()(r.as_tensor(), offset=r.start)["features"]
    rec = r.summary()
    rec.update({"wall_s": round(wall, 4), "sha256": hashlib.sha256(r.data).hexdigest(),
                "feature_shape": list(feats.shape), "user_agent": fetcher.user_agent,
                "measured_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "preview": r.data[:160].decode("utf-8", errors="replace")})
    tag = SYNTHETIC_TAG + " " if r.synthetic else ""
    print(f"{tag}{json.dumps(rec, ensure_ascii=False, indent=1)}")
    if a.log:
        os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
        with open(a.log, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


if __name__ == "__main__":
    main()
