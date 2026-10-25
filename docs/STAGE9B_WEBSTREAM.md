# Stage 9(b): webstream (HTTP Range byte streaming)

`webstream.py` streams real text bytes over HTTP Range requests without a tokenizer. It is
ported from `docs/reference/clf_stream_original.py`. Nothing else in the repo imports it yet
(stage 9c will), and no existing module or default changed.

## API
- `RangeFetcher(cache_dir=None, user_agent=USER_AGENT, min_interval=1.0, timeout=20, max_retries=3, max_retry_after=120, max_ignored_range_bytes=8 MiB)`
  - `.fetch(url, start, end)` fetches bytes `[start, end]` (end inclusive) and returns a `FetchResult` (`data`, `status`, `source` = network/cache/SYNTHETIC, `range_honored`, `total_length`, `elapsed_s`, `eof`, `retries`, `synthetic`).
  - It sends `Range: bytes=start-end`, `Accept-Encoding: identity` and an identifying User-Agent.
  - **206**: Content-Range must start at the requested `start`, otherwise it raises `WebStreamError`. At most `end-start+1` bytes are read.
  - **200** (server ignored Range): reads at most `end+1` bytes, slices `[start:end+1]` explicitly, logs a warning and sets `range_honored=False`. If `end+1 > max_ignored_range_bytes` it raises instead of downloading the whole file.
  - **416**: start is past the end of the file. Returns empty data with `eof=True`.
  - **429/503**: waits for Retry-After (seconds or HTTP-date; exponential backoff from `min_interval` if the header is missing), then retries. It gives up loudly after `max_retries`, or if Retry-After is longer than `max_retry_after`.
  - Any other HTTP status or network error raises `WebStreamError`.
  - Rate limit: at least `min_interval` (default 1.0 s) between network sends, retries included. Cache hits don't wait.
  - Cache: `cache_dir/<sha256(url + range)>.bin` plus a `.json` sidecar (url, range, status, sha256). A hit sends nothing over the network. A truncated cache file is refetched. Synthetic data is never cached.
- `fetch_bytes(url, start, length, fetcher, fallback="error")`: the default re-raises failures. Only an explicit `fallback="synthetic"` returns the original wrapper's sine bytes, and then `synthetic=True`, `source="SYNTHETIC"`, a `label` of "SYNTHETIC (not internet data)", and a `[SYNTHETIC] ... NOT internet data` log line. The CLI prefixes its output with `[SYNTHETIC]` in that case.
- `WebByteStream(url, chunk_size, start, end, fetcher, state_path, fallback)`: iterates chunks. After every chunk it writes `{url, offset, done, synthetic_chunks}` atomically to `state_path`, so a new stream resumes from that offset. A URL mismatch raises an error. The stream stops on 416, on a short chunk, or when the offset reaches Content-Range's total.
- `ByteFeaturizer(value_frequencies=8, position_frequencies=16, position_scale=2**20)`: produces per-byte features `[FourierFeatures(1, Lv)(v/255), FourierFeatures(1, Lp)(abs_offset/position_scale)]` using `fourier.py` from 9(a). Each byte is encoded on its own (no window mixing), and it has no parameters. It also returns raw `ids` (LongTensor 0..255) for the `nn.Embedding(256, d)` front end used in 9(c). Everything is in torch.

## Fixes vs the reference wrapper
1. Its URL (`w3.org/TR/PNG/iso_8859-1.txt`) returns 404, after which it silently fell back to sine bytes. Now a failure raises, and the fallback is opt-in and labelled in every output.
2. It never sent a Range header (it just read the first N bytes). Now a real Range is sent and 206 is required, with 200 sliced and bounded.
3. The source is now a Project Gutenberg plain-text book (Pride and Prejudice, `https://www.gutenberg.org/cache/epub/1342/pg1342.txt`).
4. It mixed the whole window into one vector with a random B drawn per window length (np.random.seed reset globally). Now each element gets its own value and position features.
5. NumPy has been replaced with torch.

## Real smoke fetch (measured 2026-10-25, ET)
`python3 webstream.py --start 0 --length 4096 --log runs/stage9b/smoke.jsonl` (default URL above):

| when (EDT) | URL | Range | status | bytes | time | source |
|---|---|---|---|---|---|---|
| 18:08:18 | pg1342.txt | 0-4095 | 206 (Content-Range total 772,386) | 4096 | 0.289 s | network |
| 18:08:18 | pg1342.txt | 0-4095 | 206 (cached) | 4096 | 0.0001 s | cache (no request) |

- sha256 of the 4096 bytes: `0e7606040046b52126c52ec2e191f5b972be9c37559566baf7908a935e1cfe78`. A separate `curl -H "Range: bytes=0-4095"` of the same URL gives the same hash.
- The text starts with "The Project Gutenberg eBook of Pride and Prejudice". The per-byte feature tensor is [4096, 48] (2·8 value + 2·16 position).
- Resumable stream (`tools/stage9b_stream_smoke.py`, `runs/stage9b/stream_smoke.json`) read 3 × 2048-byte chunks from offset 100,000, stopping after 2. The state file then held offset 104,096, and a new stream resumed from 104,096 and finished at 106,144 (`done: true`). All 3 requests returned 206 with 2048 B in 0.280–0.288 s, and the measured gaps between sends were 1.0 s and 1.0 s. That was 3 network requests in total, with nothing refetched.

## Tests
`tests/test_webstream.py` (44 tests, about 5 s) runs against a local `http.server` in a thread. It covers:
- exact byte ranges, including a range past EOF that gets clipped
- the Range header that is sent, and the User-Agent
- 416 as EOF
- 200 sliced, the read limit on 200, and 206 with a wrong Content-Range rejected
- cache hits sending nothing (same fetcher and a new fetcher), cache keyed by URL + range, truncated cache refetched, cache bypass
- default interval ≥ 1 s; real-time spacing ≥ 0.3 s; fake-clock spacing of exactly 1.0 s; cache hits don't sleep
- 429 with Retry-After: 1 (real wait ≥ 0.95 s), HTTP-date Retry-After, the wait being max(Retry-After, interval), persistent 429 giving up, and an over-long Retry-After raising
- 404 and connection refused raising by default; synthetic fallback labelled in the result, logs and CLI and never cached; real data never labelled synthetic
- streaming a whole document in chunks, resuming from a state file without refetching, state/URL mismatch, start/end bounds, stopping at the Content-Range total without an extra request, counting synthetic chunks
- per-element features matching a FourierFeatures reference, value and position independence, no neighbour mixing, offset and batch shapes, features in [-1, 1], no parameters, global RNG untouched
