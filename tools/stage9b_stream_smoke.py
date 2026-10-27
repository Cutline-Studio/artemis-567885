"""Stage 9b real smoke: stream 3 chunks of a Gutenberg book with WebByteStream, stop after 2,
resume from the state file, and log status / bytes / timing / request spacing."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import webstream as ws

URL = ws.DEFAULT_URL
OUT = "runs/stage9b"
os.makedirs(OUT, exist_ok=True)
state = os.path.join(OUT, "stream_state.json")
if os.path.exists(state):
    os.remove(state)
logs = []
f = ws.RangeFetcher(cache_dir=None, log=lambda m: (logs.append(m), print(m)))
recs = []
s1 = ws.WebByteStream(URL, chunk_size=2048, start=100_000, end=100_000 + 3 * 2048, fetcher=f,
                      state_path=state)
for _ in range(2):
    recs.append(s1.next_chunk().summary())
offset_after_stop = json.load(open(state))["offset"]
s2 = ws.WebByteStream(URL, chunk_size=2048, end=100_000 + 3 * 2048, fetcher=f, state_path=state)
resumed_from = s2.offset
recs += [c.summary() for c in s2]
gaps = [round(b - a, 3) for a, b in zip(f.request_times, f.request_times[1:])]
out = {"url": URL, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
       "chunks": recs, "offset_after_stop": offset_after_stop, "resumed_from": resumed_from,
       "final_state": json.load(open(state)), "request_gaps_s": gaps,
       "network_requests": f.network_requests, "log": logs}
with open(os.path.join(OUT, "stream_smoke.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: out[k] for k in ("offset_after_stop", "resumed_from", "request_gaps_s",
                                      "network_requests", "final_state")}))
