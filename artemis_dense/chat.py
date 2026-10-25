"""Local REPL chat: python -m artemis_dense.chat [--size tiny] [--ckpt path] [--no-rag]
Loads the ONE dense model through the transformer engine. Without a trained checkpoint the output
is gibberish; the banner says so. Commands: /reset, /facts, /quit.
"""
import argparse, os, sys, tempfile

from .engines import transformer as tx
from .engines.conversation import ChatEngine


def build_engine(size="tiny", ckpt=None, rag=True, seed=0):
    model = tx.load(ckpt) if ckpt else tx.build(size, seed=seed)
    r = None
    if rag:
        from .engines.rag import RagEngine
        r = RagEngine(os.path.join(tempfile.gettempdir(), "artemis-rag-index"))
        r.refresh()
    from .engines.heart import HeartEngine
    return ChatEngine(model, rag=r, heart=HeartEngine(), trained=bool(ckpt), seed=seed)


def main(argv=None, stdin=None, stdout=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--ckpt")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args(argv)
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    eng = build_engine(a.size, a.ckpt, not a.no_rag)
    eng.max_new = a.max_new
    if not eng.trained:
        print("[Artemis] UNTRAINED weights: replies are byte-level gibberish except when a doc answer is "
              "retrieved (shown with citations). Load a trained --ckpt for real chat.", file=stdout)
    while True:
        stdout.write("you> "); stdout.flush()
        line = stdin.readline()
        if not line or line.strip() in ("/quit", "/exit"):
            break
        line = line.strip()
        if line == "/reset":
            eng.state.__init__(eng.state.system.content); continue
        if line == "/facts":
            print(eng.state.facts, file=stdout); continue
        if not line:
            continue
        r = eng.respond(line)
        print("artemis> " + r["reply"].replace("\x00", ""), file=stdout)
        if r["citations"]:
            print("  sources: " + ", ".join(r["citations"]), file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
