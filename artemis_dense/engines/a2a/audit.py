"""Append-only, hash-chained JSONL audit log (tamper-evident, not tamper-proof)."""
import hashlib, json, os, threading, time


class AuditLog:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._prev = "0" * 64
        if os.path.exists(self.path):
            for line in open(self.path):
                self._prev = json.loads(line)["hash"]

    def record(self, event: str, allowed: bool, **fields) -> dict:
        with self._lock:
            rec = {"ts": time.time(), "event": event, "allowed": allowed, **fields, "prev": self._prev}
            rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            self._prev = rec["hash"]
            return rec

    def entries(self):
        return [json.loads(l) for l in open(self.path)] if os.path.exists(self.path) else []

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for rec in self.entries():
            h = rec.pop("hash")
            if rec["prev"] != prev or hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest() != h:
                return False
            prev = h
        return True
