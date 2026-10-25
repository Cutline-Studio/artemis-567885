import time
from typing import Dict, Tuple


class RateLimiter:
    """Token bucket per key: `rate` ops per `per_s` seconds."""

    def __init__(self, rate: int, per_s: float = 60.0, clock=time.time):
        self.rate, self.per_s, self.clock = rate, per_s, clock
        self.buckets: Dict[str, Tuple[float, float]] = {}

    def allow(self, key: str, rate: int = None) -> bool:
        rate = rate or self.rate
        now = self.clock()
        tokens, last = self.buckets.get(key, (float(rate), now))
        tokens = min(rate, tokens + (now - last) * rate / self.per_s)
        if tokens < 1:
            self.buckets[key] = (tokens, now)
            return False
        self.buckets[key] = (tokens - 1, now)
        return True
