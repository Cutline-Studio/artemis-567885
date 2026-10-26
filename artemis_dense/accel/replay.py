"""Method 19: mistake-first replay.

Two sources, mistakes first:
  1. engines/mistakes.MistakeJournal corrective samples (context + verified fix), when a journal dir
     is given - these are the model's recorded real mistakes and are replayed before anything else;
  2. a hard-example buffer: the training windows with the highest per-sample loss so far.
A fraction `replay_frac` of each batch is drawn from these; the rest is fresh data.
"""
import heapq, itertools
from typing import List, Optional

import torch


class MistakeReplay:
    def __init__(self, capacity: int = 256, replay_frac: float = 0.25, journal_dir: str = "", codec=None):
        self.capacity, self.frac = capacity, replay_frac
        self.heap: list = []                      # (loss, tiebreak, x, y) min-heap of the hardest windows
        self._c = itertools.count()
        self.journal: List[List[int]] = []
        if journal_dir:
            from ..engines.mistakes import MistakeJournal
            enc = codec.encode if codec else (lambda s: list(s.encode()))
            self.journal = [enc(s.decode("utf-8", "replace")) for s in MistakeJournal(journal_dir).corrective_samples()]

    def __len__(self):
        return len(self.heap)

    @torch.no_grad()
    def observe(self, x, y, per_sample_loss):
        for i, l in enumerate(per_sample_loss.tolist()):
            item = (l, next(self._c), x[i].clone(), y[i].clone())
            if len(self.heap) < self.capacity:
                heapq.heappush(self.heap, item)
            elif l > self.heap[0][0]:
                heapq.heapreplace(self.heap, item)

    def mix(self, x, y, g: Optional[torch.Generator] = None):
        """Overwrite the first round(B*frac) rows: journal mistakes first, then hardest windows."""
        B, T = x.shape
        n = int(round(B * self.frac))
        if n == 0 or (not self.heap and not self.journal):
            return x, y, 0
        x, y = x.clone(), y.clone()
        used = 0
        for s in self.journal[:n]:
            buf = torch.tensor((s * (T // max(1, len(s)) + 2))[:T + 1])
            x[used], y[used] = buf[:T], buf[1:T + 1]; used += 1
        pool = [h for h in self.heap if h[2].shape[0] == T]
        while used < n and pool:
            j = int(torch.randint(0, len(pool), (1,), generator=g))
            x[used], y[used] = pool[j][2], pool[j][3]; used += 1
        return x, y, used


def per_sample_loss(logits, y):
    l = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")
    return l.mean(-1).detach()
