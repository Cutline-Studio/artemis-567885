import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from types import SimpleNamespace
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("corpus_benchmark", ROOT / "experiments" / "corpus_benchmark.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def _docs(n=6, seed=0):
    rng = random.Random(seed)
    return [bytes(rng.randrange(97, 123) for _ in range(rng.randrange(20, 90))) for _ in range(n)]


def test_shuffled_windows_score_each_target_once():
    docs = _docs()
    ordered = list(cb.windows(docs, 16))
    shuffled = list(ordered)
    random.Random(1).shuffle(shuffled)
    assert sorted(shuffled) == sorted(ordered) and shuffled != ordered
    targets = sum(int((y != -100).sum()) for _, y in cb.batches(docs, 16, 4, "cpu", chunks=shuffled))
    assert targets == sum(len(d) - 1 for d in docs)


def _corpus(tmp_path):
    rng = random.Random(3)
    for split, n in (("train", 8), ("val", 2), ("test", 2)):
        (tmp_path / "c" / split).mkdir(parents=True)
        for i in range(n):
            words = " ".join(rng.choice(["the", "sea", "ship", "wind", "a", "and"]) for _ in range(rng.randrange(30, 80)))
            (tmp_path / "c" / split / f"{i}.txt").write_text(f"{split}{i} {words}\n")
    return tmp_path / "c"


def _run(tmp_path, name, **kw):
    args = dict(data=_corpus(tmp_path) if not (tmp_path / "c").exists() else tmp_path / "c",
                output=tmp_path / name, seed=5, context=16, batch_size=4, epochs=1, steps=5, lr=1e-3,
                threads=1, device="cpu", model="dense", position="learned", experts=4,
                capacity_factor=1.25, capacity_mode="none", shuffle_windows=True, eval_every=2,
                aux_weight=0.01)
    args.update(kw)
    cb.main(SimpleNamespace(**args))
    out = tmp_path / name
    return [json.loads(l) for l in (out / "validation.jsonl").read_text().splitlines()], \
        json.loads((out / "metrics.json").read_text())


def test_step_based_validation_and_selection(tmp_path):
    val, metrics = _run(tmp_path, "dense")
    assert [r["optimizer_steps"] for r in val] == [0, 2, 4, 5]
    assert metrics["best_validation"]["neural_bpb"] == min(r["neural_bpb"] for r in val)


@pytest.mark.parametrize("position", ["learned", "sinusoidal", "fourier"])
def test_moe_arm_runs_dropless_and_is_deterministic(tmp_path, position):
    _, a = _run(tmp_path, f"a-{position}", model="moe", position=position)
    _, b = _run(tmp_path, f"b-{position}", model="moe", position=position)
    assert a["test"] == b["test"]
    assert a["test"]["dropped_assignment_fraction"] == 0.0


def test_matched_arms_consume_identical_training_targets(tmp_path):
    _, dense = _run(tmp_path, "d")
    _, moe = _run(tmp_path, "m", model="moe", position="fourier")
    assert dense["training_target_bytes"] == moe["training_target_bytes"]
