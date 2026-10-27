"""BRKCOM-2115 sizing formulas (tools/artemis_sizing.py), checked against the deck's own worked examples."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import artemis_sizing as sz  # noqa: E402

GB = 1e9


def test_llama_8b_examples():
    assert sz.raw_size_bytes(8.03e9, "bf16") / GB == pytest.approx(16.06)
    assert sz.kv_bytes_per_token(32, 4096, 32, 8) == 131072
    assert sz.inference_bytes(8.03e9, 32, 4096, 32, 8, 131072, overhead=1.0) / GB == pytest.approx(33.24, abs=0.01)
    t = sz.training_bytes(8.03e9)
    assert t / GB == pytest.approx(120.45) and sz.gpus_needed(t, 80) == 2 and sz.gpus_needed(t, 141) == 1


def test_405b_examples():
    assert sz.raw_size_bytes(405e9, "bf16") / GB == pytest.approx(810)
    assert sz.raw_size_bytes(405e9, "bf16") * sz.INFERENCE_OVERHEAD / GB == pytest.approx(972)
    assert sz.raw_size_bytes(405e9, "int4") / GB == pytest.approx(202.5)
    assert sz.parallelism(405e9, 8).startswith("hybrid")


def test_table_rows():
    rows = sz.main(["--json"])
    assert len(rows) == len(sz.STAGES)
    big = {r["name"]: r for r in rows}["Llama 3.1 405B (deck example)"]
    assert big["training_gpus"]["H100-80GB"] == 76 and big["inference_gpus_bf16"]["B200-192GB"] == 6
