"""Stage 1/2 trainer, assistant-only SFT masking, bits-per-byte, and the byte|bpe tokenizer switch."""
import json, math, os

import pytest
import torch

from artemis_dense.config import DenseConfig
from artemis_dense.engines import transformer as tx
from artemis_dense.engines.conversation import END, TOK
from artemis_dense.tok import ByteCodec, get_codec
from artemis_dense import train as T

BPE = os.path.join(os.path.dirname(T.PKG), "artemis_dense", "data", "tokenizer", "bpe4k", "tokenizer.json")
tokenizers = pytest.importorskip("tokenizers")


def test_byte_is_default_and_old_configs_load():
    c = DenseConfig()
    assert c.tokenizer == "byte" and c.vocab_size == 256
    old = {k: v for k, v in c.to_dict().items() if k not in ("tokenizer", "tokenizer_path")}
    assert DenseConfig.from_dict(old).tokenizer == "byte"
    with pytest.raises(ValueError):
        DenseConfig(tokenizer="wordpiece")


def test_validate_byte_vs_bpe():
    assert tx.validate(DenseConfig(vocab_size=4096)) != []               # byte needs 256
    assert any("tokenizer_path" in p for p in tx.validate(DenseConfig(vocab_size=4096, tokenizer="bpe")))
    assert tx.validate(DenseConfig(vocab_size=4096, tokenizer="bpe", tokenizer_path=BPE)) == []


def test_split_is_deterministic_interleaved_and_disjoint():
    text = "".join(f"line {i} " + "x" * 50 + "\n" for i in range(3000))
    tr, va = T.split_corpus(text)
    assert (tr, va) == T.split_corpus(text)
    assert 0.03 < len(va) / len(text) < 0.07
    assert set(tr.splitlines()).isdisjoint(va.splitlines())
    assert "line 50 " in tr and any(l.startswith("line 2") for l in va.splitlines())


def test_sft_mask_covers_assistant_tokens_only():
    c = ByteCodec()
    x, y = T.encode_masked([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."}], c,
                           system="S")
    assert len(x) == len(y)
    trained = bytes(t for t in y if t != T.IGNORE).decode()
    assert trained == "Hello." + END


@pytest.mark.skipif(not os.path.exists(BPE), reason="fixed tokenizer not built")
def test_bpe_chat_tokens_are_single_ids_and_round_trip():
    c = get_codec(kind="bpe", path=BPE)
    assert c.vocab_size == 4096
    for t in list(TOK.values()) + [END]:
        assert len(c.encode(t)) == 1, t
    s = f"{TOK['user']}define serendipity{END}\n{TOK['assistant']}héllo — ok{END}"
    assert c.decode(c.encode(s)) == s
    x, y = T.encode_masked([{"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}], c)
    assert c.decode([t for t in y if t != T.IGNORE]) == "answer" + END


@pytest.mark.skipif(not os.path.exists(BPE), reason="fixed tokenizer not built")
def test_bpe_vocab_mismatch_rejected():
    with pytest.raises(ValueError):
        get_codec(DenseConfig(vocab_size=5000, tokenizer="bpe", tokenizer_path=BPE))


def test_bits_per_byte_uniform_model_is_8_for_bytes():
    m = tx.build("tiny", seed=0, context_len=64)
    for p in m.parameters():
        torch.nn.init.zeros_(p)
    r = T.bits_per_byte(m, ByteCodec(), "abc" * 100)
    assert abs(r["bits_per_byte"] - 8.0) < 1e-4 and r["tokens"] == 299


def test_generate_ids_stops_and_respects_context():
    m = tx.build("tiny", seed=0, context_len=32)
    out = tx.generate_ids(m, [1, 2, 3], 100)
    assert len(out) == 29
    first = tx.generate_ids(m, [1, 2, 3], 1)[0]
    assert tx.generate_ids(m, [1, 2, 3], 10, stop_ids=[first]) == []
    assert bytes(out) == tx.generate(m, bytes([1, 2, 3]), 100)            # same greedy path as bytes API


@pytest.mark.parametrize("kind", ["byte", "bpe"])
def test_train_cli_smoke_reduces_loss_and_checkpoint_loads(tmp_path, kind):
    if kind == "bpe" and not os.path.exists(BPE):
        pytest.skip("fixed tokenizer not built")
    corpus = tmp_path / "c.txt"
    corpus.write_text("".join(f"<doc>Artemis note {i}: the sandbox keeps work local.</doc>\n" for i in range(1200)) +
                      "".join(f'<define word="w{i}" pos="noun" source="t">a thing {i}</define>\n' for i in range(200)))
    args = ["--size", "tiny", "--context-len", "64", "--run", "t", "--out", str(tmp_path), "--corpus", str(corpus),
            "--stage1-steps", "30", "--stage2-steps", "6", "--batch", "4", "--sft-batch", "2", "--warmup", "2",
            "--eval-every", "15", "--threads", "2", "--tokenizer", kind]
    if kind == "bpe":
        args += ["--tokenizer-path", BPE]
    s = T.main(args)
    assert s["after_stage1"]["bits_per_byte"] < s["before"]["bits_per_byte"]
    assert set(s["samples_after_stage2"]) == set(T.SAMPLE_PROMPTS)
    for k in ("seed", "versions", "command", "stage1", "stage2"):
        assert k in s
    m = tx.load(str(tmp_path / "t" / "stage2_final.pt"))
    assert m.cfg.tokenizer == kind and m.trained_info["stage1_steps"] == 30
    assert json.load(open(tmp_path / "t" / "summary.json"))["run"] == "t"


@pytest.mark.skipif(not os.path.exists(BPE), reason="fixed tokenizer not built")
def test_chat_engine_runs_with_bpe_model(tmp_path):
    from artemis_dense.engines.conversation import ChatEngine
    m = tx.build("tiny", seed=0, context_len=512, tokenizer="bpe", vocab_size=4096, tokenizer_path=BPE)
    e = ChatEngine(m, max_new_tokens=8, trained=True)
    r = e.respond("My name is Latoya. What can you do?")
    assert isinstance(r["reply"], str) and r["model_trained"] is True
