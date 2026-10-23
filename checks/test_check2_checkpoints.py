"""Check 2: save a real state_dict to disk, load it into a FRESH instance built
with a DIFFERENT seed, and require bit-identical eval-mode predictions (torch.equal,
max |diff| == 0). Each model is first trained for a few optimizer steps so the
checkpoint is not just the init."""
import os
import pytest
import torch
import torch.nn.functional as F
from checks._load import load
from checks import historical as H
import physics_simulation as P
from model import DynamicMultiBlockTransformer

BYTE = load("bytelm/artemis_corpus_benchmark.py", "byte_bench")
LEDGER = load("bytelm/ledger_check/artemis_fixed_benchmark.py", "ledger_bench")


def _bytes(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (4, 128), generator=g)


def _coords(seed):
    return torch.rand(512, 3, generator=torch.Generator().manual_seed(seed))


def _seq(seed):
    return torch.randn(3, 16, 32, generator=torch.Generator().manual_seed(seed))


def lm_step(m, x):
    return F.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())


CASES = {
    "bytelm.ByteModel": (lambda: BYTE.ByteModel(128), _bytes, lm_step, lambda m, x: m(x)),
    "ledger.ArtemisFixedLanguageModel": (lambda: LEDGER.ArtemisFixedLanguageModel(), _bytes, lm_step, lambda m, x: m(x)),
    "historical_siren_study": (lambda: H.build_historical_siren(out_init="study"), _coords,
                               lambda m, x: (m(x) - H.target(x)).pow(2).mean(), lambda m, x: m(x)),
    "physics_moe": (lambda: P.PhysicsMoELayer(4, 2, hidden=16), _coords,
                    lambda m, x: sum(P.losses(m, 64, torch.Generator().manual_seed(1))[:3]),
                    lambda m, x: m(x)[0]),
}
for legacy in (True, False):
    for dispatch in (("loop",) if legacy else ("loop", "vectorized")):
        for causal in (False, True):
            name = f"moe_transformer legacy={legacy} dispatch={dispatch} causal={causal}"
            CASES[name] = (
                (lambda l=legacy, d=dispatch, c=causal: DynamicMultiBlockTransformer(
                    32, 4, 64, 4, num_layers=2, top_k=2, capacity_factor=1.0, legacy=l, causal=c, dispatch=d)),
                _seq, lambda m, x: (lambda o: o[0].pow(2).mean() + 0.01 * o[1])(m(x)), lambda m, x: m(x)[0])


@pytest.mark.parametrize("name", list(CASES))
def test_checkpoint_roundtrip_identical(name, tmp_path):
    make, data, loss_fn, pred = CASES[name]
    torch.manual_seed(0)
    m = make()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x_train = data(1)
    m.train()
    for _ in range(3):
        opt.zero_grad(); loss_fn(m, x_train).backward(); opt.step()
    m.eval()
    x = data(123)
    with torch.no_grad():
        before = pred(m, x)
    path = tmp_path / "ckpt.pt"
    torch.save({k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, path)
    torch.manual_seed(999)  # fresh instance with different random init
    fresh = make()
    with torch.no_grad():
        differs = not torch.equal(pred(fresh.eval(), x), before)
    fresh.load_state_dict(torch.load(path, weights_only=True))
    fresh.eval()
    with torch.no_grad():
        after = pred(fresh, x)
    diff = (after - before).abs().max().item()
    print(f"{name}: fresh-init differs={differs} reload max|diff|={diff}")
    assert differs
    assert torch.equal(before, after), diff
