"""Check 1: historical target, normalisation and init (bitwise, torch.equal)."""
import importlib.util, math, os
import pytest
import torch
from checks import historical as H

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


FC = load("pilot/frequency_capacity_study.py", "fc_study")
PILOT = load("pilot/artemis_physics_pilot.py", "pilot_mod")
BENCH = load("pilot/artemis_physics_benchmark.py", "bench_mod")
RAW = H.sample_raw(4096, torch.Generator().manual_seed(20261023))


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_target_matches_study(k):
    assert torch.equal(H.target(RAW, k), FC.analytical(RAW, k))


def test_target_matches_pilot_benchmark_k1():
    assert torch.equal(H.target(RAW, 1), BENCH.analytical_solver(RAW))
    # pilot inlines the same formula
    assert torch.equal(H.target(RAW, 1), torch.sin(RAW[:, 0:1] - 0.5 * RAW[:, 2:3]))


def test_sampling_matches_study():
    g1, g2 = torch.Generator().manual_seed(20261023), torch.Generator().manual_seed(20261023)
    assert torch.equal(H.sample_raw(2000, g1), FC.sample(2000, g2))


def test_normalisation_matches_study_and_maps_to_unit_cube():
    assert torch.equal(H.normalize(RAW), FC.featurise(RAW, True, True))
    assert torch.equal(RAW, FC.featurise(RAW, False, True))  # norm=0 == raw (pilot)
    z = H.normalize(RAW)
    assert z.min() >= -1.0 and z.max() <= 1.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_init_matches_study_build_model(seed):
    torch.manual_seed(seed); a = FC.build_model("siren", 3, 32, 2, 30.0, 30.0)
    torch.manual_seed(seed); b = H.build_historical_siren(3, 32, 2, 30.0, 30.0, out_init="study")
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_init_matches_pilot_model(seed):
    for mod in (PILOT, BENCH):
        torch.manual_seed(seed); a = mod.ContinuousWaveFieldSurrogate(hidden_dim=32)
        torch.manual_seed(seed); b = H.build_historical_siren(3, 32, 2, 30.0, 30.0, out_init="pilot")
        for p, q in zip(a.parameters(), b.parameters()):
            assert torch.equal(p, q)


def test_FINDING_study_output_init_differs_from_pilot():
    """The study's 'original k=1 setup' ablation is NOT bit-identical to the pilot model:
    hidden layers match exactly, the output-layer WEIGHT is re-initialised by the study."""
    torch.manual_seed(0); a = PILOT.ContinuousWaveFieldSurrogate(hidden_dim=32)
    torch.manual_seed(0); b = FC.build_model("siren", 3, 32, 2, 30.0, 30.0)
    pa, pb = list(a.named_parameters()), list(b.named_parameters())
    same = {na: torch.equal(p, q) for (na, p), (_, q) in zip(pa, pb)}
    print("pilot vs study param equality:", same)
    assert all(v for n, v in same.items() if not n.startswith("net.2"))
    assert not same["net.2.weight"] and same["net.2.bias"]


def test_physics_simulation_is_exploratory_and_deviates():
    import physics_simulation as P
    assert "EXPLORATORY" in P.__doc__
    # different target/domain
    raw01 = torch.tensor([[0.3, 0.2, 0.5]])
    assert not torch.allclose(P.exact(raw01[:, :1], raw01[:, 1:2], raw01[:, 2:3]), H.target(raw01))
    # SIREN weights use the historical bound, biases do not (nn.Linear default)
    torch.manual_seed(0); s = P.SineLayer(64, 64, omega0=30.0)
    b = math.sqrt(6 / 64) / 30
    assert s.linear.weight.abs().max() <= b
    assert s.linear.bias.abs().max() > b  # default bias bound 1/sqrt(64)=0.125 >> b=0.0102
