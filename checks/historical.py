"""Historical (restored) target, input normalisation and SIREN initialisation.

Source of truth, read from the repo's own pilot code (not re-derived):
  * pilot/artemis_physics_pilot.py, pilot/artemis_physics_benchmark.py (k = 1, raw inputs)
  * pilot/frequency_capacity_study.py (PR #3; k in {1,2,4,8}, normalised inputs)

TARGET         h(x, y, t) = sin(k (x - 0.5 t)),  x, y ~ U[0, 2pi], t ~ U[0, 2]
NORMALISATION  x/pi - 1, y/pi - 1, t - 1   (maps the raw domain to [-1, 1]^3)
               (the original pilot used raw, un-normalised inputs; the study's
               ablation phase reproduces that as norm=0)
SIREN INIT     every SIREN linear layer: weight AND bias ~ U(-b, b) with
               b = 1/fan_in for the first layer, sqrt(6/fan_in)/w0 otherwise,
               applied right after nn.Linear's default init (same RNG order).
OUTPUT LAYER   two historical variants (they differ; see checks/test_check1_historical.py):
               "pilot": nn.Linear default init (artemis_physics_pilot / benchmark)
               "study": weight re-drawn ~ U(-sqrt(6/fan_in)/w0_last, +...), bias default
               (frequency_capacity_study.build_model)

Anything that deviates from this (e.g. physics_simulation.py: Gaussian advection
target on [-1,1]^2 x [0,1], default-initialised SIREN biases) is EXPLORATORY.
"""
import math
import torch
import torch.nn as nn

C = 0.5
TWO_PI = 2.0 * math.pi


def target(raw: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    return torch.sin(k * (raw[:, 0:1] - C * raw[:, 2:3]))


def normalize(raw: torch.Tensor) -> torch.Tensor:
    return torch.cat([raw[:, 0:1] / math.pi - 1.0, raw[:, 1:2] / math.pi - 1.0, raw[:, 2:3] - 1.0], dim=1)


def sample_raw(n: int, gen: torch.Generator = None) -> torch.Tensor:
    x = torch.rand(n, 1, generator=gen) * 2 * math.pi
    y = torch.rand(n, 1, generator=gen) * 2 * math.pi
    t = torch.rand(n, 1, generator=gen) * 2.0
    return torch.cat([x, y, t], dim=1)


class HistSiren(nn.Module):
    def __init__(self, fin, fout, w0, is_first):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(fin, fout)
        with torch.no_grad():
            b = 1.0 / fin if is_first else math.sqrt(6.0 / fin) / w0
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


def build_historical_siren(fin=3, width=32, depth=2, w0_first=30.0, w0_hidden=30.0, out_init="study"):
    if out_init not in ("pilot", "study"):
        raise ValueError(out_init)
    layers, f = [], fin
    for i in range(depth):
        layers.append(HistSiren(f, width, w0_first if i == 0 else w0_hidden, is_first=(i == 0)))
        f = width
    out = nn.Linear(f, 1)
    if out_init == "study":
        with torch.no_grad():
            b = math.sqrt(6.0 / f) / (w0_hidden if depth > 1 else w0_first)
            out.weight.uniform_(-b, b)
    layers.append(out)
    return nn.Sequential(*layers)


def flat_params(m: nn.Module):
    return [p.detach().clone() for p in m.parameters()]
