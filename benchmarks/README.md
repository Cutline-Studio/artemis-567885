# Stage 4: loop vs vectorized MoE dispatch (measured, CPU)

Script: [`bench.py`](../bench.py). Raw consoles and JSON: [`results/`](results/).

```
python3 bench.py --threads 1 --json benchmarks/results/stage4_bench_threads1.json
python3 bench.py --threads 8 --json benchmarks/results/stage4_bench_threads8.json
```

Run 2026-10-24 between 09:05 and 09:13 ET on the shared Linux box (no GPU; CUDA unavailable).
Environment: Python 3.13.5, torch 2.14.1+cpu, CPU "Intel(R) Xeon(R) Processor" (8 logical CPUs),
Linux-6.12.94+-x86_64-with-glibc2.41. Grid: N in {1024, 4096, 16384} tokens x E in {4, 16, 64} x d_model in {64, 256},
d_ff = 4·d, top_k = 2, capacity_factor = 1.0, float32 timing. Warmup 5, 20 timed repeats per
path and mode, loop/vectorized interleaved (order alternates every repeat). Seed base 567885
(per config: 567885 + 1000·E + d; routing-noise seed 567885 + repeat index). speedup = loop median /
vectorized median (>1 means vectorized is faster). The box is shared, so some p90 values
show interference (e.g. 4096/64/64 fwd at 1 thread); medians are the headline numbers.

## Parity (asserted before any timing, every config, both thread counts)
All 18 configs x {eval, train} passed. float64: top-k routing, kept slots, drop counts,
outputs, aux loss and input gradients identical; parameter gradients max relative diff
1.6e-15 (tolerance rtol=atol=1e-12). float32: routing/kept/drops identical, outputs and
input grads max abs diff 0.0 at 1 thread and 4.2e-7 / 2.4e-7 at 8 threads (tol rtol=1e-5,
atol=1e-6); parameter-gradient max relative diff 1.07e-6, reported not asserted (same
reduction-order effect as the strict xfail in checks/test_check3_moe_dispatch.py).
No idle experts occurred in the grid.

Found while writing this stage: when an expert receives no tokens, the loop path leaves
its `.grad` as `None` and the vectorized path gives an all-zero `.grad`. Values agree,
but optimizers treat them differently (Adam/AdamW skip `None` params; a zero grad still
applies momentum and weight decay). `bench.compare_bench` allows exactly this case,
asserts the extra grads are exactly zero, and counts them; documented by
`tests/test_bench.py::test_idle_expert_grad_none_vs_zero_is_a_known_difference`.
`moe.py` was not changed.

## 1 thread (`torch.set_num_threads(1)`)
| N | E | d | d_ff | fwd loop med/p90 ms | fwd vec med/p90 ms | fwd speedup | fwd+bwd loop med/p90 ms | fwd+bwd vec med/p90 ms | fwd+bwd speedup |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 4 | 64 | 256 | 2.257 / 2.46 | 2.458 / 2.649 | 0.918x | 5.411 / 5.832 | 7.026 / 7.943 | 0.77x |
| 1024 | 4 | 256 | 1024 | 13.733 / 15.489 | 14.858 / 16.127 | 0.924x | 36.532 / 39.419 | 49.028 / 51.94 | 0.745x |
| 1024 | 16 | 64 | 256 | 3.395 / 3.667 | 2.826 / 3.07 | 1.201x | 7.158 / 7.965 | 6.054 / 6.393 | 1.182x |
| 1024 | 16 | 256 | 1024 | 20.136 / 22.09 | 39.977 / 44.084 | 0.504x | 46.254 / 49.768 | 61.623 / 72.399 | 0.751x |
| 1024 | 64 | 64 | 256 | 8.326 / 8.932 | 4.83 / 5.122 | 1.724x | 18.81 / 20.392 | 11.828 / 14.01 | 1.59x |
| 1024 | 64 | 256 | 1024 | 43.327 / 46.39 | 105.291 / 107.617 | 0.411x | 92.907 / 96.791 | 200.3 / 246.383 | 0.464x |
| 4096 | 4 | 64 | 256 | 7.445 / 8.283 | 8.652 / 9.174 | 0.86x | 17.397 / 19.219 | 19.222 / 21.163 | 0.905x |
| 4096 | 4 | 256 | 1024 | 59.154 / 65.04 | 72.795 / 76.376 | 0.813x | 151.748 / 162.085 | 170.506 / 182.498 | 0.89x |
| 4096 | 16 | 64 | 256 | 10.259 / 11.221 | 9.344 / 10.335 | 1.098x | 21.904 / 23.569 | 19.692 / 24.082 | 1.112x |
| 4096 | 16 | 256 | 1024 | 64.366 / 71.005 | 81.348 / 90.817 | 0.791x | 162.305 / 167.771 | 193.784 / 201.553 | 0.838x |
| 4096 | 64 | 64 | 256 | 25.641 / 203.155 | 15.563 / 62.755 | 1.648x | 46.268 / 51.271 | 31.395 / 36.111 | 1.474x |
| 4096 | 64 | 256 | 1024 | 101.109 / 106.17 | 113.347 / 117.144 | 0.892x | 222.204 / 229.588 | 366.926 / 376.49 | 0.606x |
| 16384 | 4 | 64 | 256 | 30.535 / 32.607 | 52.58 / 55.087 | 0.581x | 84.096 / 86.77 | 100.929 / 108.795 | 0.833x |
| 16384 | 4 | 256 | 1024 | 279.374 / 315.772 | 357.796 / 431.47 | 0.781x | 703.111 / 782.556 | 866.627 / 952.385 | 0.811x |
| 16384 | 16 | 64 | 256 | 43.593 / 47.936 | 59.615 / 64.082 | 0.731x | 103.398 / 106.908 | 108.473 / 112.135 | 0.953x |
| 16384 | 16 | 256 | 1024 | 278.974 / 301.177 | 373.414 / 392.222 | 0.747x | 652.09 / 677.677 | 868.028 / 883.811 | 0.751x |
| 16384 | 64 | 64 | 256 | 81.51 / 84.295 | 70.937 / 74.554 | 1.149x | 153.742 / 155.762 | 129.291 / 133.749 | 1.189x |
| 16384 | 64 | 256 | 1024 | 349.711 / 368.669 | 487.801 / 499.94 | 0.717x | 819.688 / 869.22 | 1147.325 / 1206.605 | 0.714x |

## 8 threads
| N | E | d | d_ff | fwd loop med/p90 ms | fwd vec med/p90 ms | fwd speedup | fwd+bwd loop med/p90 ms | fwd+bwd vec med/p90 ms | fwd+bwd speedup |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 4 | 64 | 256 | 1.581 / 1.799 | 1.389 / 1.568 | 1.138x | 3.563 / 4.259 | 3.7 / 4.103 | 0.963x |
| 1024 | 4 | 256 | 1024 | 3.619 / 4.187 | 3.471 / 4.842 | 1.043x | 9.395 / 11.482 | 13.974 / 16.361 | 0.672x |
| 1024 | 16 | 64 | 256 | 3.669 / 5.713 | 2.037 / 2.34 | 1.801x | 6.983 / 7.224 | 3.846 / 4.224 | 1.816x |
| 1024 | 16 | 256 | 1024 | 8.068 / 9.761 | 4.27 / 6.359 | 1.889x | 17.224 / 20.953 | 19.001 / 24.443 | 0.906x |
| 1024 | 64 | 64 | 256 | 9.932 / 12.05 | 3.348 / 3.725 | 2.967x | 19.33 / 22.766 | 8.566 / 9.796 | 2.257x |
| 1024 | 64 | 256 | 1024 | 19.64 / 23.996 | 28.539 / 32.138 | 0.688x | 38.154 / 43.363 | 90.586 / 112.691 | 0.421x |
| 4096 | 4 | 64 | 256 | 3.234 / 3.461 | 4.084 / 4.659 | 0.792x | 6.752 / 7.425 | 6.926 / 7.61 | 0.975x |
| 4096 | 4 | 256 | 1024 | 9.935 / 11.236 | 11.712 / 13.483 | 0.848x | 30.528 / 35.823 | 33.731 / 48.191 | 0.905x |
| 4096 | 16 | 64 | 256 | 5.823 / 6.079 | 3.981 / 4.297 | 1.463x | 12.692 / 14.956 | 7.734 / 10.212 | 1.641x |
| 4096 | 16 | 256 | 1024 | 15.312 / 17.9 | 12.634 / 14.798 | 1.212x | 40.929 / 45.603 | 44.423 / 52.611 | 0.921x |
| 4096 | 64 | 64 | 256 | 13.157 / 15.222 | 5.421 / 5.911 | 2.427x | 26.075 / 31.719 | 13.111 / 14.389 | 1.989x |
| 4096 | 64 | 256 | 1024 | 38.707 / 42.421 | 34.95 / 38.479 | 1.107x | 76.139 / 87.138 | 119.87 / 159.436 | 0.635x |
| 16384 | 4 | 64 | 256 | 7.387 / 8.271 | 10.08 / 11.294 | 0.733x | 20.012 / 23.37 | 22.293 / 24.451 | 0.898x |
| 16384 | 4 | 256 | 1024 | 39.032 / 47.172 | 72.814 / 93.231 | 0.536x | 176.572 / 187.84 | 211.588 / 297.633 | 0.835x |
| 16384 | 16 | 64 | 256 | 22.011 / 27.04 | 18.913 / 24.343 | 1.164x | 47.416 / 53.135 | 35.715 / 48.525 | 1.328x |
| 16384 | 16 | 256 | 1024 | 92.814 / 123.476 | 103.174 / 133.362 | 0.9x | 218.428 / 827.616 | 223.994 / 367.701 | 0.975x |
| 16384 | 64 | 64 | 256 | 32.359 / 37.519 | 15.721 / 17.249 | 2.058x | 72.553 / 77.294 | 35.961 / 39.308 | 2.018x |
| 16384 | 64 | 256 | 1024 | 85.206 / 122.721 | 103.211 / 124.805 | 0.826x | 189.686 / 212.443 | 278.658 / 308.387 | 0.681x |

## Reading
- The vectorized path wins clearly only with many experts and small width: E=64, d=64 is
  1.15-1.72x faster at 1 thread and 1.99-2.97x at 8 threads. E=16, d=64 ranges 0.73-1.82x.
  With d=256, forward+backward is slower in all 18 runs (0.42-0.98x); forward-only at d=256
  is slower in every 1-thread run and mixed at 8 threads (0.54-1.89x). With E=4 it is
  slower in every 1-thread run.
- Likely cause (not profiled, an inference): it pads every expert to the largest kept
  count, then runs dense `baddbmm` over [E, L, d]. It also re-stacks all expert weights
  every call (`torch.stack`), which costs E·(2·d·d_ff) copies forward and backward. That
  matters more as d grows, while the Python-loop overhead it removes matters more as E grows.
- Default stays `dispatch="loop"`; this data does not justify switching the default.
