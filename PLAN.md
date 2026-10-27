# Artemis 567885 — expansion roadmap

Each stage is small, ships on its own branch/PR, keeps `legacy=True` reproducing the
original numbers, and must leave `python -m pytest` and `python sandbox.py` green.

| Stage | Scope | Done when |
|---|---|---|
| 1 ✅ | pytest suite (shapes, gradients to input + gate, capacity drop counts, determinism); fix review bugs #1, #2, #3a/b, #7 behind `legacy` flag (default = fixed) | tests green, `--mode legacy` reproduces baseline exactly |
| 2 ✅ | **Expand engine** (`expand.py`) + **self-fixing supervisor** (`supervisor.py`): sandbox target becomes regression X → sin(X) on random continuous vectors; each cycle trains a few hundred Adam steps on CPU; rules: dropped>0 → capacity_factor += 0.05; aux>1.2 → alpha ×1.5; stalled loss (<1% rel. improvement) → add one expert (copy+noise, gate grows one row, trained weights kept). State in `config.json`, checkpoint in `runs/`, every cycle logged to `runs/expand_log.jsonl`. Hard caps (CF ≤ 2.0, experts ≤ 32, alpha ≤ 1.0, cycles per invocation). Supervisor backs up to `.backup_checkpoint/`, ast-parses sources, runs the cycle in a subprocess with timeout, rolls back on failure/NaN/shape mismatch and tries a different mutation path | ~10 real cycles run through the supervisor; tests for the rules and for rollback. Also in stage 2: `model.py` DynamicMultiBlockTransformer (aux + dropped summed over layers, vertical growth with zero-init identity block, num_layers ≤ 8), matrix auto-correct in the supervisor (shape-error regexes, config realigned to the checkpoint, one validation re-run), and the separate `physics_simulation.py` PINN sandbox (SIREN experts, coordinate router, advection PDE + IC/BC, analytic comparison) |
| 3 | `causal=True` attention mask and positional encoding option (sinusoidal + learned) | tests: causal (future change doesn't affect past), permutation no longer equivariant |
| 4 | Vectorized (sort-based) dispatch path + `bench.py` comparing loop vs vectorized across E ∈ {4,16,64} | identical outputs to loop path (allclose), benchmark table committed |
| 5 | `ArtemisModel`: N stacked blocks, byte embedding (256) in, byte logits out, aux summed across layers | shape + gradient tests |
| 6 | Tokenization-free byte-level training loop on CPU on real public-domain text (e.g. a Project Gutenberg book), with eval bits-per-byte and sampling | real bpb curve logged |
| 7 | Sweep alpha ∈ {0, 0.01, 0.1, 0.3} × capacity_factor ∈ {0.8, 1.0, 1.25, 1.5} on the stage-6 task; report loss, drop rate, expert load entropy | sweep table in docs/ |
| 8b | Physics sandbox follow-ups: lr 1e-4 or cosine schedule, stronger alpha / noisy routing to fix the observed expert collapse (2 of 4 experts unused), causal-in-time training, error vs t study | rel. L2 at t=1 below the iteration-1 value 0.250 |
| 8 | Optional: Switch-style top-1 gate weighting (fix dead top-1 gate grad, review #6), expert-choice routing, z-loss | tests + sweep comparison |

## Later stages from the user's concept specs (iteration 1 additions, not built yet)

Source specs, copied verbatim: `docs/INTERNET_COORDINATE_FIELD.md` and
`docs/CONTINUOUS_LANGUAGE_FIELD.md`. Note: the second file is **truncated** — it ends at the
heading "4. Conceptual Python Proof-of-Concept" right before its code section; no code was
included in what was received.

| Stage | Scope | Done when |
|---|---|---|
| 9 (a) | `fourier.py`: multiscale Fourier feature encoding γ(t) = [sin(2^k π t), cos(2^k π t)]_{k<L} (+ optional random Gaussian features) as an `nn.Module`, any input dim | tests: output dim = 2·L·in_dim (+in if include_input), values bounded in [-1,1], exact known values at t=0/0.5, differentiable, frequency scaling checked |
| 9 (b) | `webstream.py`: stream real text bytes with HTTP Range requests (a Common Crawl WET file slice or a Project Gutenberg plain-text book), no tokenizer; local cache keyed by URL+range; polite rate limit (≥1 s between requests, identifying User-Agent, respects 429/Retry-After), resumable offsets | tests with a local HTTP server: correct byte ranges, cache hit avoids network, 206 vs 200 handling; one real smoke fetch logged. **Starting point:** the user's streaming wrapper, saved as `docs/reference/clf_stream_original.py`. Fixes to make when porting: (1) its target URL `https://www.w3.org/TR/PNG/iso_8859-1.txt` returns 404 (checked by the parent agent), and it then *silently* falls back to synthetic sine bytes — the port must fail loudly or clearly label fallback data as synthetic in every output/log, never pass it off as internet data; (2) send a real `Range: bytes=start-end` header and require HTTP 206 (treat 200 as 'server ignored Range' and slice/limit explicitly); (3) use a working source (a Project Gutenberg plain-text .txt or a Common Crawl WET slice); (4) its Fourier projection mixes the whole 32-byte window into one vector with a random B drawn per window length — replace with per-element features (each byte value and its own position encoded separately) so it plugs into the 9(c) byte-level model; (5) port from NumPy to torch. |
| 9 (c) | Byte-level Continuous-Language-Field prototype: raw bytes → continuous vectors (learned 256-value byte embedding, or byte/255 scalar) **plus** γ(t) for the continuous document position t, through the MoE blocks with a **causal mask** (stage 3), predicting the next byte (256-way softmax). A field f(γ(t)) on position alone can only memorise a single document (it maps a coordinate to a value), so the model must be conditioned on the preceding bytes to generalise. | bits-per-byte on held-out text from 9(b) reported vs a unigram baseline (≈ entropy of the byte distribution) and vs a no-position ablation |
| 9 (d) | Optional hypernetwork: a context vector (e.g. pooled preceding bytes) generates low-rank modulations (FiLM / LoRA-style deltas) of expert weights | tests: zero context ⇒ base experts exactly; bpb compared with 9(c) |
| 9 (e) | Serving over HTTP — **deferred**. The user does not want Vercel deployment, and nothing gets hosted without asking first. | — |

## Review notes from Codex (2026-10-23)
- Stage 3 acceptance checks: (1) causality — perturbing inputs at positions > t must leave outputs at positions <= t unchanged; (2) position — with positional encoding on, reordering a sequence must change outputs; keep the unmasked/no-position mode as an explicit option for non-causal tasks.
- New stage (control experiment): compare the auto-growing expand run against a fixed-architecture baseline trained with comparable compute (same total steps / FLOPs) on the same X -> sin(X) data; report both held-out losses before claiming growth helps.
- Keep three milestones separate in docs: (a) correctness tests pass, (b) PDE fit, (c) language generalization (held-out bits-per-byte, stage 9c — not yet tested).

## Deferred: SaaS layer (user decision 2026-10-23)
On hold until stage 9c shows real held-out bits-per-byte results. When unblocked: local-only auth + usage limits first, then Stripe test mode. No hosting, no Vercel, no live payments without the user's approval. Spec: user's SaaS blueprint (tiers, /v1/field/query, /v1/context/infiltrate, /v1/field/stream).

## Physics pilot benchmark (user spec, 2026-10-23) — measured, did NOT converge
- pilot/artemis_physics_benchmark.py as given: 1000 epochs, test MSE 0.3567, rel L2 0.852, y-variance 0.441 (target 1e-4 never reached).
- Variant with inputs normalized to [-1,1], 5000 epochs: test MSE 0.0935, rel L2 0.436, y-variance 0.155.
- Next stage: diagnose — w0=30 is far too high-frequency for sin(x - 0.5t) (period ~2pi); sweep w0 (1, 3, 10, 30), lr, epochs, train set size; add a plain MLP/Fourier-feature baseline; report steps/time to MSE<1e-4 and y-variance. Only then consider Navier-Stokes.

## Frequency-and-capacity study (user spec, 2026-10-23) — measured
Script `pilot/frequency_capacity_study.py`, summary `pilot/results/FREQUENCY_CAPACITY_STUDY.md` (all numbers from `pilot/results/frequency_capacity_runs.csv`).
Supervised regression on analytical labels sin(k(x-0.5t)); not physics-informed; same-domain interpolation only. One fixed split, 5 init seeds per finalist, selection on validation only.
- Selected (val only): SIREN w0_first 3 / w0_hidden 10 (w0 30/30 hypothesis refuted), lr 3e-3, 10000 epochs; tanh lr 3e-3. lr and budget are at the edge of their grids.
- Study A (SIREN 32x2, 1217 params): val-threshold success k=1,2,4: 5/5 each (test rel L2 ~1e-3); k=8: 0/5 (rel L2 ~1.0).
- Width 16-128 / depth 1-4, SIREN and matched tanh: no 3-input config reached the threshold at k=8. Wide/deep SIRENs fit train but not val (poor generalisation); smallest models stall.
- (x,t)-only SIREN: k=8 5/5 (rel L2 ~1e-2). Implicates the 3-input setup; not isolated (input dim, init bound, params also change).
- Ablation (k=1, w0 30/30, lr 1e-3): only normalisation + 5000 epochs succeeded (5/5); each alone 0/5.
- Latency: the formula was 1.3x-15.9x faster than every network (same procedure). No speed advantage.
- Cost: CPU only, no cloud charges; study wall clock 667 s.
- Next step: per-k selection of lr/budget at k=8 (3-input), plus an ablation that separates the y-input effect (3 inputs with y fixed to a constant, matched first-layer init bound) before any Navier-Stokes work.
