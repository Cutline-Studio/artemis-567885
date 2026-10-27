# Artemis 567885 — expansion roadmap

Each stage is small, ships on its own branch/PR, keeps `legacy=True` reproducing the
original numbers, and must leave `python -m pytest` and `python sandbox.py` green.

| Stage | Scope | Done when |
|---|---|---|
| 1 | pytest suite (shapes, gradients to input + gate, capacity drop counts, determinism); fix review bugs #1, #2, #3a/b, #7 behind `legacy` flag (default = fixed) | tests green, `--mode legacy` reproduces baseline exactly |
| 2 | **Expand engine** (`expand.py`) + **self-fixing supervisor** (`supervisor.py`): sandbox target becomes regression X → sin(X) on random continuous vectors; each cycle trains a few hundred Adam steps on CPU; rules: dropped>0 → capacity_factor += 0.05; aux>1.2 → alpha ×1.5; stalled loss (<1% rel. improvement) → add one expert (copy+noise, gate grows one row, trained weights kept). State in `config.json`, checkpoint in `runs/`, every cycle logged to `runs/expand_log.jsonl`. Hard caps (CF ≤ 2.0, experts ≤ 32, alpha ≤ 1.0, cycles per invocation). Supervisor backs up to `.backup_checkpoint/`, ast-parses sources, runs the cycle in a subprocess with timeout, rolls back on failure/NaN/shape mismatch and tries a different mutation path | ~10 real cycles run through the supervisor; tests for the rules and for rollback |
| 3 | `causal=True` attention mask and positional encoding option (sinusoidal + learned) | tests: causal (future change doesn't affect past), permutation no longer equivariant |
| 4 | Vectorized (sort-based) dispatch path + `bench.py` comparing loop vs vectorized across E ∈ {4,16,64} | identical outputs to loop path (allclose), benchmark table committed |
| 5 | `ArtemisModel`: N stacked blocks, byte embedding (256) in, byte logits out, aux summed across layers | shape + gradient tests |
| 6 | Tokenization-free byte-level training loop on CPU on real public-domain text (e.g. a Project Gutenberg book), with eval bits-per-byte and sampling | real bpb curve logged |
| 7 | Sweep alpha ∈ {0, 0.01, 0.1, 0.3} × capacity_factor ∈ {0.8, 1.0, 1.25, 1.5} on the stage-6 task; report loss, drop rate, expert load entropy | sweep table in docs/ |
| 8 | Optional: Switch-style top-1 gate weighting (fix dead top-1 gate grad, review #6), expert-choice routing, z-loss | tests + sweep comparison |
