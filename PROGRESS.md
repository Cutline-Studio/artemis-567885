# Progress log (dates in America/New_York)

## Iteration 1 — 2026-10-23
- Baseline: the four files committed exactly as provided. `python sandbox.py` → Total Loss 1.5083, dropped 6, ∇x 0.1581, ∇W_g 0.0546 (torch 2.14.1+cpu).
- Stage 1: verified review findings (`docs/REVIEW.md`, `tools/investigate_baseline.py`); fixed aux loss, capacity priority, dropped-token fallback and drop stats behind `legacy` flag (default fixed); 18 pytest tests; sandbox runs both modes; legacy reproduces baseline exactly.
