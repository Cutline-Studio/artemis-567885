> **Scope note for artemis-567885.** The sin(X) expand-loop model in this repo is a research toy as far as these decks are concerned. It has `d_model` 16 and 2 heads, works on continuous vectors with no tokenizer, has unmasked attention by default, and has no KV cache or serving path. GQA, KV-cache quantization and serving telemetry were **not** bolted onto it.
>
> This repo gets two things:
>
> - the standalone sizing module `tools/artemis_sizing.py`, with tests in `tests/test_artemis_sizing.py`;
> - this mapping.
>
> The code changes for the real LLM serving path (`gateway/artemis/` in `lpittman-blip/Cutline-Industries`) are in that repo's PR, and the file paths below refer to that repo. The expand-loop growth code and caps were not touched.

# Artemis AI architecture: two reference decks mapped onto the real code

Status date: 2026-10-27. Two reference decks were applied:

- **Modern AI Stack**, 11 layers (Organizational Zen, 12 pages). It is cited below as "Zen p.N".
- **Cisco Live BRKCOM-2115**, "Building a full stack AI-ready data center" (Roger Dickinson, 2025). It is cited as "Cisco slide N", using the slide numbers printed in the deck.

Ground rules for this document:

- Every number from either deck is a **target or an example**, never an Artemis result.
- The sizing figures are **estimates** produced by the deck's formulas (`gateway/artemis/sizing.py`). Nothing here was measured on a GPU.
- No code in this change was run on the Grok Bot box. The tests run in GitHub Actions CI (`.github/workflows/ci.yml`, job "Artemis gateway tests" runs `pytest tests/`). To print the table, run `python -m artemis.sizing` from `gateway/`.

## Where the code is

| Repo | What it is | Relevant to the decks? |
|---|---|---|
| `lpittman-blip/Cutline-Industries`, `gateway/artemis/` | The real Artemis LLM stack. It has a decoder-only transformer with GQA + RoPE + top-2 MoE experts (`model.py`), checkpoint serving (`infer.py`), an orchestrator over the planet brains (`orchestrator.py`), the API server (`server.py`), the chat site backend (`chat_app.py`, which subclasses the server handler) and the training and data pipeline | **Yes.** All the code changes below are here |
| `Cutline-industries/artemis-567885` | The sin(X) expand-loop research model. It has `d_model` 16 and 2 heads, works on continuous vectors with no tokenizer, has unmasked attention by default, and has no KV cache or serving path | **No, it's a toy for these purposes.** GQA, KV-cache quantization and serving telemetry were deliberately *not* added to it. It gets the sizing module and this mapping only |

---

## Part 1: Modern AI Stack, layer by layer

Legend: **exists** = already in the code before this change. **added** = this change. **missing** = not built (deferred, with the reason given).

| # | Layer (Zen page) | Exists (file paths under `gateway/`) | Added in this change | Missing / deferred |
|---|---|---|---|---|
| 1 | Model Stack (p.2) | **Foundation:** Artemis's own `ArtemisLM` (`artemis/model.py`), with sizes in `configs/models.yaml`. **Fine-tuned:** SFT with provenance gates (`artemis/sft.py`, `artemis/teachers.py`), self-play (`artemis/selfplay.py`) and LoRA brains (`brains/*.yaml` `model:`). **Registry with versioning + rollback:** `artemis/promote.py` `Registry` (`apply` / `current` / `rollback`, plus an evaluation gate from `configs/promotion.yaml`) | Serving limits come from the sizing module. `LocalModel.info()["sizing"]` is returned by `/v1/status` (`artemis/infer.py`) | No embedding model and no vision/speech models (`chat_app` reports `voice_input: false`). The registry tracks checkpoints only and has no semver tags |
| 2 | Semantic layer (p.3) | Keyword routing vocab per brain (`brains/*.yaml` `keywords`, `orchestrator.keyword_route`) | none | **Missing:** canonical entity and synonym normalization ("customer" = "client"). Deferred because changing the router vocabulary changes routing behaviour, which needs its own eval run |
| 3 | Ontology (p.3) | none | none | **Missing:** a knowledge graph. Nothing in the product needs one yet |
| 4 | RAG (p.4) | Retrieval into context with a labelled block: `artemis/memory.py` (BM25 over account notes, top-k=3, no model). Web retrieval tools `web_search`, `web_fetch` and `research_search` (`artemis/tools.py`) | none | **Missing:** embeddings, a vector DB (pgvector, Qdrant), a re-ranker, and enforced citations in answers. The Cisco slide 13 RAG sizing (8-12 GB for the embedder, 8-12 GB for the reranker, 33 GB for an 8B LLM) is the reference when this is built |
| 5 | Orchestration (p.5) | **Decomposition and tool routing:** `Artemis.plan` (keyword or model router). **Parallel execution:** `ThreadPoolExecutor` drafts per specialist. **State and limits:** `MAX_TOOL_STEPS`, `ResponseIncomplete`. **Synthesis:** `_merge_messages`. All of this is in `artemis/orchestrator.py` | Each request now carries a trace_id across plan → tools → done | **Missing:** automatic retry of failed sub-steps. Failures are surfaced to the user instead, which is by design |
| 6 | Memory (p.6) | **Working:** history capped at `MAX_HISTORY`=20 and truncated to `max_seq_len`. **Episodic and semantic:** `artemis/memory.py` per-account notes (BM25). **Procedural:** brain instructions in `brains/*.yaml`. Chat history is stored in `artemis/conversations.py` | none | **Missing:** vector (similarity) recall. Memory is lexical only |
| 7 | Agents (p.7) | **ReAct-style tool loop:** `Artemis._respond`, reason → `<tool_call>` → result → continue, ≤4 steps. **Tools:** `artemis/tools.py`. **Multi-agent:** planet specialists plus a Venus audit. **Human-in-the-loop:** plan gating per tool (`Business.authorize_tool`) | none | **Missing:** explicit approval checkpoints for consequential tool calls. No current tool is consequential (search, fetch and sandboxed Python only) |
| 8 | Governance (p.8) | **RBAC-style plan entitlements:** `artemis/business.py` (`authorize_chat`, `authorize_model`, `authorize_tool`, `authorize_api`). API keys are hashed. Request text is kept out of logs (`server.py log_message`). Data licences are enforced (`data.py`, `code_corpus.py`, `sft.py`) | **Immutable audit log with trace_id:** `artemis/telemetry.py` `AuditLog`. It is an append-only, SHA-256 hash-chained JSONL file (`ARTEMIS_AUDIT_LOG`), and `python -m artemis.telemetry <log>` verifies the chain. **PII scan + redaction:** `artemis/guardrails.py` (email, phone, US SSN, Luhn-valid card numbers), flagged on every request and redacted in feedback comments. **Input guardrails:** prompt-injection and tool-markup flags. **Versioned prompt registry:** `artemis/prompts.py` gives each brain system prompt a sha256 version that is recorded per request, and has a `lock`/`check` review gate | **Missing:** ABAC, data-residency enforcement, output guardrails beyond the Venus audit, and bias/fairness evals mapped to NIST AI RMF and EU AI Act tiers. The prompt lock file isn't committed yet; run `python -m artemis.prompts lock` and commit it to turn on the `check` gate. The audit log is tamper-*evident*, not tamper-*proof*, until it is shipped to write-once storage |
| 9 | Evaluation (p.9) | Model evals (`artemis/evaluate.py`: validation loss, per-expert routing, ablation). Promotion gate (`artemis/promote.py`) | **The deck's targets as pass/fail thresholds:** `artemis/eval_targets.py`. These are faithfulness >90%, relevance@k >85%, harmful <0.1%, P95 <2 s and <$0.01/query. It scores judged JSONL rows and falls back to the audit log for latency and cost. A target with no data is "no data", never a pass | **Missing:** the judges themselves (LLM-as-judge, human panels) and A/B testing. **Artemis has not been scored against any of these targets** |
| 10 | UX (p.10) | Chat UI with streaming plan, tool and token events. `external: true` labels a third-party answer. Honest readiness (`/v1/status`, "training" state) | **Thumbs feedback:** `POST /v1/feedback {trace_id, rating: up/down, comment}`. It is stored as an audit `feedback` record joined to the request's trace_id, with the comment PII-redacted. The `trace_id` is on every done event and response | **Missing:** a UI button that calls `/v1/feedback` (the site still has to wire it), confidence signals and an "I don't know" fallback, and feeding feedback into fine-tune data. `chat_app` reports `training_collection: false` and that stays so until consent rules exist |
| 11 | Infrastructure / MLOps (p.11) | **CI:** `.github/workflows/ci.yml`. **Experiment records:** run JSON plus checkpoint format versions (`artemis/checkpoint.py`). **Cost:** `Business.record_api_call` per API call. **Secrets:** env-only, with `.env.example` | **Per-request tracing and cost attribution:** the audit record has approximate tokens, TTFT, total latency and cost. Cost is real for API calls and *estimated* for chat via `ARTEMIS_EST_USD_PER_1K_TOKENS`. `ARTEMIS_TELEMETRY_STDERR=1` sends one JSON line per request to the container log collector | **Missing:** MLflow-style experiment tracking, dashboards, SIEM and distributed tracing across services |

### Code changed (gateway)

| File | Change |
|---|---|
| `gateway/artemis/sizing.py` | **new**: the sizing module and CLI (Part 2) |
| `gateway/artemis/telemetry.py` | **new**: `RequestTrace` (TTFT, total, tokens, cost), the hash-chained `AuditLog`, and `summarize` (P50/P95) |
| `gateway/artemis/guardrails.py` | **new**: injection and PII flags, redaction, and `SlidingWindowLimiter` (`ARTEMIS_RATE_PER_MIN`, off by default) |
| `gateway/artemis/prompts.py` | **new**: the prompt registry (`show` / `lock` / `check`) |
| `gateway/artemis/eval_targets.py` | **new**: the deck targets as thresholds |
| `gateway/artemis/data_checks.py` | **new**: eval contamination, poisoning flags and the licence denylist |
| `gateway/artemis/server.py` | trace_id, an audit record per `/v1/chat`, `/v1/chat/stream` and `/v1/chat/completions` request, the `/v1/feedback` route, and the per-client rate limit. Behaviour is otherwise unchanged and flags never block |
| `gateway/artemis/data.py` | opt-in `prepare(..., screen=True, eval_texts=[...])`. The default behaviour is byte-for-byte unchanged |
| `gateway/artemis/infer.py` | `info()["sizing"]`: weights, GQA factor, KV bytes per token and at max context, and the concurrency limit (1, set by the existing lock) |
| `gateway/tests/test_ai_architecture.py` | **new**: tests for all of the above, including the deck's worked examples |

---

## Part 2: Cisco BRKCOM-2115 sizing applied to Artemis (estimates)

### Formulas (`gateway/artemis/sizing.py`)

- **Raw size** = params × bytes/param. FP32 is 4, FP16/BF16 2, INT8/FP8 1 and INT4 0.5 (Cisco slides 11 and 16).
- **KV bytes per token** = 2 × precision × n_layers × d_model / GQA factor, where GQA = n_heads / n_kv_heads (Cisco slide 12).
- **Inference memory** = (raw size + KV/token × context × concurrent sessions) × 1.2 overhead (Cisco slides 12, 14 and 16).
- **Training memory** = FP16 weights + activations (2.5 × weights) + Adam FP32 (params × 4 × 2) = 15 B/param. This is inside the deck's "raw × 6-10" range (Cisco slide 18).
  - It is a **floor**: it has no FP32 master weights or gradients, and activations really scale with batch × sequence length.
  - MoE models are sized on **total** parameters, because all experts must be resident.
- **Parallelism** (Cisco slide 16): no split when one GPU holds it; tensor within one 8-GPU NVLink node (Cisco slide 17); pipeline across nodes; hybrid at ≥400B. MoE models also get expert parallelism.

### Deck sanity checks (asserted in `tests/test_ai_architecture.py`)

- **Llama 3.1 8B:**
  - 16.06 GB in BF16.
  - 131,072 B (131 KB) of KV per token at GQA 4.
  - 16.06 + 17.18 = **33.2 GB** at 131,072 context.
  - Training needs 16.06 + 40.15 + 64.24 = **120.45 GB**, which is 1 × H200, 1 × B200 or 2 × H100. This matches Cisco slides 11, 12 and 18.
- **Llama 3.1 405B:**
  - **810 GB** in BF16, or 972 GB with the 20% overhead.
  - FP32 is 1,620 / 1,944 GB, INT8 405 / 486 GB and INT4 202.5 / 243 GB.
  - Recommended parallelism is **hybrid**. This matches Cisco slide 16.

### Sizing table (estimates)

The table is BF16, 1 concurrent session at the context shown, with GPUs on H100 80 GB / H200 141 GB / B200 192 GB. It was computed by hand from the formulas above. `python -m artemis.sizing` regenerates it, and the key cells are asserted in the tests. Parallelism is based on the H100 count.

| model | params (B) | weights BF16 / FP8 / INT4 (GB) | KV KB/token with GQA (no GQA) | context | inference GB (+20%) | inference GPUs H100/H200/B200 | training GB (deck floor) | training GPUs H100/H200/B200 | parallelism: inference ; training |
|---|---|---|---|---|---|---|---|---|---|
| artemis 100m (`models.yaml`) | 0.153 (0.111 active) | 0.306 / 0.153 / 0.077 | 12.3 (36.9), GQA 3 | 2,048 | 0.397 | 1/1/1 | 2.30 | 1/1/1 | none ; none |
| artemis 1b | 1.31 (1.06 active) | 2.62 / 1.31 / 0.655 | 81.9 (164), GQA 2 | 4,096 | 3.55 | 1/1/1 | 19.7 | 1/1/1 | none ; none |
| artemis 3b | 4.17 (3.38 active) | 8.34 / 4.17 / 2.09 | 115 (344), GQA 3 | 4,096 | 10.6 | 1/1/1 | 62.6 | 1/1/1 | none ; none |
| artemis 8b | 10.2 (8.4 active) | 20.4 / 10.2 / 5.10 | 147 (590), GQA 4 | 8,192 | 25.9 | 1/1/1 | 153 | 2/2/1 | none ; tensor + expert |
| Qwen2.5-7B-Instruct (base for Artemis-7B) | 7.62 | 15.2 / 7.62 / 3.81 | 57.3 (401), GQA 7 | 32,768 | 20.5 | 1/1/1 | 114 | 2/1/1 | none ; tensor |
| Llama 3.1 8B (teacher) | 8.03 | 16.1 / 8.03 / 4.02 | 131 (524), GQA 4 | 131,072 | 39.9 | 1/1/1 | 120 | 2/1/1 | none ; tensor |
| Llama 3.1 405B (deck example) | 405 | 810 / 405 / 202.5 | 516 (8,258), GQA 16 | 131,072 | 1,053 | 14/8/6 | 6,075 | 76/44/32 | hybrid ; hybrid |
| 100B MoE plan (shape assumed: 48 layers × 5120, 40 heads / 8 KV) | 100 (8 active) | 200 / 100 / 50 | 197 (983), GQA 5 | 131,072 | 271 | 4/2/2 | 1,500 | 19/11/8 | tensor + expert ; pipeline + expert |
| 1T MoE plan (Kimi-K2-like 61 × 7168, GQA 8 assumed) | 1,000 (32 active) | 2,000 / 1,000 / 500 | 219 (1,749), GQA 8 | 131,072 | 2,434 | 31/18/13 | 15,000 | 188/107/79 | hybrid + expert ; hybrid + expert |
| 6T MoE option (shape assumed: 96 × 12288, 96 heads / 8 KV) | 6,000 (180 active) | 12,000 / 6,000 / 3,000 | 393 (4,719), GQA 12 | 131,072 | 14,462 | 181/103/76 | 90,000 | 1,125/639/469 | hybrid + expert ; hybrid + expert |

How to read this table:

- **The GPU counts are memory floors, not throughput plans.** The Oct 27 planning page's 64 H100 for a 100B run in about 4.6 days is set by FLOPs and wall-clock time. The table's 19 H100 is only the minimum needed to *hold* training state.
- **The 100B, 1T and 6T layer and head shapes are assumptions.** They exist only so the KV cache can be sized. Real Kimi K2 uses MLA rather than GQA, so its actual KV cache is smaller.
- **Concurrency (Cisco slide 14), Llama 3.1 8B at 131,072 context:**
  - With a BF16 KV cache, one GPU holds 2 sessions on an H100, 5 on an H200 and 8 on a B200 (`max_concurrent_sessions`).
  - Quantizing the KV cache to FP8 raises H100 to 5 sessions.
  - At 8K context the per-session KV is 1.07 GB, so dozens of sessions fit.

### Inference path: what was applied

- **GQA already exists** in the real model. `ModelConfig.n_kv_heads` and `Attention` in `gateway/artemis/model.py` use `repeat_interleave` over the KV heads, and every size in `configs/models.yaml` sets `n_kv_heads` (GQA 2-4). Nothing was changed there.
- **KV-cache quantization is not applicable yet.** `ArtemisLM.stream` has **no KV cache**: each new token re-runs the whole context (up to `max_seq_len`). Quantizing a cache that doesn't exist would be an invented feature. It is deferred until a cached decode path exists, or until serving moves to vLLM (`ARTEMIS_INFERENCE_URL`), where FP8 KV is a launch flag. The sizing module's `kv_precision="fp8"` gives the memory effect when that happens.
- **Context and concurrency limits**, which `LocalModel.info()["sizing"]` now reports:
  - Context is capped at `cfg.max_seq_len` (prompts are already truncated to it in `LocalModel.stream`).
  - Concurrency is 1 generation per loaded model (the existing `threading.Lock`).
  - Server-side caps are unchanged: `MAX_MESSAGE_CHARS`=8000, `MAX_HISTORY`=20 and a 64 KB request body.
  - The new optional `ARTEMIS_RATE_PER_MIN` adds a per-client request limit.
- **artemis-567885 (sin(X) MoE expand loop) is a toy for this purpose.** It has `d_model` 16 and works on continuous vectors with no tokens, so it has no KV cache to size. GQA was not bolted onto it.

### Observability (Cisco slide 15; Zen p.9, p.11)

- Every chat, stream and completions request writes one audit record containing:
  - trace_id, endpoint, account (guest IPs reduced to a hash), model, status and tier;
  - prompt and completion tokens (≈4 chars/token, the same estimate the API already bills with);
  - `ttft_ms` (first token event) and `total_ms`;
  - `cost_usd` with `cost_is_estimate`;
  - guardrail flags and prompt versions.
- `python -m artemis.telemetry audit.jsonl` verifies the hash chain and prints P50/P95 latency, P50/P95 TTFT, mean cost and total tokens.
- Not covered: the chat site's own `/api/...` routes in `chat_app.py` have not been traced yet. They are handled by the subclass before the base handler.

### Security checklist (Cisco slide 34; Zen p.8)

**Training data** (data source and training risks):

- [x] Licence recorded for every source (`data.py`). Permissive-only code (`code_corpus.py`). Provenance-gated SFT rows (`sft.py`). *(already existed)*
- [x] Exact-duplicate removal (`data.py`, `prepare_lake.py`). *(already existed)*
- [x] Licence denylist (GPL/AGPL/LGPL, CC-BY-NC*, CC-BY-SA, SSPL, unknown): `data_checks.license_ok`, enforced by `prepare(screen=True)`. *(added)*
- [x] Eval-set contamination (13-word shingles) drops overlapping training docs: `data_checks.ContaminationIndex`. *(added)*
- [x] Poisoning and injection screening (injection phrases, tool-call markup, single-line-dominated docs): `data_checks.poisoning_flags`. *(added; heuristic)*
- [ ] Near-duplicate (MinHash) dedup across the full lake. `prepare_lake.py` already notes that this is an offline pass.
- [ ] Open-model contamination: before fine-tuning on third-party weights (Qwen, Llama, Kimi), check the licence (Llama needs a "Built with Llama" display) and record the base model hash in the promotion registry.
- [ ] PII scrubbing of web text before pretraining (`guardrails.redact` can be applied; not wired into the lake pipeline).

**Inference** (DoS, prompt injection, data mining):

- [x] Body ≤64 KB, message ≤8,000 chars, history ≤20. Plan daily limits. API-call rate window (`business.py`). *(already existed)*
- [x] Per-client requests-per-minute limiter, `ARTEMIS_RATE_PER_MIN`. *(added; off by default)*
- [x] Prompt-injection and tool-markup flags on every request, recorded in the audit log. *(added; flag-only)*
- [x] PII detection on input and redaction of stored feedback. Request and answer text is never logged. *(added / existing)*
- [x] Third-party models are passthrough only and labelled `external` (`server.py`). *(already existed)*
- [ ] Promote flags to blocking once the false-positive rate on real traffic has been reviewed.
- [ ] Output guardrails beyond the Venus audit, and a tool-call approval gate if a consequential tool is ever added.

---

## Part 3: Cisco deck sections, applied vs deferred

| Cisco deck section (slides) | Applied to Artemis | Deferred / not applicable, and when it would matter |
|---|---|---|
| Operationalizing AI, the full-stack picture (5-7, 42-43) | Layer mapping (Part 1) | n/a |
| Prompt lifecycle: prefill, KV cache, TTFT (9) | TTFT logged per request | A KV-cached decode path in `ArtemisLM.stream` (would cut per-token cost from O(context) recompute) |
| Model / token / GPU sizing (11-12) | `sizing.py` and tests | n/a |
| RAG sizing (13) | Cited as the reference for the missing RAG layer | When an embedder and reranker are added |
| KV cache and concurrency, GPU table (14) | `max_concurrent_sessions`, the KV precision option, and the H100/H200/B200 figures | KV quantization deferred until a KV cache or vLLM serving exists |
| AI observability, Splunk (15) | Tokens per user, latency and cost in the audit log | Splunk or another SIEM export (the stderr JSON lines are collector-ready) |
| Model parallelism (16) | `parallelism()` rules and the 20% overhead | Actual tensor/pipeline code (Megatron, DeepSpeed, TorchTitan). `test_fsdp.py` covers FSDP only |
| Intra-node NVLink / NVSwitch, UCS C885A / C845A (17, 20) | The 8-GPU NVLink domain is used as the tensor-parallel boundary | **Hardware for owned clusters.** It matters only when Cutline buys or colocates HGX nodes instead of renting (Vast, Azure ND H100) |
| Training memory (18) | `training_bytes` (the deck breakdown) | n/a |
| Network requirements, NCCL / CUDA / Kubernetes stack, collectives (22-25) | Noted: all-reduce = reduce-scatter + all-gather is what FSDP uses | Tuning matters for multi-node training (≥2 nodes, i.e. the 100B training floor of 19 H100) |
| RoCEv2, rail-optimized backend, ECMP / flowlet / per-packet load balancing (26-28) | none | **Owned-fabric only.** On rented clouds the provider's InfiniBand/RoCE fabric covers it. It matters at NVIDIA ERA scale (**32-256 GPUs**, slides 29-30) on owned racks |
| ERA / NCP reference architectures (29-30) | The 1T plan's 512 H100 and the 6T plan's 4,096 H100 sit in NCP territory (256-16,000+ GPUs) | Only with a cluster partner. Not before the 100B stage is proven |
| Cisco Nexus 9000 / 8000 / Hyperfabric AI, Silicon One (31-32) | none | Switch vendor choice for owned racks only |
| Security: data, training and inference risks (34) | The checklist above, with code where the paths exist | n/a |
| Hybrid Mesh Firewall, Hypershield, DPU / Smart Switch, Cilium (35) | none | Network-level workload security for owned clusters or Kubernetes. Azure NSGs and Container Apps ingress cover the current deployment |
| AI Defense: discover, detect, protect (36) | The guardrail flags and audit log are the in-app "protect" piece | Red-team testing ("detect") with an eval suite of attack prompts |
| Secure AI Factory, AI POD (40-41) | none | Vendor-integrated stack. It is relevant only for on-prem procurement |
