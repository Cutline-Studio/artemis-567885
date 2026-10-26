# Artemis skills (top level)

What Artemis 567885 (dense) can do, which engine does it, and who may invoke it. "Local" means
the autonomous orchestrator or a human on the box; "external" means an A2A agent from another org.
Approvals follow [GOVERNANCE.md](GOVERNANCE.md).

| skill | engine | local | external |
|---|---|---|---|
| Train one cycle (train → eval → checkpoint/rollback → plan growth) | self_train | yes, within caps | never |
| Grow (add layer / widen d_ff), function-preserving, within caps | self_train | yes, within caps; raising caps = human edit | never |
| Record / look up / reuse a procedure, with success scoring | skills | yes | lookup only (read-only facade) |
| Write → run → observe → revise code in the sandbox | code_exec | yes, sandboxed | only via an environment `execute` grant |
| Choose the next action, bounded by steps, time and the kill switch | autonomous | yes | never |
| Signed handshake, scoped token, task delegation | a2a | yes | allowlisted orgs only |
| Load / build / generate / embed / score with the one model | transformer | yes | never |
| Exact, symbolic and arbitrary-precision math with worked, verified answers; Z3 proofs | math | yes (tool calls sandboxed) | never |
| Unit-checked physics: constants, kinematics, ODE simulation | physics | yes | never |
| Score and choose replies by ranked values | heart | yes | never |
| Retrieve from the docs with citations, or say "I don't know" | rag | yes | only scoped to a granted world (planned) |
| Plan model size and tokens for a compute budget; fit our own scaling curve | scaling | yes (plans only) | never |
| Train faster with the 20 accelerated-learning methods (flags, presets `baseline`/`fast`/`gpu_full`; see [Accelerate.md](Accelerate.md)) | accel | yes on CPU; GPU, DDP, pretrained weights and external teachers need Latoya's OK | never |
| Multi-turn chat: template, state, clarifying questions, grounding, tools | conversation | yes (REPL; API planned on 127.0.0.1) | never |
| Define any English word (WordNet + GCIDE, 292,125 records) | skills.define | yes | never |
| Test untrusted output in an isolated environment | sandbox.env / sandbox.py | yes | never |
| Constitution check before promotion (loose, strict on escalation) | custom_constitutional + promotion.py | yes | never |
| **Environment capability**: enter a defined environment ("world") and read / write / execute inside it | a2a.environments | yes | only with a signed per-agent, per-environment grant |

## Environment capability (summary; details in [worlds.md](worlds.md))
- An environment has an id, a type (`workspace`, `sandbox`, `runtime`, `model_weights` or
  `self_train`), a root dir, resources, and default-deny.
- A grant is a signed token tied to one agent principal and one environment. Its scopes are a
  subset of `read`, `write` and `execute`. It has an expiry, optional path prefixes, a
  per-minute rate limit, and a `jti` that can be revoked.
- Every `enter`/`read`/`write`/`execute`/`leave` is checked in this order: kill switch, session,
  token signature, audience, subject, expiry and revocation, environment, scope, path containment
  (realpath inside root, blocking `..`, absolute paths and symlink escape), path prefix, then rate
  limit. Each decision goes to the hash-chained audit log.
- `execute` runs only through the code_exec sandbox.
- `model_weights` and `self_train` environments are protected. A grant needs `human_approved=True`
  and `allow_protected=True`, set by the operator in code. An external request can never set
  them.

## Promotion
Everything Artemis generates is tested under [Sandbox.md](Sandbox.md) and promoted only through the
constitution gate ([CustomConstitutionalAI.md](CustomConstitutionalAI.md)).

## Not skills (refused by design)
Spending money, renting GPUs, network access from sandboxed code, pushing or merging code, deleting
files outside a run dir, and changing caps or policies. See [GOVERNANCE.md](GOVERNANCE.md).
