# Artemis A2A engine: architecture outline

Status: a **local prototype** (in-process loopback, no sockets). Nothing is deployed or hosted.
Items marked **(planned)** are not implemented yet.

## 1. Goals
Let Artemis and its engines work with agents from other companies without exposing internals.
Outsiders reach only a capability facade. Every request is signed, scoped, rate-limited, audited
and default-deny.

## 2. Components
| component | module | role |
|---|---|---|
| Identity + crypto | `crypto.py` | Ed25519 keypairs, canonical JSON, sign/verify |
| Agent card | `cards.py` | discovery manifest at `/.well-known/agent-card.json`, signed |
| Capability tokens | `tokens.py` | signed, scoped, expiring, revocable (`jti`) |
| Node / client / transport | `node.py` | signed envelopes, handshake, tasks, loopback transport |
| Policy gate | `policy.py` | allowlist and trust level, facade vs protected vs costly, rate limits |
| Environments ("worlds") | `environments.py` | per-agent, per-world grants; enter/read/write/execute/leave |
| Audit | `audit.py` | append-only, hash-chained JSONL |
| Rate limits | `ratelimit.py` | token bucket per org and per agent+world |

## 3. Discovery
- **Outbound:** fetch the peer's card from `/.well-known/agent-card.json` (prototype:
  `LoopbackTransport.discover`). Verify it against the key **pinned in our allowlist**, never
  against the key the card presents.
- **Inbound:** our card lists only facade skills. Identity data lives in `x-artemis`
  (orgId, agentId, publicKey, alg).

## 4. Authenticated cross-org handshake
1. The client sends `kind=handshake` with its signed card and `requested_scopes`, in a signed
   envelope (`ts`, `nonce`).
2. The server checks the kill switch, protocol, recipient, allowlist, signature (pinned key),
   timestamp within ±60 s, and that the nonce is unseen. It then verifies the card.
3. The server issues a capability token: scopes = requested ∩ (trust-level scopes ∪ protected
   grants), TTL 300 s, `aud` = server org, `sub` = `org/agent`.
4. Every later request carries the token in a fresh signed envelope with a new nonce.

mTLS **(planned)** as an extra transport-layer factor for high-trust peers. See
[../../../HTTP_HTTPS.md](../../../HTTP_HTTPS.md).

## 5. Task delegation and results
- The task body is `{messageId, idempotencyKey, capability, input, contextId?}`.
- Lifecycle uses A2A task-state names: `TASK_STATE_SUBMITTED → TASK_STATE_WORKING →
  TASK_STATE_COMPLETED | TASK_STATE_FAILED | TASK_STATE_REJECTED`. `CANCELED`, `INPUT_REQUIRED`
  and `AUTH_REQUIRED` are **(planned)**.
- Results come back as `artifacts[].parts[].data`.
- Idempotency: the same `(principal, idempotencyKey)` returns the original task, even inside a
  new signed request.
- `get_task` returns only the caller's own tasks. Anything else gets `NOT_FOUND`, so the reply
  doesn't reveal whether the task exists.

## 6. Trust boundary
```mermaid
flowchart TB
  X([External agent]) -->|signed envelope + token| N[A2ANode]
  N -->|sig / skew / nonce / token| P{PolicyGate}
  P -->|facade: echo, text_stats| F[Facade handlers]
  P -->|facade: skills.lookup, read-only| S[(SkillStore)]
  P -->|facade: env.*| E[EnvironmentManager]
  E -->|grant: scope + path + expiry + revocation + rate| W[(World root)]
  E -->|execute| C[code_exec sandbox]
  P -. protected: self_train.*, skills.write, model.weights*, code_exec.*, orchestrator.* .-> D[[DENY unless org grant + local human approval]]
  P -. costly: gpu.*, billing.* .-> D
  N & P & E --> A[(hash-chained audit log)]
  K{{KILL}} -.-> N & E
```
- **Facade only.** External agents can't call the model weights, the self_train loop, skill
  writes or the code_exec engine directly.
- **Protected capabilities** need an explicit per-org grant plus local human approval. The wire
  can never assert approval.
- **Costly capabilities** always need human approval.

## 7. Environment permissioning (worlds)
- **Worlds** have an id, type, root, resources and default-deny. `model_weights` and
  `self_train` types are protected.
- **Grants** are signed `env_grant` tokens bound to one agent and one world. Scopes are
  read/write/execute, each grant has an expiry, and optional path prefixes and a rate limit can
  be added. They are revocable by `jti`.
- **Every operation** is checked in order:
  1. Kill switch.
  2. Session owner.
  3. Token signature, audience, subject, expiry and revocation.
  4. World match.
  5. Scope.
  6. Path containment: realpath inside the root, blocking `..`, absolute paths and symlink escape.
  7. Path prefix.
  8. Rate limit.
- **Audit:** every check result is written to the audit log, allowed or denied.
- **Execute** runs through code_exec.
- **Protected worlds** need `human_approved` and `allow_protected`, which only the local
  operator can set.

Details: [../../../worlds.md](../../../worlds.md).

## 8. Rate limits, audit, human approval
- **Rate limits:** token bucket per org (`OrgTrust.rate_per_min`) and per agent+world
  (`rate_per_min` in the grant).
- **Audit:** every policy decision, handshake, task, grant, revocation and world operation is
  logged with allow or deny and the reason. `verify_chain()` detects edits.
- **Human approval:** required for costly and protected actions. **(planned)** an approval UI
  that issues one-shot, signed approval tokens.

## 9. Alignment with open protocols (verified 2026-10-25)
- **Google's A2A protocol, v1.0.0** (https://a2a-protocol.org/latest/specification/). We reuse
  its Agent Card field names, the well-known path `/.well-known/agent-card.json`, its
  `TASK_STATE_*` names, and its Task/Artifact/Part shapes.
  - Differences: our card signature is a detached Ed25519 field, not JWS `signatures`. Our
    binding is a custom in-process loopback, not JSON-RPC, gRPC or HTTP+JSON. We add signed
    envelopes with nonce and timestamp.
  - **(planned)** an HTTP+JSON binding (`POST /message:send`, `GET /tasks/{id}`).
- **MCP** (https://modelcontextprotocol.io/specification/2026-07-28) uses JSON-RPC 2.0 between
  hosts, clients and servers to expose tools and context to models. **(planned)** expose the
  facade capabilities as MCP tools behind the same policy gate.

## 10. Out of scope now
Any public server or open port; deployment; real mTLS/PKI; persisting tokens, nonces and grants
across restarts; streaming and push notifications; payments.
