# agent: a2a
- **Does:** handles cross-org collaboration through signed envelopes (Ed25519, nonce and
  timestamp), handshakes that issue scoped, expiring capability tokens, A2A-style tasks with
  idempotency, a policy gate, world permissioning, and a hash-chained audit log.
- **Transport:** in-process `LoopbackTransport` only. No socket is opened, and nothing is deployed
  or hosted.
- **Exposes:** facade only (`artemis.echo`, `artemis.text_stats`, `artemis.skills.lookup`, and
  `env.*` under grants).
- **Never exposes:** the model weights, the self_train loop, skill writes, code_exec, the
  orchestrator, or anything that costs money. Protected capabilities need an org grant plus local
  human approval, and the wire can't assert approval.
- **Writes only:** `<run_dir>/a2a_audit.jsonl`, the world audit log, and the world roots it was given.
- **Stops on:** the kill switch.
- **Docs:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/PROTOTYPE_SPEC.md](docs/PROTOTYPE_SPEC.md),
  [skills.md](skills.md), [../../worlds.md](../../worlds.md), [../../GOVERNANCE.md](../../GOVERNANCE.md).
