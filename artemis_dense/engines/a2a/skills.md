# A2A engine skills

What the A2A engine lets Artemis do with agents from other orgs. This is the engine-level
companion to [../../skills.md](../../skills.md). Rules are in
[../../GOVERNANCE.md](../../GOVERNANCE.md).

| skill | function | status |
|---|---|---|
| Publish a signed agent card | `make_card`, `A2ANode.well_known` | implemented (loopback) |
| Verify a peer card against a pinned key | `verify_card` | implemented |
| Signed handshake → scoped capability token | `A2AClient.handshake`, `A2ANode._handshake` | implemented |
| Delegate a task and get the result (idempotent) | `A2AClient.send_task`, `kind=get_task` | implemented |
| Policy-gate every call (allowlist, trust level, facade/protected/costly, rate limit) | `PolicyGate.decide` | implemented |
| Tamper-evident audit | `AuditLog.record/verify_chain` | implemented |
| **Environment capability**: define worlds, issue signed per-agent grants, enforce enter/read/write/execute/leave | `EnvironmentManager` | implemented |
| HTTP+JSON binding, mTLS | — | planned ([../../HTTP_HTTPS.md](../../HTTP_HTTPS.md)) |
| MCP tool exposure of the facade | — | planned |

## Environment capability
- **Define:** `define(id, type, root)`. Types: `workspace`, `sandbox`, `runtime`,
  `model_weights` (protected) and `self_train` (protected). Every world is default-deny.
- **Grant:** `grant(principal, env_id, scopes⊆{read,write,execute}, ttl_s, paths, rate_per_min)`
  returns a signed `env_grant` token bound to that agent and that world (`aud = <org>:env:<id>`).
- **Revoke:** `revoke(jti)` takes effect on the next operation.
- **Operate:**
  - `enter` → `read` / `write` / `execute` → `leave`.
  - Each call checks the kill switch, session, token, world, scope, path containment and
    prefixes, and the rate limit, then writes the decision to the audit log.
  - `execute` runs in the code_exec sandbox.
- **Protected worlds:** a grant needs both `human_approved` and `allow_protected`, set locally by
  the operator.

## Firewall
This is the A2A firewall: the rules every inbound request passes, in order. Anything not
explicitly allowed is denied.
1. **Kill switch.** If engaged, reply `UNAVAILABLE`.
2. **Protocol and recipient.** Only `artemis-a2a/0.1` addressed to this org.
3. **Allowlist.** Only orgs with a pinned Ed25519 key. There is no trust on first use and no
   trust from network location or physical proximity.
4. **Signature.** The envelope must verify with the pinned key.
5. **Freshness.** The timestamp must be within ±60 s and the nonce must be unseen (replay
   protection).
6. **Token.** It must be signed by us, with `aud` = us, `sub` = the sender, unexpired and
   unrevoked.
7. **Policy.** Requests to `PROTECTED` and `COSTLY` capabilities, or to anything off the facade
   or outside the token scopes, are denied.
8. **Rate limit.** Per org, and per agent+world.
9. **World grant checks** for `env.*`.
10. **Audit.** Every allow or deny is logged with the reason.

Inbound content (task inputs, files written into worlds) is treated as data, never as
instructions to the engines. **(planned)** network-layer filtering and the HTTP limits in
[../../HTTP_HTTPS.md](../../HTTP_HTTPS.md).
