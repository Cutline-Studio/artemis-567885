# A2A prototype spec (v0.1, local only)

## Scope
- Two or more simulated orgs in one Python process exchange signed JSON through
  `LoopbackTransport`. No sockets, no server, no deployment.
- The prototype covers discovery, handshake, task delegation and result, `get_task`, the policy
  gate, environment grants and operations, the audit log, and the kill switch.

## Message schemas
**Signed envelope** (every request):
```json
{"header": {"protocol": "artemis-a2a/0.1", "kind": "handshake|task|get_task",
            "from_org": "partnerco", "from_agent": "helper", "to_org": "cutline",
            "ts": 1793000000.0, "nonce": "b64url-16-bytes", "token": {"claims": {}, "sig": "..."}},
 "body": {},
 "sig": "b64url Ed25519 over canonical({header, body})"}
```
**Handshake body / response:**
```json
{"card": {"name": "...", "x-artemis": {"orgId": "partnerco", "agentId": "helper", "publicKey": "...", "alg": "Ed25519"}, "x-artemis-signature": "..."},
 "requested_scopes": ["artemis.echo", "env.read"]}
```
```json
{"token": {"claims": {"iss": "cutline", "sub": "partnerco/helper", "aud": "cutline", "kind": "capability",
                      "scopes": ["artemis.echo", "env.read"], "iat": 0, "exp": 300, "jti": "hex"}, "sig": "..."},
 "card": {}}
```
**Task body / response:**
```json
{"messageId": "hex", "idempotencyKey": "hex", "capability": "artemis.text_stats", "input": {"text": "one big brain"}}
```
```json
{"task": {"id": "hex", "contextId": "hex",
          "status": {"state": "TASK_STATE_COMPLETED", "timestamp": "2026-10-25T22:00:00Z"},
          "artifacts": [{"artifactId": "hex", "parts": [{"data": {"chars": 13, "bytes": 13, "words": 3}}]}],
          "history": [], "metadata": {"principal": "partnerco/helper", "capability": "artemis.text_stats"}}}
```
**Environment grant** (issued locally by the operator, carried by the agent):
```json
{"claims": {"iss": "cutline", "sub": "partnerco/helper", "aud": "cutline:env:ws-a", "kind": "env_grant",
            "scopes": ["read"], "env": "ws-a", "paths": [""], "rate_per_min": 60,
            "iat": 0, "exp": 600, "jti": "hex"}, "sig": "..."}
```
**Error:** `{"error": {"code": "UNAUTHENTICATED|PERMISSION_DENIED|UNAVAILABLE|VERSION_NOT_SUPPORTED|INVALID_ARGUMENT|NOT_FOUND", "message": "reason"}}`

## Endpoints (logical; loopback kinds now, HTTP paths planned in ../../../HTTP_HTTPS.md)
| logical | prototype | planned HTTP |
|---|---|---|
| card | `node.well_known("/.well-known/agent-card.json")` | `GET /.well-known/agent-card.json` |
| handshake | `kind=handshake` | `POST /a2a/handshake` |
| task submit | `kind=task` | `POST /a2a/tasks` |
| task status/result | `kind=get_task` | `GET /a2a/tasks/{id}` |
| world enter/act/leave | `task` with `capability=env.enter / env.read / env.write / env.execute / env.leave` | `POST /a2a/worlds/{id}:enter` etc. |

## Auth flow, step by step
1. Out of band, each org pins the other's Ed25519 public key in its allowlist (`OrgTrust`) and
   assigns a trust level.
2. The client fetches the peer's card and verifies it against the pinned key.
3. The client sends a handshake in a signed envelope with a fresh nonce and the current `ts`.
4. The server checks, in order: kill switch, protocol, `to_org`, allowlist, signature with the
   pinned key, |now − ts| ≤ 60 s, and that the nonce is unseen (nonces are kept for 2× the skew
   window). It then verifies the card's signature and identity.
5. The server issues a token with scopes = requested ∩ allowed, TTL 300 s.
6. For each task, the client sends a new signed envelope with a new nonce and the token.
7. The server re-runs step 4 and checks the token (signature with its own key, `aud`,
   `sub` = sender principal, `exp`, revocation). It then runs the policy gate and the
   idempotency check, and executes the capability.
8. Every step is audited.

## Policy rules
1. Org not allowlisted → deny.
2. Rate limit exceeded → deny.
3. Costly (`gpu.*`, `compute.paid*`, `billing.*`) → deny without human approval.
4. Protected (`self_train.*`, `skills.write`, `skills.acquire`, `model.weights*`, `code_exec.*`,
   `orchestrator.*`, `autonomous.*`) → deny unless the org has an explicit protected grant AND
   local human approval AND the capability is in the token scopes. The wire can't supply
   approval.
5. Not on the facade → deny.
6. Not in the token scopes → deny.
7. Otherwise → allow.
8. World operations add the grant checks described in [../../../worlds.md](../../../worlds.md).

## Test plan (implemented in `tests/test_a2a.py`, all passing)
- Discovery: the card verifies against the pinned key, and a wrong path returns `NOT_FOUND`.
- Valid handshake: scopes are intersected, and protected scopes are never issued.
- Rejections:
  - A request tampered after signing.
  - A forged key using a real org name.
  - An expired token.
  - A replayed nonce.
  - A stale timestamp.
  - A non-allowlisted org.
- Delegated task round trip, the idempotent replay, `get_task`, and a read-only skills lookup.
- An external call to `self_train.run_cycle`, `skills.write`, `skills.acquire`,
  `model.weights.read`, `code_exec.run` or `gpu.rent` is rejected, even with a token that names
  `self_train`. The audit chain verifies afterwards.
- The kill switch blocks A2A.
- Worlds:
  - Read-only can't write.
  - Write without execute can't run.
  - Execute goes through the sandbox.
  - `../`, absolute paths and symlink escapes are blocked.
  - Path prefixes are enforced.
  - Expired and revoked grants are denied.
  - Grants are bound to one agent and one world.
  - Two worlds with different grants behave differently.
  - Protected worlds need human approval.
  - World operations over A2A are audited.

## Explicitly out of scope
- Any listening socket, public or private; deployment and hosting.
- Real TLS/mTLS and PKI or certificate management.
- Persistent storage of nonces, tokens, grants and revocations.
- JWS card signatures; streaming, push notifications, `CANCELED` and `INPUT_REQUIRED` flows.
- An MCP binding; payments; any GPU use; proximity channels (NFC, BLE, UWB, optical).
