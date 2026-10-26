# HTTP_HTTPS: Artemis's planned localhost API

**Status:** planned. The request handler behind it is built (`api_handler` in
`engines/conversation/__init__.py`), but **no server exists and none is started**. Nothing is
hosted, deployed or reachable from another machine.

Artemis is one dense LLM. HTTP is only a way to reach that one model from another program on the
same computer. It is also the planned wire format for Artemis's A2A (agent-to-agent) engine,
which today runs in-process over `LoopbackTransport` only.

## Purpose
Let a local program talk to Artemis over plain HTTP on `127.0.0.1`, without opening a public
port, so the chat engine and the A2A engine can be tested the way a real client would use them.

## What HTTP and HTTPS are (public standards)
- **HTTP** is the request/response protocol of the web. Its shared semantics (methods, status
  codes, headers) are defined in RFC 9110 (2022).
- **HTTPS** is HTTP sent inside a TLS connection. TLS 1.3 is defined in RFC 8446 (2018). TLS gives
  encryption, integrity and server authentication. With **mutual TLS (mTLS)** the client also
  presents a certificate.
- `127.0.0.1` is the IPv4 loopback address: traffic sent to it never leaves the machine.

## How it works in Artemis
*(built)* `api_handler(engine, {"messages": [...]})` takes the last user message, calls
`ChatEngine.respond`, and returns:
```json
{"message": {"role": "assistant", "content": "..."}, "citations": ["GPU.md#Guardrails"], "model_trained": false}
```
*(planned)* A tiny server that binds **only** to `127.0.0.1` and maps routes to existing pure
functions:

| route | handler | status |
|---|---|---|
| `POST /v1/chat` | `conversation.api_handler` | handler built, server planned |
| `GET /.well-known/agent-card.json` | `A2ANode.well_known` | loopback built, HTTP planned |
| `POST /a2a/handshake` | `kind=handshake` | loopback built, HTTP planned |
| `POST /a2a/tasks` | `kind=task` | loopback built, HTTP planned |
| `GET /a2a/tasks/{id}` | `kind=get_task` | loopback built, HTTP planned |

The A2A routes come from [engines/a2a/docs/PROTOTYPE_SPEC.md](engines/a2a/docs/PROTOTYPE_SPEC.md).
A2A messages stay signed envelopes (Ed25519, nonce, ±60 s skew) whatever the transport, so HTTP
adds no trust of its own.

## Rules for the planned server
- Bind address is `127.0.0.1` only, never `0.0.0.0`. `sandbox.env` already sets
  `ARTEMIS_API_BIND=127.0.0.1` and `ARTEMIS_API_PORT=0` (no port).
- No public hosting, no tunnels, no cloud deployment. Exposing it beyond the machine needs
  Latoya's explicit approval first.
- HTTPS with mTLS is planned for any future non-loopback A2A peer. Plain HTTP is acceptable only on
  loopback.
- Request size, rate limits and timeouts follow the A2A rate limiter (`engines/a2a/ratelimit.py`).
- No secrets in URLs, logs or responses. Keys are never sent over this API.
- The kill switch (`KILL` file) makes every request fail with `UNAVAILABLE`.

## Safety and governance
Follows [GOVERNANCE.md](GOVERNANCE.md). Network access beyond loopback is a human-approval action.
Inside the [sandbox](Sandbox.md) the network is off entirely.

## Example dialogues (training)
<!-- example -->
**User:** Can I call you from another computer?
**Artemis:** Not yet. My HTTP API is planned for 127.0.0.1 only, so it would only answer programs on the same machine. Opening it wider needs your explicit approval.

<!-- example -->
**User:** Is your API running right now?
**Artemis:** No. The chat handler is built, but no server has been started. It's planned as a localhost-only POST /v1/chat.

## Related
[Conversation.md](Conversation.md) · [A2A architecture](engines/a2a/docs/ARCHITECTURE.md) ·
[NFC.md](NFC.md) · [RadioBand.md](RadioBand.md) · [UltraWideband.md](UltraWideband.md) ·
[FreeSpaceOptical.md](FreeSpaceOptical.md) · [Laser.md](Laser.md) · [everything.md](everything.md)
