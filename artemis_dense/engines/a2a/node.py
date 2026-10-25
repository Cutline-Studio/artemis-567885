"""A2A node (one org's endpoint), client, signed request envelope, and in-process loopback transport.

Request  = {"header": {protocol, kind, from_org, from_agent, to_org, ts, nonce, token?},
            "body": {...}, "sig": Ed25519(sender key, canonical({header, body}))}
Kinds    : "handshake" -> issues a scoped capability token
           "task"      -> runs one facade capability, returns an A2A-style Task
           "get_task"  -> returns a stored Task
Checks (in order, all audited): kill switch, protocol, to_org, org allowlist, signature with
PINNED key, timestamp skew, nonce replay, then kind-specific (card / token / policy / idempotency).
No sockets are opened anywhere in this module.
"""
import copy, json, time, uuid
from typing import Callable, Dict, Optional

from . import tokens
from .audit import AuditLog
from .cards import make_card, verify_card, PROTOCOL_VERSION, WELL_KNOWN_PATH
from .crypto import Identity, verify, new_nonce
from .policy import PolicyGate, OrgTrust
from ... import killswitch

TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"}


def _iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t or time.time()))


def sign_request(identity: Identity, to_org: str, kind: str, body: dict, token: Optional[dict] = None,
                 ts: Optional[float] = None, nonce: Optional[str] = None) -> dict:
    header = {"protocol": PROTOCOL_VERSION, "kind": kind, "from_org": identity.org_id,
              "from_agent": identity.agent_id, "to_org": to_org, "ts": time.time() if ts is None else ts,
              "nonce": nonce or new_nonce()}
    if token is not None:
        header["token"] = token
    return {"header": header, "body": body, "sig": identity.sign({"header": header, "body": body})}


class A2ANode:
    def __init__(self, identity: Identity, run_dir: str, allowlist: Dict[str, OrgTrust],
                 handlers: Optional[Dict[str, Callable]] = None, env_manager=None, skills=None,
                 skew_s: float = 60.0, token_ttl_s: float = 300.0, clock=time.time):
        self.identity, self.run_dir = identity, run_dir
        self.audit = AuditLog(f"{run_dir}/a2a_audit.jsonl")
        self.policy = PolicyGate(self.audit, allowlist)
        self.skew_s, self.token_ttl_s, self.clock = skew_s, token_ttl_s, clock
        self.seen_nonces: Dict[str, float] = {}
        self.revoked: set = set()
        self.tasks: Dict[str, dict] = {}
        self.idem: Dict[str, str] = {}
        self.envs = env_manager
        self.handlers = {"artemis.echo": lambda inp, ctx: {"echo": inp},
                         "artemis.text_stats": lambda inp, ctx: {"chars": len(inp.get("text", "")),
                                                                 "bytes": len(inp.get("text", "").encode()),
                                                                 "words": len(inp.get("text", "").split())}}
        if skills is not None:   # read-only facade over the skills store (no writes)
            self.handlers["artemis.skills.lookup"] = lambda inp, ctx: {
                "skills": [{k: s[k] for k in ("name", "description", "trigger")}
                           for s in skills.lookup(inp.get("query", ""), k=int(inp.get("k", 3)))]}
        if env_manager is not None:
            m = env_manager
            self.handlers.update({
                "env.enter": lambda i, c: {"session": m.enter(c["principal"], i["env"], i["grant"])},
                "env.read": lambda i, c: {"content": m.read(i["session"], c["principal"], i["path"])},
                "env.write": lambda i, c: {"bytes": m.write(i["session"], c["principal"], i["path"], i["content"])},
                "env.execute": lambda i, c: m.execute(i["session"], c["principal"], i["path"]),
                "env.leave": lambda i, c: (m.leave(i["session"], c["principal"]), {"left": True})[1],
            })
        self.handlers.update(handlers or {})
        self.card = make_card(identity, f"Artemis 567885 ({identity.org_id})",
                              "Artemis dense model capability facade (prototype, loopback only)",
                              [{"id": k, "name": k, "description": f"facade capability {k}", "tags": ["artemis"]}
                               for k in sorted(self.handlers)])

    # ---------------------------------------------------------------- discovery
    def well_known(self, path: str = WELL_KNOWN_PATH) -> dict:
        if path != WELL_KNOWN_PATH:
            return {"error": {"code": "NOT_FOUND", "message": path}}
        return copy.deepcopy(self.card)

    # ---------------------------------------------------------------- inbound
    def _reject(self, why: str, hdr: dict, code: str = "UNAUTHENTICATED") -> dict:
        self.audit.record("a2a.request", False, kind=hdr.get("kind"), org=hdr.get("from_org"),
                          agent=hdr.get("from_agent"), reason=why)
        return {"error": {"code": code, "message": why}}

    def handle(self, req: dict) -> dict:
        hdr = (req or {}).get("header", {})
        if killswitch.is_engaged(self.run_dir):
            return self._reject("kill switch engaged", hdr, "UNAVAILABLE")
        if hdr.get("protocol") != PROTOCOL_VERSION:
            return self._reject("unsupported protocol version", hdr, "VERSION_NOT_SUPPORTED")
        if hdr.get("to_org") != self.identity.org_id:
            return self._reject("wrong recipient", hdr, "INVALID_ARGUMENT")
        trust = self.policy.allowlist.get(hdr.get("from_org"))
        if trust is None:
            return self._reject("org not allowlisted", hdr, "PERMISSION_DENIED")
        if not verify(trust.public_key, {"header": hdr, "body": req.get("body")}, req.get("sig", "")):
            return self._reject("bad request signature", hdr)
        now = self.clock()
        if abs(now - float(hdr.get("ts", 0))) > self.skew_s:
            return self._reject("timestamp outside allowed skew", hdr)
        for n, t in list(self.seen_nonces.items()):          # prune outside the skew window
            if now - t > 2 * self.skew_s:
                del self.seen_nonces[n]
        nkey = f"{hdr['from_org']}|{hdr.get('nonce')}"
        if not hdr.get("nonce") or nkey in self.seen_nonces:
            return self._reject("replayed or missing nonce", hdr)
        self.seen_nonces[nkey] = now
        principal = f"{hdr['from_org']}/{hdr['from_agent']}"
        kind = hdr.get("kind")
        if kind == "handshake":
            return self._handshake(hdr, req["body"], trust, principal)
        try:
            claims = tokens.check(hdr.get("token"), self.identity.public_key_b64, aud=self.identity.org_id,
                                  sub=principal, revoked=self.revoked, now=now)
        except tokens.TokenError as e:
            return self._reject(str(e), hdr)
        if kind == "task":
            return self._task(hdr, req["body"], claims, principal)
        if kind == "get_task":
            t = self.tasks.get(req["body"].get("id"))
            if not t or t["metadata"]["principal"] != principal:
                return {"error": {"code": "NOT_FOUND", "message": "TaskNotFoundError"}}
            return {"task": copy.deepcopy(t)}
        return self._reject("unknown kind", hdr, "INVALID_ARGUMENT")

    def _handshake(self, hdr, body, trust: OrgTrust, principal):
        card = body.get("card", {})
        if not verify_card(card, trust.public_key) or card["x-artemis"]["orgId"] != hdr["from_org"]:
            return self._reject("agent card signature/identity invalid", hdr)
        granted = set(body.get("requested_scopes", [])) & (self.policy.allowed_scopes(hdr["from_org"]) |
                                                           trust.protected_grants)
        tok = tokens.issue(self.identity, principal, self.identity.org_id, granted, self.token_ttl_s,
                           now=self.clock())
        self.audit.record("a2a.handshake", True, principal=principal, scopes=sorted(granted),
                          jti=tok["claims"]["jti"])
        return {"token": tok, "card": self.well_known()}

    def _task(self, hdr, body, claims, principal):
        idem = f"{principal}|{body.get('idempotencyKey') or body.get('messageId')}"
        if idem in self.idem:                                   # idempotent replay of a *new* signed request
            return {"task": copy.deepcopy(self.tasks[self.idem[idem]]), "idempotent": True}
        tid = uuid.uuid4().hex
        cap = body.get("capability", "")
        task = {"id": tid, "contextId": body.get("contextId") or uuid.uuid4().hex,
                "status": {"state": "TASK_STATE_SUBMITTED", "timestamp": _iso()},
                "artifacts": [], "history": [{"role": "ROLE_USER", "messageId": body.get("messageId"),
                                              "parts": [{"data": {"capability": cap, "input": body.get("input")}}]}],
                "metadata": {"principal": principal, "capability": cap}}
        self.tasks[tid] = task
        self.idem[idem] = tid
        # An external party can never assert human approval: any "humanApproval" field in the body
        # is ignored. Approval for protected/costly capabilities is a local human action (planned).
        ok, why = self.policy.decide(principal, hdr["from_org"], cap, claims["scopes"],
                                     human_approved=False, task_id=tid)
        if not ok:
            task["status"] = {"state": "TASK_STATE_REJECTED", "timestamp": _iso(), "message": why}
            return {"task": copy.deepcopy(task)}
        task["status"] = {"state": "TASK_STATE_WORKING", "timestamp": _iso()}
        try:
            out = self.handlers[cap](body.get("input") or {}, {"principal": principal, "task_id": tid})
            task["artifacts"] = [{"artifactId": uuid.uuid4().hex, "parts": [{"data": out}]}]
            task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": _iso()}
        except PermissionError as e:
            task["status"] = {"state": "TASK_STATE_REJECTED", "timestamp": _iso(), "message": str(e)}
        except Exception as e:
            task["status"] = {"state": "TASK_STATE_FAILED", "timestamp": _iso(), "message": f"{type(e).__name__}: {e}"}
        self.audit.record("a2a.task", task["status"]["state"] == "TASK_STATE_COMPLETED", principal=principal,
                          capability=cap, task_id=tid, state=task["status"]["state"])
        return {"task": copy.deepcopy(task)}


class LoopbackTransport:
    """In-process 'network': JSON round-trip between registered nodes. No sockets."""

    def __init__(self):
        self.nodes: Dict[str, A2ANode] = {}

    def register(self, node: A2ANode):
        self.nodes[node.identity.org_id] = node

    def discover(self, org_id: str) -> dict:
        return json.loads(json.dumps(self.nodes[org_id].well_known(WELL_KNOWN_PATH)))

    def send(self, to_org: str, req: dict) -> dict:
        return json.loads(json.dumps(self.nodes[to_org].handle(json.loads(json.dumps(req)))))


class A2AClient:
    def __init__(self, identity: Identity, transport: LoopbackTransport, card: dict):
        self.identity, self.transport, self.card = identity, transport, card
        self.tokens: Dict[str, dict] = {}

    def handshake(self, peer_org: str, scopes) -> dict:
        r = self.transport.send(peer_org, sign_request(self.identity, peer_org, "handshake",
                                                       {"card": self.card, "requested_scopes": sorted(scopes)}))
        if "token" in r:
            self.tokens[peer_org] = r["token"]
        return r

    def send_task(self, peer_org: str, capability: str, inp: dict, idempotency_key: Optional[str] = None,
                  token: Optional[dict] = None) -> dict:
        body = {"messageId": uuid.uuid4().hex, "idempotencyKey": idempotency_key or uuid.uuid4().hex,
                "capability": capability, "input": inp}
        return self.transport.send(peer_org, sign_request(self.identity, peer_org, "task", body,
                                                          token or self.tokens.get(peer_org)))
