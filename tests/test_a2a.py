import os, time
import pytest
from artemis_dense.engines.a2a import (Identity, A2ANode, A2AClient, LoopbackTransport, OrgTrust,
                                       make_card, sign_request, EnvironmentManager, EnvDenied, AuditLog,
                                       WELL_KNOWN_PATH)
from artemis_dense.engines.a2a import tokens
from artemis_dense.engines.code_exec import CodeExecEngine, SandboxLimits
from artemis_dense.engines.skills import SkillStore
from artemis_dense import killswitch

ALL = ["artemis.echo", "artemis.text_stats", "artemis.skills.lookup", "env.enter", "env.read",
       "env.write", "env.execute", "env.leave", "self_train.run_cycle", "skills.write"]


@pytest.fixture
def world(tmp_path):
    artemis = Identity.generate("cutline", "artemis")
    partner = Identity.generate("partnerco", "helper")
    stranger = Identity.generate("strangerco", "bot")
    skills = SkillStore(str(tmp_path / "skills.json"))
    skills.acquire("sum-list", "Sum numbers", "sum numbers", "steps", [])
    audit = AuditLog(str(tmp_path / "env_audit.jsonl"))
    envs = EnvironmentManager(artemis, str(tmp_path), audit, CodeExecEngine(str(tmp_path), SandboxLimits(timeout_s=3)))
    node = A2ANode(artemis, str(tmp_path), {"partnerco": OrgTrust("partnerco", partner.public_key_b64, "partner")},
                   env_manager=envs, skills=skills)
    net = LoopbackTransport(); net.register(node)
    mk = lambda ident: A2AClient(ident, net, make_card(ident, ident.agent_id, "test agent", []))
    return {"node": node, "net": net, "envs": envs, "partner": mk(partner), "stranger": mk(stranger),
            "partner_id": partner, "artemis": artemis, "tmp": tmp_path}


def test_discovery_card_is_signed(world):
    card = world["net"].discover("cutline")
    from artemis_dense.engines.a2a import verify_card
    assert verify_card(card, world["artemis"].public_key_b64)
    assert world["node"].well_known("/other")["error"]["code"] == "NOT_FOUND"
    assert WELL_KNOWN_PATH == "/.well-known/agent-card.json"


def test_valid_handshake_scopes_are_intersected(world):
    r = world["partner"].handshake("cutline", ALL)
    scopes = r["token"]["claims"]["scopes"]
    assert "artemis.echo" in scopes and "env.read" in scopes
    assert "self_train.run_cycle" not in scopes and "skills.write" not in scopes


def test_bad_signature_rejected(world):
    req = sign_request(world["partner_id"], "cutline", "handshake",
                       {"card": world["partner"].card, "requested_scopes": ["artemis.echo"]})
    req["body"]["requested_scopes"] = ["artemis.echo", "env.write"]      # tamper after signing
    assert world["node"].handle(req)["error"]["message"] == "bad request signature"
    forged = Identity.generate("partnerco", "helper")                    # right org name, wrong key
    r = world["net"].send("cutline", sign_request(forged, "cutline", "handshake",
                                                  {"card": world["partner"].card, "requested_scopes": []}))
    assert r["error"]["message"] == "bad request signature"


def test_expired_token_rejected(world):
    node = world["node"]
    tok = tokens.issue(node.identity, "partnerco/helper", "cutline", ["artemis.echo"], ttl_s=-1)
    r = world["partner"].send_task("cutline", "artemis.echo", {"x": 1}, token=tok)
    assert r["error"]["message"] == "token expired"


def test_replayed_nonce_rejected(world):
    world["partner"].handshake("cutline", ["artemis.echo"])
    req = sign_request(world["partner_id"], "cutline", "task",
                       {"messageId": "m1", "idempotencyKey": "k1", "capability": "artemis.echo", "input": {}},
                       world["partner"].tokens["cutline"])
    assert "task" in world["net"].send("cutline", req)
    assert world["net"].send("cutline", req)["error"]["message"] == "replayed or missing nonce"


def test_stale_timestamp_rejected(world):
    req = sign_request(world["partner_id"], "cutline", "handshake",
                       {"card": world["partner"].card, "requested_scopes": []}, ts=time.time() - 3600)
    assert world["net"].send("cutline", req)["error"]["message"] == "timestamp outside allowed skew"


def test_non_allowlisted_org_rejected(world):
    r = world["stranger"].handshake("cutline", ["artemis.echo"])
    assert r["error"]["code"] == "PERMISSION_DENIED"


def test_delegated_task_round_trip_and_idempotency(world):
    c = world["partner"]
    c.handshake("cutline", ALL)
    r = c.send_task("cutline", "artemis.text_stats", {"text": "one big brain"}, idempotency_key="abc")
    t = r["task"]
    assert t["status"]["state"] == "TASK_STATE_COMPLETED"
    assert t["artifacts"][0]["parts"][0]["data"] == {"chars": 13, "bytes": 13, "words": 3}
    again = c.send_task("cutline", "artemis.text_stats", {"text": "different"}, idempotency_key="abc")
    assert again["idempotent"] and again["task"]["id"] == t["id"]
    got = c.transport.send("cutline", sign_request(c.identity, "cutline", "get_task", {"id": t["id"]},
                                                   c.tokens["cutline"]))
    assert got["task"]["id"] == t["id"]
    look = c.send_task("cutline", "artemis.skills.lookup", {"query": "sum numbers"})
    assert look["task"]["artifacts"][0]["parts"][0]["data"]["skills"][0]["name"] == "sum-list"


def test_external_agent_cannot_reach_self_train_or_write_skills(world, tmp_path):
    c = world["partner"]
    c.handshake("cutline", ALL)
    for cap in ["self_train.run_cycle", "skills.write", "skills.acquire", "model.weights.read", "code_exec.run",
                "gpu.rent"]:
        t = c.send_task("cutline", cap, {"humanApproval": True})["task"]
        assert t["status"]["state"] == "TASK_STATE_REJECTED", cap
    # even a token that names self_train (forged by us for the test) cannot pass the policy gate
    tok = tokens.issue(world["node"].identity, "partnerco/helper", "cutline", ["self_train.run_cycle"], 60)
    t = c.send_task("cutline", "self_train.run_cycle", {}, token=tok)["task"]
    assert t["status"]["state"] == "TASK_STATE_REJECTED"
    assert world["node"].audit.verify_chain()


def test_kill_switch_blocks_a2a(world):
    killswitch.engage(str(world["tmp"]))
    assert world["partner"].handshake("cutline", ALL)["error"]["code"] == "UNAVAILABLE"


# ---------------------------------------------------------------- environment permissioning
@pytest.fixture
def envs(world):
    m = world["envs"]
    a = m.define("ws-a", "workspace", str(world["tmp"] / "envs" / "a"))
    b = m.define("ws-b", "sandbox", str(world["tmp"] / "envs" / "b"))
    open(os.path.join(a.root, "notes.txt"), "w").write("hello from A")
    open(os.path.join(b.root, "notes.txt"), "w").write("hello from B")
    open(os.path.join(world["tmp"], "outside.txt"), "w").write("secret-ish")
    return m


P = "partnerco/helper"


def test_read_only_agent_cannot_write(envs):
    g = envs.grant(P, "ws-a", ["read"])
    sid = envs.enter(P, "ws-a", g)
    assert envs.read(sid, P, "notes.txt") == "hello from A"
    with pytest.raises(EnvDenied, match="scope 'write'"):
        envs.write(sid, P, "new.txt", "x")


def test_write_without_execute_cannot_run(envs):
    g = envs.grant(P, "ws-a", ["read", "write"])
    sid = envs.enter(P, "ws-a", g)
    envs.write(sid, P, "run.py", "print('ran')")
    with pytest.raises(EnvDenied, match="scope 'execute'"):
        envs.execute(sid, P, "run.py")


def test_execute_routes_through_sandbox(envs):
    g = envs.grant(P, "ws-a", ["read", "write", "execute"])
    sid = envs.enter(P, "ws-a", g)
    envs.write(sid, P, "run.py", "print(open('notes.txt').read())")
    r = envs.execute(sid, P, "run.py")
    assert r["ok"] and r["stdout"].strip() == "hello from A" and "rlimits" in r["sandbox"]


@pytest.mark.parametrize("bad", ["../outside.txt", "../../outside.txt", "sub/../../outside.txt", "/etc/passwd"])
def test_path_traversal_blocked(envs, bad):
    g = envs.grant(P, "ws-a", ["read", "write"])
    sid = envs.enter(P, "ws-a", g)
    with pytest.raises(EnvDenied):
        envs.read(sid, P, bad)
    with pytest.raises(EnvDenied):
        envs.write(sid, P, bad, "x")


def test_symlink_escape_blocked(envs, world):
    os.symlink(str(world["tmp"] / "outside.txt"), os.path.join(envs.envs["ws-a"].root, "link.txt"))
    sid = envs.enter(P, "ws-a", envs.grant(P, "ws-a", ["read"]))
    with pytest.raises(EnvDenied, match="escapes"):
        envs.read(sid, P, "link.txt")


def test_path_prefix_constraint(envs):
    sid = envs.enter(P, "ws-a", envs.grant(P, "ws-a", ["read", "write"], paths=["out"]))
    envs.write(sid, P, "out/result.txt", "ok")
    with pytest.raises(EnvDenied, match="prefixes"):
        envs.read(sid, P, "notes.txt")


def test_expired_and_revoked_grants_denied(envs):
    expired = envs.grant(P, "ws-a", ["read"], ttl_s=-1)
    with pytest.raises(EnvDenied, match="expired"):
        envs.enter(P, "ws-a", expired)
    g = envs.grant(P, "ws-a", ["read"])
    sid = envs.enter(P, "ws-a", g)
    envs.read(sid, P, "notes.txt")
    envs.revoke(g["claims"]["jti"])
    with pytest.raises(EnvDenied, match="revoked"):
        envs.read(sid, P, "notes.txt")


def test_grant_bound_to_agent_and_env(envs):
    g = envs.grant(P, "ws-a", ["read"])
    with pytest.raises(EnvDenied, match="subject"):
        envs.enter("partnerco/other", "ws-a", g)
    with pytest.raises(EnvDenied, match="audience"):
        envs.enter(P, "ws-b", g)


def test_same_agent_different_grants_in_two_envs(envs):
    sa = envs.enter(P, "ws-a", envs.grant(P, "ws-a", ["read"]))
    sb = envs.enter(P, "ws-b", envs.grant(P, "ws-b", ["read", "write"]))
    with pytest.raises(EnvDenied):
        envs.write(sa, P, "x.txt", "no")
    assert envs.write(sb, P, "x.txt", "yes") == 3
    assert envs.read(sb, P, "x.txt") == "yes"


def test_protected_env_needs_human_approval(envs, world):
    envs.define("weights", "model_weights", str(world["tmp"] / "envs" / "weights"))
    envs.define("train", "self_train", str(world["tmp"] / "envs" / "train"))
    for e in ("weights", "train"):
        with pytest.raises(EnvDenied, match="protected"):
            envs.grant(P, e, ["read"])
        with pytest.raises(EnvDenied):
            envs.grant(P, e, ["read"], human_approved=True)          # still needs allow_protected
    assert envs.grant(P, "weights", ["read"], human_approved=True, allow_protected=True)


def test_env_ops_over_a2a_and_audit(envs, world):
    c = world["partner"]
    c.handshake("cutline", ALL)
    g = envs.grant(P, "ws-a", ["read"])
    sid = c.send_task("cutline", "env.enter", {"env": "ws-a", "grant": g})["task"]["artifacts"][0]["parts"][0]["data"]["session"]
    ok = c.send_task("cutline", "env.read", {"session": sid, "path": "notes.txt"})["task"]
    assert ok["artifacts"][0]["parts"][0]["data"]["content"] == "hello from A"
    bad = c.send_task("cutline", "env.write", {"session": sid, "path": "n.txt", "content": "x"})["task"]
    assert bad["status"]["state"] == "TASK_STATE_REJECTED"
    log = envs.audit.entries()
    assert any(e["event"] == "env.write" and not e["allowed"] for e in log)
    assert envs.audit.verify_chain()
