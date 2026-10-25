"""Agent cards: discovery manifests published at WELL_KNOWN_PATH.

Field names follow the A2A AgentCard (name, description, supportedInterfaces, provider, version,
capabilities, defaultInputModes, defaultOutputModes, skills). Artemis-specific identity data
(orgId, agentId, Ed25519 publicKey) lives under "x-artemis". The prototype signs the card with a
detached Ed25519 signature over canonical JSON ("x-artemis-signature"), NOT the JWS
`signatures` field of the A2A spec; switching to JWS is listed as future work.
"""
from typing import List
from .crypto import Identity, verify

WELL_KNOWN_PATH = "/.well-known/agent-card.json"   # A2A v1.0 well-known path
PROTOCOL_VERSION = "artemis-a2a/0.1"


def make_card(identity: Identity, name: str, description: str, skills: List[dict],
              url: str = None) -> dict:
    card = {
        "name": name, "description": description, "version": "0.1.0",
        "provider": {"organization": identity.org_id, "url": f"loopback://{identity.org_id}"},
        "supportedInterfaces": [{"url": url or f"loopback://{identity.org_id}/a2a",
                                 "protocolBinding": "https://artemis.local/bindings/loopback/v0",
                                 "protocolVersion": PROTOCOL_VERSION}],
        "capabilities": {"streaming": False, "pushNotifications": False, "extendedAgentCard": False},
        "defaultInputModes": ["application/json"], "defaultOutputModes": ["application/json"],
        "skills": skills,
        "x-artemis": {"orgId": identity.org_id, "agentId": identity.agent_id,
                      "publicKey": identity.public_key_b64, "alg": "Ed25519"},
    }
    return {**card, "x-artemis-signature": identity.sign(card)}


def verify_card(card: dict, pinned_pub: str) -> bool:
    """Verify against the key PINNED in our allowlist, never the key the card presents."""
    body = {k: v for k, v in card.items() if k != "x-artemis-signature"}
    return body.get("x-artemis", {}).get("publicKey") == pinned_pub and \
        verify(pinned_pub, body, card.get("x-artemis-signature", ""))
