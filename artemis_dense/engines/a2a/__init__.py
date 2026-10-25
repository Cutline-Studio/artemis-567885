"""Artemis A2A engine (local prototype): agent cards, Ed25519 handshake, scoped tokens,
policy gate, environment permissioning, audit log, in-process loopback transport."""
from .crypto import Identity
from .cards import make_card, verify_card, WELL_KNOWN_PATH
from .policy import PolicyGate, OrgTrust, FACADE, PROTECTED, COSTLY, TRUST_SCOPES
from .environments import EnvironmentManager, Environment, EnvDenied
from .node import A2ANode, A2AClient, LoopbackTransport, sign_request
from .audit import AuditLog
