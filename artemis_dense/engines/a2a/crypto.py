"""Ed25519 identities, canonical JSON, sign/verify (via the `cryptography` package)."""
import base64, json, os
from dataclasses import dataclass
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


def b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def canonical(obj) -> bytes:
    """Deterministic JSON bytes for signing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def new_nonce() -> str:
    return b64e(os.urandom(16))


@dataclass
class Identity:
    org_id: str
    agent_id: str
    private_key: Ed25519PrivateKey

    @classmethod
    def generate(cls, org_id: str, agent_id: str) -> "Identity":
        return cls(org_id, agent_id, Ed25519PrivateKey.generate())

    @property
    def principal(self) -> str:
        return f"{self.org_id}/{self.agent_id}"

    @property
    def public_key_b64(self) -> str:
        raw = self.private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return b64e(raw)

    def sign(self, obj) -> str:
        return b64e(self.private_key.sign(canonical(obj)))


def verify(public_key_b64: str, obj, sig_b64: str) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(b64d(public_key_b64)).verify(b64d(sig_b64), canonical(obj))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False
