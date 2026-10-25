"""Signed, scoped, expiring capability tokens (also used for environment grants).

token = {"claims": {iss, sub, aud, kind, scopes, iat, exp, jti, ...extra}, "sig": Ed25519(iss key)}
"""
import time, uuid
from typing import Iterable, Optional
from .crypto import Identity, verify


class TokenError(Exception):
    pass


def issue(issuer: Identity, sub: str, aud: str, scopes: Iterable[str], ttl_s: float,
          kind: str = "capability", now: Optional[float] = None, **extra) -> dict:
    now = time.time() if now is None else now
    claims = {"iss": issuer.org_id, "sub": sub, "aud": aud, "kind": kind, "scopes": sorted(set(scopes)),
              "iat": now, "exp": now + ttl_s, "jti": uuid.uuid4().hex, **extra}
    return {"claims": claims, "sig": issuer.sign(claims)}


def check(token: dict, issuer_pub: str, *, aud: str, sub: Optional[str] = None, kind: str = "capability",
          revoked: Optional[set] = None, now: Optional[float] = None) -> dict:
    now = time.time() if now is None else now
    if not isinstance(token, dict) or "claims" not in token or "sig" not in token:
        raise TokenError("malformed token")
    c = token["claims"]
    if not verify(issuer_pub, c, token["sig"]):
        raise TokenError("bad token signature")
    if c.get("kind") != kind:
        raise TokenError(f"wrong token kind {c.get('kind')!r}")
    if c.get("aud") != aud:
        raise TokenError("token audience mismatch")
    if sub is not None and c.get("sub") != sub:
        raise TokenError("token subject mismatch")
    if now >= c.get("exp", 0):
        raise TokenError("token expired")
    if revoked and c.get("jti") in revoked:
        raise TokenError("token revoked")
    return c
