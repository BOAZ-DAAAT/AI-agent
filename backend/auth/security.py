from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

_ALGO = "pbkdf2_sha256"
_ROUNDS = 120_000

# ── 비밀번호 해시 ──


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """비밀번호를 'pbkdf2_sha256$rounds$salt$hash' 문자열로 만든다."""
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ROUNDS)
    return f"{_ALGO}${_ROUNDS}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, b64salt, b64hash = stored.split("$")
        if algo != _ALGO:
            return False
        salt = _unb64(b64salt)
        expected = _unb64(b64hash)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(rounds))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

# ── 로그인 토큰 (HMAC 서명) ──

def create_token(secret: str, *, username: str, role: str, ttl_seconds: int) -> str:
    """HMAC-SHA256 서명 토큰: '<base64url payload>.<base64url sig>'."""
    payload = {"sub": username, "role": role, "exp": int(time.time()) + ttl_seconds}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _sign(secret, body)
    return f"{body}.{sig}"


def verify_token(secret: str, token: str) -> dict | None:
    """유효하면 payload dict, 아니면(위조·만료·형식오류) None."""
    try:
        body, sig = token.split(".")
        if not hmac.compare_digest(sig, _sign(secret, body)):
            return None
        payload = json.loads(_b64d(body))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def _sign(secret: str, body: str) -> str:
    return _b64e(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))