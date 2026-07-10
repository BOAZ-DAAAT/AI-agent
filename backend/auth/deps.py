from __future__ import annotations

from fastapi import Header, HTTPException

from backend.auth.security import verify_token
from backend.config import Auth


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    """Bearer 토큰을 검증해 payload를 돌려준다. AUTH_ENABLED=false면 통과."""
    if not Auth.ENABLED:
        return {"sub": "dev", "role": "admin"}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="인증이 필요합니다.")
    token = authorization[len("Bearer "):].strip()
    payload = verify_token(Auth.SECRET, token)
    if payload is None:
        raise HTTPException(status_code=401, detail="토큰이 유효하지 않거나 만료되었습니다.")
    return payload