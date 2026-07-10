from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.auth.accounts import ACCOUNTS
from backend.auth.deps import get_current_user
from backend.auth.schemas import LoginRequest, LoginResponse, UserInfo
from backend.auth.security import create_token, verify_password
from backend.config import Auth

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    account = ACCOUNTS.get(payload.username)
    if account is None or not verify_password(payload.password, account["password_hash"]):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    token = create_token(
        Auth.SECRET,
        username=payload.username,
        role=account["role"],
        ttl_seconds=Auth.TOKEN_TTL_SECONDS,
    )
    return LoginResponse(token=token, user=UserInfo(username=payload.username, role=account["role"]))


@router.get("/me", response_model=UserInfo)
def me(user: dict = Depends(get_current_user)) -> UserInfo:
    return UserInfo(username=user["sub"], role=user["role"])