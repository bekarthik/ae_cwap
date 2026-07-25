"""Auth routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_gateway.schemas import LoginRequest, RegisterRequest, TokenResponse
from api_gateway.security import (
    Principal,
    authenticate,
    create_user,
    current_principal,
    issue_token,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _token_response(principal: Principal) -> TokenResponse:
    token, expires_in = issue_token(principal)
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user_id=principal.user_id,
        tenant_id=principal.tenant_id,
        scopes=sorted(principal.scopes),
    )


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(request: RegisterRequest) -> TokenResponse:
    principal = create_user(request.email, request.password, tenant_id=request.tenant_id)
    return _token_response(principal)


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest) -> TokenResponse:
    return _token_response(authenticate(request.email, request.password))


@router.get("/me", response_model=TokenResponse)
def me(principal: Principal = Depends(current_principal)) -> TokenResponse:
    """Refresh the caller's view of their own session (and rotate the token)."""
    return _token_response(principal)
