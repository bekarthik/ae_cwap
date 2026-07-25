"""Auth Service — registration, login, JWT issue/verify, request principal.

Passwords are stored as salted PBKDF2-HMAC-SHA256 digests. That is a deliberate
choice of the strongest primitive available without adding a native dependency;
a production deployment should move to Argon2id, which is a change to
`hash_password`/`verify_password` and nothing else.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import User
from cwap_contracts import JobContext, PermissionRequirement

PBKDF2_ROUNDS = 240_000

#: Everyone gets this on registration. It is enough to build and run workflows
#: made of reasoning, retrieval and transformation steps.
DEFAULT_SCOPES = ("READ_WORKFLOWS",)

#: Required by any workflow containing a node that can act on the outside world.
#: Deliberately *not* granted by default — an operator grants it explicitly.
WRITE_EXTERNAL = "WRITE_EXTERNAL"

bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt_hex, digest_hex = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        expected = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        )
    except (ValueError, TypeError):
        return False
    # Constant-time comparison: a timing oracle on password verification is a
    # real credential-discovery vector.
    return hmac.compare_digest(expected.hex(), digest_hex)


@dataclass(frozen=True)
class Principal:
    """The authenticated caller behind one request."""

    user_id: str
    email: str
    tenant_id: str
    scopes: frozenset[str]

    def job_context(self, *, needs_write: bool, trace_id: str | None = None) -> JobContext:
        """Build the security anchor that travels with a run's jobs."""
        return JobContext(
            tenant_id=self.tenant_id,
            initiating_user_id=self.user_id,
            permissions=PermissionRequirement(
                required_scope="READ_WORKFLOWS",
                required_write=WRITE_EXTERNAL if needs_write else None,
            ),
            trace_id=trace_id or f"tr_{uuid.uuid4().hex[:16]}",
        )


def create_user(email: str, password: str, *, tenant_id: str, scopes=None) -> Principal:
    normalized = email.strip().lower()
    with unit_of_work() as session:
        if session.query(User).filter_by(email=normalized).one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="an account with that email already exists",
            )
        user = User(
            id=f"usr_{uuid.uuid4().hex[:16]}",
            email=normalized,
            tenant_id=tenant_id,
            password_hash=hash_password(password),
            scopes=list(scopes or DEFAULT_SCOPES),
        )
        session.add(user)
        session.flush()
        return Principal(
            user_id=user.id,
            email=user.email,
            tenant_id=user.tenant_id,
            scopes=frozenset(user.scopes),
        )


def authenticate(email: str, password: str) -> Principal:
    normalized = email.strip().lower()
    with read_only_session() as session:
        user = session.query(User).filter_by(email=normalized).one_or_none()
        if user is None or not verify_password(password, user.password_hash):
            # One message for both cases: distinguishing them tells an attacker
            # which addresses are registered.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid email or password"
            )
        return Principal(
            user_id=user.id,
            email=user.email,
            tenant_id=user.tenant_id,
            scopes=frozenset(user.scopes or []),
        )


def issue_token(principal: Principal) -> tuple[str, int]:
    from cwap_common.settings import get_settings  # noqa: PLC0415 - read at call time

    settings = get_settings()
    now = datetime.now(timezone.utc)
    expires_in = settings.jwt_ttl_seconds
    payload = {
        "sub": principal.user_id,
        "email": principal.email,
        "tenant": principal.tenant_id,
        "scopes": sorted(principal.scopes),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_in


def decode_token(token: str) -> Principal:
    from cwap_common.settings import get_settings  # noqa: PLC0415

    settings = get_settings()
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="session expired; sign in again"
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials"
        ) from exc

    return Principal(
        user_id=claims["sub"],
        email=claims.get("email", ""),
        tenant_id=claims["tenant"],
        scopes=frozenset(claims.get("scopes", [])),
    )


def current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> Principal:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


def principal_from_query_token(token: str | None) -> Principal:
    """WebSocket authentication.

    Browsers cannot set headers on a WebSocket handshake, so the token arrives
    as a query parameter. It is the same signed JWT and is verified identically.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing token query parameter"
        )
    return decode_token(token)


def bootstrap_demo_user() -> Principal | None:
    """Create the demo account in dev mode so the UI has something to sign in as.

    Refuses to run unless `CWAP_DEMO_PASSWORD` is set, so a deployment cannot
    accidentally ship a well-known credential.
    """
    password = os.environ.get("CWAP_DEMO_PASSWORD")
    if not password:
        return None
    email = os.environ.get("CWAP_DEMO_EMAIL", "demo@example.com")
    with read_only_session() as session:
        existing = session.query(User).filter_by(email=email.lower()).one_or_none()
        if existing is not None:
            return Principal(
                user_id=existing.id,
                email=existing.email,
                tenant_id=existing.tenant_id,
                scopes=frozenset(existing.scopes or []),
            )
    return create_user(email, password, tenant_id="demo-tenant")
