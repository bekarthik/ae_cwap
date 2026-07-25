"""Where a tenant's chosen model lives, and how it reaches a running step.

Changing model used to mean editing the environment and restarting, which is
wrong for the thing it configures: a user comparing a local Llama against a
hosted Claude wants to try one, look at the result, and try the other. So a
tenant's choice is stored, and `get_provider()` resolves it per call.

The interesting problem is *reaching* the call. A skill deep inside an agent
loop asks for a provider and has no tenant argument — threading one through
every signature would touch every executor for a concern none of them own. A
context variable is the honest fit: the worker sets it once from the job's
`job_context.tenant_id`, and everything under that step resolves the right
backend without knowing it happened. It is set and reset around one step, so a
worker thread that moves on to another tenant's job cannot inherit the previous
one's credentials.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime

from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import ModelSetting
from cwap_common.secrets import decrypt, encrypt

KIND_LLM = "llm"
KIND_EMBEDDING = "embedding"


def kind_for_provider(provider: str) -> str:
    """The `kind` a per-provider credential is stored under.

    An agent may run on a backend the workspace has not selected — that is the
    point of choosing a model per agent — and that backend needs its own key.
    Rather than a second table, each provider gets its own row in the one that
    already exists, keyed `llm:openai` alongside the plain `llm` row that says
    which of them the workspace runs on by default.
    """
    return f"{KIND_LLM}:{provider.strip().lower()}"[:32]

#: Whose configuration the current thread should resolve. Empty means "use the
#: deployment default", which is what the API process does outside a request and
#: what every existing test does without changing a line.
_current_tenant: ContextVar[str] = ContextVar("cwap_current_tenant", default="")


@dataclass(frozen=True)
class StoredModelConfig:
    provider: str
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    #: Seconds to wait for a reply. 0 means "use the deployment default".
    timeout_seconds: int = 0
    updated_by: str = ""
    updated_at: datetime | None = None

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    def redacted(self) -> dict[str, object]:
        """Safe to return over the API: says whether a key is set, never what."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "has_api_key": self.has_key,
            "timeout_seconds": self.timeout_seconds,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@contextmanager
def acting_for(tenant_id: str):
    """Resolve model configuration as this tenant for the duration of a block."""
    token = _current_tenant.set(tenant_id or "")
    try:
        yield
    finally:
        _current_tenant.reset(token)


def current_tenant() -> str:
    return _current_tenant.get()


def credential_for(tenant_id: str, provider: str) -> StoredModelConfig | None:
    """The credential this tenant has for one backend, wherever it was entered.

    Checked in the order a person would expect: a key saved specifically for
    this provider, then the workspace's main configuration if it happens to be
    the same provider. Nothing else — a key entered for OpenAI must never be
    sent to somebody else's endpoint.
    """
    if not tenant_id:
        return None

    specific = load(tenant_id, kind=kind_for_provider(provider))
    if specific is not None:
        return specific

    main = load(tenant_id)
    if main is not None and main.provider.strip().lower() == provider.strip().lower():
        return main
    return None


def load(tenant_id: str, kind: str = KIND_LLM) -> StoredModelConfig | None:
    """A tenant's saved choice, or None to mean "the deployment default"."""
    if not tenant_id:
        return None
    with read_only_session() as session:
        row = (
            session.query(ModelSetting)
            .filter_by(tenant_id=tenant_id, kind=kind)
            .one_or_none()
        )
        return _to_config(row) if row else None


def save(
    tenant_id: str,
    *,
    kind: str = KIND_LLM,
    provider: str,
    model: str = "",
    base_url: str = "",
    api_key: str | None = None,
    timeout_seconds: int = 0,
    updated_by: str = "",
) -> StoredModelConfig:
    """Store a choice. `api_key=None` keeps the existing key.

    That distinction matters: the API never returns a key, so a UI re-saving the
    form has nothing to send back. Without "None means unchanged", editing the
    model would silently erase the credential.
    """
    with unit_of_work() as session:
        row = (
            session.query(ModelSetting)
            .filter_by(tenant_id=tenant_id, kind=kind)
            .one_or_none()
        )
        if row is None:
            row = ModelSetting(tenant_id=tenant_id, kind=kind, provider=provider)
            session.add(row)

        row.provider = provider
        row.model = model
        row.base_url = base_url
        row.timeout_seconds = max(0, int(timeout_seconds or 0))
        row.updated_by = updated_by
        if api_key is not None:
            row.api_key = encrypt(api_key)
        session.flush()
        return _to_config(row)


def list_configs(tenant_id: str) -> list[StoredModelConfig]:
    """Every model configuration this tenant has saved, of any kind."""
    if not tenant_id:
        return []
    with read_only_session() as session:
        rows = (
            session.query(ModelSetting)
            .filter(ModelSetting.tenant_id == tenant_id)
            .filter(ModelSetting.kind.like(f"{KIND_LLM}%"))
            .all()
        )
        return [_to_config(row) for row in rows]


def clear(tenant_id: str, kind: str = KIND_LLM) -> bool:
    """Fall back to the deployment default."""
    with unit_of_work() as session:
        return bool(
            session.query(ModelSetting).filter_by(tenant_id=tenant_id, kind=kind).delete()
        )


def _to_config(row: ModelSetting) -> StoredModelConfig:
    return StoredModelConfig(
        provider=row.provider,
        model=row.model or "",
        base_url=row.base_url or "",
        api_key=decrypt(row.api_key or ""),
        timeout_seconds=row.timeout_seconds or 0,
        updated_by=row.updated_by or "",
        updated_at=row.updated_at,
    )
