"""Runtime configuration, read once from the environment.

Every service imports the same `settings` object so that a deployment knob is
set in exactly one place. Defaults are chosen so the whole platform boots with
zero configuration (SQLite + in-memory broker + deterministic stub LLM), which
is what makes `make dev` and the test suite work on a laptop with no Redis,
no Postgres and no API keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- persistence -------------------------------------------------
    database_url: str = field(default_factory=lambda: _env("CWAP_DATABASE_URL", "sqlite:///./cwap.sqlite3"))
    sql_echo: bool = field(default_factory=lambda: _env_bool("CWAP_SQL_ECHO", False))

    # --- transport ---------------------------------------------------
    # "memory" runs the worker in-process (dev/test); "redis" uses a real broker.
    broker_backend: str = field(default_factory=lambda: _env("CWAP_BROKER", "memory"))
    redis_url: str = field(default_factory=lambda: _env("CWAP_REDIS_URL", "redis://localhost:6379/0"))
    work_queue: str = field(default_factory=lambda: _env("CWAP_WORK_QUEUE", "cwap.jobs"))
    dead_letter_queue: str = field(default_factory=lambda: _env("CWAP_DLQ", "cwap.jobs.dlq"))
    # Run the worker as a thread inside the API process. True by default with the
    # in-memory broker so `make dev` needs no second process; a Redis deployment
    # runs the worker separately and leaves this off.
    inline_worker: bool = field(
        default_factory=lambda: _env_bool("CWAP_INLINE_WORKER", _env("CWAP_BROKER", "memory") == "memory")
    )

    # --- auth --------------------------------------------------------
    jwt_secret: str = field(default_factory=lambda: _env("CWAP_JWT_SECRET", "dev-secret-change-me"))
    jwt_algorithm: str = field(default_factory=lambda: _env("CWAP_JWT_ALG", "HS256"))
    jwt_ttl_seconds: int = field(default_factory=lambda: _env_int("CWAP_JWT_TTL", 60 * 60 * 12))

    # --- llm proxy ---------------------------------------------------
    # "stub" keeps the platform runnable with no credentials; "anthropic" calls the real API.
    llm_provider: str = field(default_factory=lambda: _env("CWAP_LLM_PROVIDER", "stub"))
    llm_model: str = field(default_factory=lambda: _env("CWAP_LLM_MODEL", "claude-opus-5"))
    llm_effort: str = field(default_factory=lambda: _env("CWAP_LLM_EFFORT", "high"))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("CWAP_LLM_MAX_TOKENS", 16000))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    llm_timeout_seconds: int = field(default_factory=lambda: _env_int("CWAP_LLM_TIMEOUT", 120))

    # --- execution limits --------------------------------------------
    max_steps_per_run: int = field(default_factory=lambda: _env_int("CWAP_MAX_STEPS", 50))
    http_node_timeout_seconds: int = field(default_factory=lambda: _env_int("CWAP_HTTP_TIMEOUT", 20))
    http_node_allowlist: str = field(default_factory=lambda: _env("CWAP_HTTP_ALLOWLIST", ""))

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgres")

    @property
    def http_allowed_hosts(self) -> frozenset[str]:
        """Empty set means "no outbound HTTP node calls permitted"."""
        raw = self.http_node_allowlist.strip()
        if not raw:
            return frozenset()
        return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that need to re-read the environment."""
    get_settings.cache_clear()
