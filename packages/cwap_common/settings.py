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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
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
    # "stub" (offline, no credentials), "anthropic", or any preset naming an
    # OpenAI-compatible server: ollama, vllm, lmstudio, llamacpp, openai,
    # together, groq, openrouter, mistral, deepseek, fireworks, litellm.
    # See llm_proxy.presets for the full table.
    llm_provider: str = field(default_factory=lambda: _env("CWAP_LLM_PROVIDER", "stub"))
    llm_model: str = field(default_factory=lambda: _env("CWAP_LLM_MODEL", ""))
    llm_base_url: str = field(default_factory=lambda: _env("CWAP_LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: _env("CWAP_LLM_API_KEY", ""))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("CWAP_LLM_MAX_TOKENS", 4096))
    llm_timeout_seconds: int = field(default_factory=lambda: _env_int("CWAP_LLM_TIMEOUT", 120))
    # "auto" tries native tool calling and permanently downgrades to a prompted
    # JSON protocol if the server rejects it. "native" or "prompted" force one.
    llm_tool_mode: str = field(default_factory=lambda: _env("CWAP_LLM_TOOL_MODE", "auto"))
    # Anthropic-only knob. Ignored (and reported as ignored) by other providers.
    llm_effort: str = field(default_factory=lambda: _env("CWAP_LLM_EFFORT", "high"))
    # Sampling knob for open models. Rejected by current Claude models, so the
    # Anthropic provider drops it rather than sending a request that would 400.
    llm_temperature: float = field(default_factory=lambda: _env_float("CWAP_LLM_TEMPERATURE", 0.7))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))

    # --- embeddings ---------------------------------------------------
    # "hashing" is the offline default. "openai_compatible" calls a /v1/embeddings
    # endpoint — the same one Ollama, vLLM, LM Studio and OpenAI all expose.
    embedding_provider: str = field(default_factory=lambda: _env("CWAP_EMBEDDING_PROVIDER", "hashing"))
    embedding_model: str = field(default_factory=lambda: _env("CWAP_EMBEDDING_MODEL", ""))
    embedding_base_url: str = field(default_factory=lambda: _env("CWAP_EMBEDDING_BASE_URL", ""))
    embedding_api_key: str = field(default_factory=lambda: _env("CWAP_EMBEDDING_API_KEY", ""))
    embedding_timeout_seconds: int = field(
        default_factory=lambda: _env_int("CWAP_EMBEDDING_TIMEOUT", 60)
    )

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
