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
    # Encrypts credentials the platform stores on a tenant's behalf (provider API
    # keys, MCP server headers). Falls back to the JWT secret so a deployment
    # that already set one thing does not have to set two — but rotating it
    # invalidates stored credentials, which is why it can be separated.
    secret_key: str = field(default_factory=lambda: _env("CWAP_SECRET_KEY", ""))
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
    # Ten minutes, because the machine on the other end is frequently somebody's
    # laptop. A local reasoning model thinks for minutes before its first token,
    # and a timeout that fires mid-thought fails a whole run over a limit that
    # was only ever a guess about hardware. Nothing is lost by waiting: a worker
    # blocks on one job, and a genuinely dead endpoint fails at connect, not at
    # the timeout. A tenant can raise or lower this from Models without a restart.
    llm_timeout_seconds: int = field(default_factory=lambda: _env_int("CWAP_LLM_TIMEOUT", 600))
    # "auto" tries native tool calling and permanently downgrades to a prompted
    # JSON protocol if the server rejects it. "native" or "prompted" force one.
    llm_tool_mode: str = field(default_factory=lambda: _env("CWAP_LLM_TOOL_MODE", "auto"))
    # "auto" engages a thinking-capable model's reasoning mode and permanently
    # stops asking if the server rejects the parameter. "off" never asks.
    llm_thinking_mode: str = field(default_factory=lambda: _env("CWAP_LLM_THINKING_MODE", "auto"))
    # "auto" reads every answer as it is produced; "off" waits for the whole
    # response. Streaming is what makes the timeout a silence detector rather
    # than a ceiling on how long an answer may take, so "off" is for a proxy
    # that mangles server-sent events, not a performance choice.
    llm_stream_mode: str = field(default_factory=lambda: _env("CWAP_LLM_STREAM", "auto"))
    # Reasoning depth. Honoured by Claude and by every thinking-capable model
    # whose server takes a depth parameter; reported as ignored elsewhere.
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
    # Whether the built-in MCP directory's hosts are reachable without being in
    # the allow-list above. On by default: those endpoints are a fixed,
    # code-reviewed list, and requiring an environment variable to connect
    # GitHub made "connect any MCP server" a claim nobody could act on. Every
    # other host still needs the allow-list. Off restores one single answer.
    mcp_directory_enabled: bool = field(
        default_factory=lambda: _env_bool("CWAP_MCP_DIRECTORY", True)
    )
    # Lets a tenant point the platform at its own model endpoint from the UI.
    # Off by default: a stored base URL is a request this server will make, so
    # enabling it widens what a tenant can reach from inside the deployment.
    allow_custom_model_endpoints: bool = field(
        default_factory=lambda: _env_bool("CWAP_ALLOW_CUSTOM_MODEL_ENDPOINTS", True)
    )
    # Who may run a workflow that acts on the outside world — an HTTP node, or
    # an agent holding a tool that changes something.
    #
    #   owner     the first account in a tenant, and nobody else (the default)
    #   everyone  every account
    #   nobody    no account; the scope must be set on the user row by hand
    #
    # "owner" rather than "nobody" because the previous default was unreachable:
    # the scope was granted by nothing, no endpoint or command could grant it,
    # and the error told the user to "ask an administrator" who, on a
    # self-hosted install, is the person reading the message. Whoever registers
    # first has root on the machine the platform runs on; withholding a
    # permission from them protects nothing.
    #
    # It does not open egress. An HTTP node still needs CWAP_HTTP_ALLOWLIST, an
    # MCP host still needs the directory or the allow-list, and every external
    # call still passes the two-phase idempotency gate. This decides *who may
    # ask*, not *what may be reached*.
    write_external_grant: str = field(
        default_factory=lambda: _env("CWAP_WRITE_EXTERNAL", "owner").strip().lower()
    )

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
