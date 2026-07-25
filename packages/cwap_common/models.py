"""Relational schema.

PostgreSQL is the production target (JSONB for graph payloads); SQLite is used
for local dev and tests. Every JSON column is declared with a `JSONB` variant so
the same models work on both without a second schema definition.

The table that matters most is `workflow_execution_state`: its
`UNIQUE(run_id, step_execution_id, source_service)` constraint is the database-level
guarantee behind mandate §3.B — however many times a worker retries step X of
run Y, only the first successful write commits.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: JSON on SQLite, JSONB on PostgreSQL.
JsonCol = JSON().with_variant(JSONB(), "postgresql")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    """Auth Service storage. Passwords are stored as salted PBKDF2 digests."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(JsonCol, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Workflow(Base):
    """State Management DB — the saved graph, verbatim, as JSON."""

    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    owner_id: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    graph: Mapped[dict[str, Any]] = mapped_column(JsonCol, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Run(Base):
    """One execution instance of a workflow."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    initiating_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    graph_snapshot: Mapped[dict[str, Any]] = mapped_column(JsonCol, nullable=False)
    inputs: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict, nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JsonCol, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    steps_executed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WorkflowExecutionState(Base):
    """Mandate §3.B — atomic, non-destructive, first-write-wins node state.

    A worker writes here with `ON CONFLICT DO NOTHING`. If the row already
    exists, the retry is a no-op and the original output stands, which is what
    makes replay safe.
    """

    __tablename__ = "workflow_execution_state"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "step_execution_id", "source_service", name="uq_execution_state_step"
        ),
        Index("ix_execution_state_run", "run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    step_execution_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_service: Mapped[str] = mapped_column(String(64), nullable=False)
    node_id: Mapped[str] = mapped_column(String(128), nullable=False)
    resulting_state: Mapped[str] = mapped_column(String(64), nullable=False)
    output: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict, nullable=False)
    derived_context: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class IdempotencyLedger(Base):
    """Mandate §3.A — the pre-check gate for external side effects.

    Before a worker calls a third-party system it claims a row here keyed by
    (run_id, step_execution_id, operation). A claim that already exists in state
    `SUCCEEDED` short-circuits the call entirely.
    """

    __tablename__ = "idempotency_ledger"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "step_execution_id", "operation", name="uq_idempotency_operation"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    step_execution_id: Mapped[str] = mapped_column(String(64), nullable=False)
    operation: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="IN_FLIGHT", nullable=False)
    external_ref: Mapped[str | None] = mapped_column(String(256), nullable=True)
    response: Mapped[dict[str, Any] | None] = mapped_column(JsonCol, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class DeadLetter(Base):
    """Mandate §2.C — observable parking for contract/authorization failures."""

    __tablename__ = "dead_letters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    queue: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    reason_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JsonCol, nullable=True)
    resubmitted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RunLog(Base):
    """Persisted copy of the streamed log feed, so a run report survives a
    browser refresh (and an auditor asking three weeks later)."""

    __tablename__ = "run_logs"
    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_run_log_seq"),
        Index("ix_run_logs_run", "run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    node: Mapped[str | None] = mapped_column(String(128), nullable=True)
    level: Mapped[str] = mapped_column(String(16), default="INFO", nullable=False)
    event: Mapped[str] = mapped_column(String(128), nullable=False)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict, nullable=False)


class Document(Base):
    """A corpus registered with the Knowledge Indexer."""

    __tablename__ = "documents"

    handle: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Which embedder produced this corpus's vectors. Retrieval refuses to
    # compare across models: cosine distance between vectors from two different
    # embedding spaces is meaningless, and would return confident nonsense.
    embedding_model: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Chunk(Base):
    """One embedded slice of a document.

    The embedding lives in a JSON column here because the reference deployment
    ships without an external vector database; `knowledge.vectorstore` swaps this
    for pgvector/Pinecone behind the same interface.
    """

    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunks_handle", "handle"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    handle: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.handle", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(JsonCol, nullable=False)


class Skill(Base):
    """A capability an agent can invoke.

    Stored declaratively — `definition` holds a prompt template, a corpus handle,
    an allow-listed URL, or a list of other skills, never code. That is what
    makes it safe for the platform to synthesise skills it does not have.
    """

    __tablename__ = "skills"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_skill_name_per_tenant"),
        Index("ix_skills_tenant", "tenant_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters: Mapped[list[dict[str, Any]]] = mapped_column(JsonCol, default=list, nullable=False)
    definition: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict, nullable=False)
    origin: Mapped[str] = mapped_column(String(32), default="user", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    # Usage counters, so a synthesised skill that never works is visible rather
    # than quietly re-selected forever.
    invocations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Agent(Base):
    """A reusable role with skills and a loop.

    Referenced by canvas nodes rather than embedded in them, so improving an
    agent improves every workflow that uses it.
    """

    __tablename__ = "agents"
    __table_args__ = (Index("ix_agents_tenant", "tenant_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    objective: Mapped[str] = mapped_column(Text, default="", nullable=False)
    instructions: Mapped[str] = mapped_column(Text, default="", nullable=False)
    skill_ids: Mapped[list[str]] = mapped_column(JsonCol, default=list, nullable=False)
    max_iterations: Mapped[int] = mapped_column(Integer, default=6, nullable=False)
    memory_config: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict, nullable=False)
    model_override: Mapped[str | None] = mapped_column(String(128), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Memory(Base):
    """One remembered thing, at whatever scope owns it.

    A single table rather than three, because the read pattern is identical —
    "give me what this scope knows about X" — and the scopes differ only in what
    `scope_id` points at. `embedding` is nullable so a memory written while the
    embedder is unavailable is still kept; it falls back to lexical recall.
    """

    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memories_scope", "scope", "scope_id"),
        Index("ix_memories_tenant", "tenant_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    scope_id: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), default="learning", nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(JsonCol, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    source_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    usefulness: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ModelSetting(Base):
    """A tenant's chosen model backend, so it can be changed without a restart.

    Environment variables remain the deployment default and the fallback; a row
    here is a tenant saying "not that one, this one". Keeping it per tenant
    rather than global is what stops one tenant's choice — or one tenant's bad
    API key — from changing what everyone else's workflows run on.

    `api_key` holds ciphertext (see `cwap_common.secrets`) and is never returned
    by the API. `kind` separates the chat model from the embedding model, which
    are frequently different services.
    """

    __tablename__ = "model_settings"
    __table_args__ = (
        UniqueConstraint("tenant_id", "kind", name="uq_model_setting_tenant_kind"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    #: "llm" or "embedding".
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(256), default="", nullable=False)
    base_url: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    api_key: Mapped[str] = mapped_column(Text, default="", nullable=False)
    #: How long to wait for a reply, in seconds. 0 means the deployment default.
    #: Stored per tenant because it is a property of *their* hardware: a hosted
    #: API answers in seconds, a 70B reasoning model on a laptop does not.
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_by: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class MCPServer(Base):
    """An MCP server a tenant has connected, and how to reach it.

    Transport is either `stdio` (the platform launches a process) or `http`
    (streamable HTTP to a URL). Those have very different blast radii, which is
    why `mcp.client` gates them separately: a stdio server is code execution on
    the worker and must be allow-listed by an operator, while an HTTP server is
    egress and goes through the host allow-list.

    `config` holds the transport's own fields — command and args, or url — and
    `credentials` holds anything secret (auth headers, tokens, env values),
    encrypted at rest and never returned.
    """

    __tablename__ = "mcp_servers"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_mcp_server_tenant_name"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    transport: Mapped[str] = mapped_column(String(16), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JsonCol, default=dict, nullable=False)
    credentials: Mapped[str] = mapped_column(Text, default="", nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    #: Cached from the last successful `tools/list`, so the UI can show what a
    #: server offers without reconnecting on every page load.
    tools: Mapped[list[dict[str, Any]]] = mapped_column(JsonCol, default=list, nullable=False)
    last_connected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
