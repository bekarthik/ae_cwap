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
    DateTime,
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
