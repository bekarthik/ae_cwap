"""Memory — remember, recall, reinforce, prune.

The hard parts of an agent memory are not storing text. They are:

* **Not accumulating duplicates.** An agent that learns the same lesson on every
  run will, after fifty runs, recall fifty copies of it and nothing else. A
  near-identical write reinforces the existing entry instead of adding another.
* **Recall that still works when embeddings are unavailable.** The embedder can
  be a remote service that is down, or can have been swapped since a memory was
  written. Either way the memory is still worth having, so recall degrades to
  lexical overlap rather than returning nothing.
* **Bounded growth.** Memory that only grows becomes memory that only costs.
  Entries are ranked by relevance *and* accumulated usefulness, and pruning
  drops the least useful rather than the oldest.
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import datetime, timezone

from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import Memory
from cwap_contracts.v2 import (
    MemoryEntry,
    MemoryKind,
    MemoryScope,
    MemoryWriteRequest,
    RecallRequest,
    RecallResult,
)
from knowledge.embeddings import EmbeddingError, cosine_similarity, get_embedder

#: Above this cosine similarity, a new write is treated as the same lesson.
DEDUPE_THRESHOLD = 0.93

#: Entries below this drop out of recall — a memory that has proved unhelpful
#: should stop crowding out ones that have not.
MIN_USEFULNESS = 0.15

#: Ceiling per scope before pruning. Generous enough that normal use never hits
#: it, low enough that a runaway loop cannot fill the table.
DEFAULT_CAPACITY = 500

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class MemoryError_(RuntimeError):
    """Memory could not be written or read."""


def remember(request: MemoryWriteRequest) -> MemoryEntry:
    """Record something learned, or reinforce it if already known."""
    text = request.text.strip()
    if not text:
        raise MemoryError_("refusing to store an empty memory")

    embedding, identity = _embed(text)

    existing = _find_duplicate(request, embedding, text)
    if existing is not None:
        # Same lesson learned again: strengthen it rather than storing it twice.
        with unit_of_work() as session:
            row = session.get(Memory, existing)
            row.usefulness = min(10.0, row.usefulness + 0.5)
            return _to_entry(row)

    entry_id = f"mem_{uuid.uuid4().hex[:20]}"
    with unit_of_work() as session:
        row = Memory(
            id=entry_id,
            tenant_id=request.tenant_id,
            scope=request.scope.value,
            scope_id=request.scope_id,
            kind=request.kind.value,
            text=text,
            embedding=embedding,
            embedding_model=identity,
            source_run_id=request.source_run_id,
            usefulness=1.0,
        )
        session.add(row)
        session.flush()
        entry = _to_entry(row)

    _prune_if_needed(request.tenant_id, request.scope, request.scope_id)
    return entry


def recall(request: RecallRequest) -> RecallResult:
    """The most relevant things this scope knows, best first."""
    query_embedding, identity = _embed(request.query)
    query_tokens = _tokens(request.query)

    with read_only_session() as session:
        rows = (
            session.query(Memory)
            .filter_by(
                tenant_id=request.tenant_id,
                scope=request.scope.value,
                scope_id=request.scope_id,
            )
            .all()
        )
        candidates = [
            (
                row.id,
                row.text,
                row.kind,
                row.embedding,
                row.embedding_model,
                row.usefulness,
                row.source_run_id,
                row.created_at,
            )
            for row in rows
        ]

    scored: list[tuple[float, tuple]] = []
    for candidate in candidates:
        usefulness = candidate[5]
        if usefulness < MIN_USEFULNESS:
            continue

        relevance = _relevance(
            query_embedding=query_embedding,
            query_identity=identity,
            query_tokens=query_tokens,
            entry_embedding=candidate[3],
            entry_identity=candidate[4],
            entry_text=candidate[1],
        )
        if relevance <= 0.0:
            continue

        # Usefulness is a gentle multiplier, not a replacement for relevance: a
        # very useful memory about something else is still the wrong memory.
        scored.append((relevance * (1.0 + math.log1p(usefulness)), candidate))

    scored.sort(key=lambda item: -item[0])

    # Top up with standing knowledge when relevance matched little or nothing.
    #
    # A lexical embedder scores zero against any query that shares no words, so
    # without this an agent would recall nothing on most turns. But an agent's
    # most-proven lessons ("always confirm the departure city") are worth
    # injecting even when they do not match today's phrasing — that is exactly
    # what standing knowledge is. Relevance hits always rank above the top-up.
    if len(scored) < request.limit:
        matched = {candidate[0] for _score, candidate in scored}
        standing = sorted(
            (c for c in candidates if c[0] not in matched and c[5] >= MIN_USEFULNESS),
            key=lambda c: (-c[5], c[7] is None, c[7]),
        )
        scored.extend((0.0, candidate) for candidate in standing[: request.limit - len(scored)])

    entries = [
        MemoryEntry(
            id=candidate[0],
            tenant_id=request.tenant_id,
            scope=request.scope,
            scope_id=request.scope_id,
            kind=MemoryKind(candidate[2]),
            text=candidate[1],
            source_run_id=candidate[6],
            usefulness=candidate[5],
            created_at=candidate[7],
        )
        for _score, candidate in scored[: request.limit]
    ]
    return RecallResult(scope=request.scope, scope_id=request.scope_id, entries=entries)


def reinforce(entry_ids: list[str], *, delta: float) -> int:
    """Adjust usefulness after an outcome.

    Called with a positive delta when recalled memories preceded a good run and a
    negative one when they preceded a failure. This is what stops a plausible but
    wrong lesson from being recalled forever.
    """
    if not entry_ids:
        return 0
    with unit_of_work() as session:
        rows = session.query(Memory).filter(Memory.id.in_(entry_ids)).all()
        for row in rows:
            row.usefulness = max(0.0, min(10.0, row.usefulness + delta))
        return len(rows)


def list_memories(
    tenant_id: str, scope: MemoryScope, scope_id: str, limit: int = 100
) -> list[MemoryEntry]:
    with read_only_session() as session:
        rows = (
            session.query(Memory)
            .filter_by(tenant_id=tenant_id, scope=scope.value, scope_id=scope_id)
            .order_by(Memory.usefulness.desc(), Memory.created_at.desc())
            .limit(limit)
            .all()
        )
        return [_to_entry(row) for row in rows]


def forget(tenant_id: str, scope: MemoryScope, scope_id: str) -> int:
    """Wipe a scope. Used when a run's working memory is finished, and available
    to a user who wants an agent to start over."""
    with unit_of_work() as session:
        return (
            session.query(Memory)
            .filter_by(tenant_id=tenant_id, scope=scope.value, scope_id=scope_id)
            .delete()
        )


def forget_entry(tenant_id: str, entry_id: str) -> bool:
    with unit_of_work() as session:
        deleted = (
            session.query(Memory).filter_by(tenant_id=tenant_id, id=entry_id).delete()
        )
        return bool(deleted)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _embed(text: str) -> tuple[list[float] | None, str]:
    """Embed, tolerating an unavailable embedder.

    A memory worth keeping is worth keeping even when the embedding service is
    down; it simply falls back to lexical recall until it is rewritten.
    """
    try:
        embedder = get_embedder()
        return embedder.embed(text), embedder.identity
    except (EmbeddingError, Exception):  # noqa: B014 - EmbeddingError is a subclass
        return None, ""


def _relevance(
    *,
    query_embedding: list[float] | None,
    query_identity: str,
    query_tokens: set[str],
    entry_embedding: list[float] | None,
    entry_identity: str,
    entry_text: str,
) -> float:
    """Semantic where both sides share an embedding space, lexical otherwise."""
    if (
        query_embedding
        and entry_embedding
        and query_identity
        and query_identity == entry_identity
        and len(query_embedding) == len(entry_embedding)
    ):
        return cosine_similarity(query_embedding, entry_embedding)
    return _lexical_overlap(query_tokens, _tokens(entry_text))


def _lexical_overlap(left: set[str], right: set[str]) -> float:
    """Jaccard overlap. Crude, but it keeps recall working rather than silently
    returning nothing when embeddings are unavailable."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _find_duplicate(
    request: MemoryWriteRequest, embedding: list[float] | None, text: str
) -> str | None:
    """The id of an existing entry that says the same thing, if there is one."""
    with read_only_session() as session:
        rows = (
            session.query(Memory)
            .filter_by(
                tenant_id=request.tenant_id,
                scope=request.scope.value,
                scope_id=request.scope_id,
                kind=request.kind.value,
            )
            .all()
        )
        candidates = [(row.id, row.text, row.embedding, row.embedding_model) for row in rows]

    normalized = text.strip().lower()
    query_tokens = _tokens(text)
    embedder_identity = get_embedder().identity if embedding else ""

    for entry_id, entry_text, entry_embedding, entry_identity in candidates:
        if entry_text.strip().lower() == normalized:
            return entry_id
        similarity = _relevance(
            query_embedding=embedding,
            query_identity=embedder_identity,
            query_tokens=query_tokens,
            entry_embedding=entry_embedding,
            entry_identity=entry_identity,
            entry_text=entry_text,
        )
        if similarity >= DEDUPE_THRESHOLD:
            return entry_id
    return None


def _prune_if_needed(
    tenant_id: str, scope: MemoryScope, scope_id: str, capacity: int = DEFAULT_CAPACITY
) -> int:
    """Drop the least useful entries once a scope exceeds its ceiling.

    Least *useful*, not oldest: an early lesson that keeps proving correct is
    worth more than yesterday's noise.
    """
    with unit_of_work() as session:
        total = (
            session.query(Memory)
            .filter_by(tenant_id=tenant_id, scope=scope.value, scope_id=scope_id)
            .count()
        )
        if total <= capacity:
            return 0

        doomed = (
            session.query(Memory.id)
            .filter_by(tenant_id=tenant_id, scope=scope.value, scope_id=scope_id)
            .order_by(Memory.usefulness.asc(), Memory.created_at.asc())
            .limit(total - capacity)
            .all()
        )
        ids = [row[0] for row in doomed]
        session.query(Memory).filter(Memory.id.in_(ids)).delete(synchronize_session=False)
        return len(ids)


def _to_entry(row: Memory) -> MemoryEntry:
    return MemoryEntry(
        id=row.id,
        tenant_id=row.tenant_id,
        scope=MemoryScope(row.scope),
        scope_id=row.scope_id,
        kind=MemoryKind(row.kind),
        text=row.text,
        source_run_id=row.source_run_id,
        usefulness=row.usefulness,
        created_at=row.created_at or datetime.now(timezone.utc),
    )
