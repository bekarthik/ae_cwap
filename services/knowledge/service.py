"""Knowledge Indexer & RAG retrieval (Epic 3).

User Story 3: link a knowledge bank to a workflow node so the agent's answers
are grounded in private data rather than general training data.

Ingest:   document text -> chunks -> embeddings -> vector store -> handle
Retrieve: handle + query -> top-K chunks -> context payload for an LLM node

Tenant isolation is enforced on *every* read. A `knowledge_handle` saved in a
graph is just a string, and a graph can be copied between tenants, so the handle
alone is never treated as authorisation.
"""

from __future__ import annotations

import re
import uuid

from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import Chunk, Document
from cwap_contracts.v2 import (
    IngestRequest,
    KnowledgeHandle,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)

from knowledge.embeddings import (
    EmbeddingError,
    cosine_similarity,
    embed_many,
    get_embedder,
)

#: Prefer to break a chunk at a paragraph or sentence boundary rather than
#: mid-word, so retrieved context reads as prose.
_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class KnowledgeError(RuntimeError):
    """Ingest or retrieval failed in a way the caller must handle."""


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split into overlapping windows, snapping to sentence boundaries.

    Overlap matters for retrieval quality: a fact that straddles a hard cut is
    otherwise unfindable, because neither chunk contains the whole statement.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if overlap >= chunk_size:
        raise KnowledgeError(
            f"chunk_overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]

        if end < len(text):
            # Snap back to the last sentence/paragraph break in the final third
            # of the window; ignore breaks too early to be worth the lost text.
            boundaries = [match.start() for match in _BOUNDARY_RE.finditer(window)]
            usable = [pos for pos in boundaries if pos > chunk_size * 0.6]
            if usable:
                end = start + usable[-1]
                window = text[start:end]

        cleaned = window.strip()
        if cleaned:
            chunks.append(cleaned)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def ingest(request: IngestRequest) -> KnowledgeHandle:
    """Index a document and return the handle a canvas node will reference."""
    chunks = chunk_text(
        request.content, chunk_size=request.chunk_size, overlap=request.chunk_overlap
    )
    if not chunks:
        raise KnowledgeError("document produced no chunks; is it empty?")

    embedder = get_embedder()
    handle = f"kb_{uuid.uuid4().hex[:16]}"

    # Embed outside the transaction: a remote embedding model can take seconds
    # per batch, and holding a write transaction open across the network is how
    # you get lock contention under concurrent uploads.
    try:
        vectors = embed_many(embedder, chunks)
    except EmbeddingError as exc:
        raise KnowledgeError(f"could not index '{request.title}': {exc}") from exc

    # One transaction for the document row and every chunk: a partially indexed
    # corpus would silently return incomplete answers forever.
    with unit_of_work() as session:
        session.add(
            Document(
                handle=handle,
                tenant_id=request.tenant_id,
                title=request.title,
                chunk_count=len(chunks),
                embedding_model=embedder.identity,
            )
        )
        session.flush()
        for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            session.add(
                Chunk(
                    id=f"ch_{uuid.uuid4().hex[:16]}",
                    handle=handle,
                    tenant_id=request.tenant_id,
                    ordinal=ordinal,
                    text=chunk,
                    embedding=vector,
                )
            )

    return KnowledgeHandle(
        handle=handle,
        tenant_id=request.tenant_id,
        title=request.title,
        chunk_count=len(chunks),
    )


def retrieve(request: RetrievalRequest) -> RetrievalResult:
    """Top-K semantic search over one corpus, scoped to one tenant."""
    embedder = get_embedder()

    with read_only_session() as session:
        document = (
            session.query(Document)
            .filter_by(handle=request.handle, tenant_id=request.tenant_id)
            .one_or_none()
        )
        if document is None:
            raise KnowledgeError(
                f"knowledge handle '{request.handle}' is not available to tenant "
                f"'{request.tenant_id}'"
            )
        indexed_with = document.embedding_model or ""
        rows = (
            session.query(Chunk)
            .filter_by(handle=request.handle, tenant_id=request.tenant_id)
            .all()
        )
        title = document.title
        candidates = [(row.id, row.text, row.embedding, row.ordinal) for row in rows]

    # Comparing vectors from two different embedding models produces scores that
    # look plausible and mean nothing. Refuse, and say exactly what to do.
    if indexed_with and indexed_with != embedder.identity:
        raise KnowledgeError(
            f"'{title}' was indexed with '{indexed_with}' but the platform is now "
            f"configured for '{embedder.identity}'. Re-upload the document, or set "
            "the embedding configuration back."
        )

    try:
        query_vector = embedder.embed(request.query)
    except EmbeddingError as exc:
        raise KnowledgeError(f"could not embed the query: {exc}") from exc

    scored = [
        (cosine_similarity(query_vector, embedding), chunk_id, text, ordinal)
        for chunk_id, text, embedding, ordinal in candidates
    ]
    # Sort by score, then by ordinal so equal scores are stable and readable.
    scored.sort(key=lambda item: (-item[0], item[3]))

    chunks = [
        RetrievedChunk(chunk_id=chunk_id, text=text, score=round(score, 6), source_title=title)
        for score, chunk_id, text, _ in scored[: request.top_k]
        if score > 0.0
    ]
    return RetrievalResult(handle=request.handle, query=request.query, chunks=chunks)


def list_handles(tenant_id: str) -> list[KnowledgeHandle]:
    with read_only_session() as session:
        documents = (
            session.query(Document)
            .filter_by(tenant_id=tenant_id)
            .order_by(Document.created_at.desc())
            .all()
        )
        return [
            KnowledgeHandle(
                handle=doc.handle,
                tenant_id=doc.tenant_id,
                title=doc.title,
                chunk_count=doc.chunk_count,
                created_at=doc.created_at,
            )
            for doc in documents
        ]


def delete(handle: str, tenant_id: str) -> bool:
    with unit_of_work() as session:
        document = (
            session.query(Document).filter_by(handle=handle, tenant_id=tenant_id).one_or_none()
        )
        if document is None:
            return False
        session.query(Chunk).filter_by(handle=handle, tenant_id=tenant_id).delete()
        session.delete(document)
        return True
