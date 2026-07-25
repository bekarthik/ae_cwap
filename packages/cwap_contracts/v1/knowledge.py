"""Epic 3 contracts — the Memory Management System."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from cwap_contracts.v1.base import ContractModel, utcnow


class IngestRequest(ContractModel):
    """A document (already decoded to text) offered to the indexer."""

    tenant_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1)
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)


class KnowledgeHandle(ContractModel):
    """The handle a canvas node stores to point at an indexed corpus."""

    handle: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    chunk_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)


class RetrievalRequest(ContractModel):
    handle: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=50)


class RetrievedChunk(ContractModel):
    chunk_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    source_title: str = ""


class RetrievalResult(ContractModel):
    handle: str
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)

    def as_context(self) -> str:
        """Flatten to the text block that gets injected into an LLM prompt."""
        return "\n\n".join(f"[{c.source_title}] {c.text}" for c in self.chunks)
