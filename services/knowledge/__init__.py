"""Memory Management System — document ingest, embeddings, RAG retrieval."""

from knowledge.embeddings import (
    EMBEDDING_DIMENSIONS,
    Embedder,
    HashingEmbedder,
    cosine_similarity,
    get_embedder,
    set_embedder,
)
from knowledge.service import (
    KnowledgeError,
    chunk_text,
    delete,
    ingest,
    list_handles,
    retrieve,
)

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "Embedder",
    "HashingEmbedder",
    "KnowledgeError",
    "chunk_text",
    "cosine_similarity",
    "delete",
    "get_embedder",
    "ingest",
    "list_handles",
    "retrieve",
    "set_embedder",
]
