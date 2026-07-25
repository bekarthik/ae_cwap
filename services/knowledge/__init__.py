"""Memory Management System — document ingest, embeddings, RAG retrieval.

Embeddings come from whatever `CWAP_EMBEDDING_PROVIDER` names: the offline
hashing default, or any server exposing an OpenAI-compatible `/embeddings`
endpoint (Ollama, vLLM, LM Studio, LocalAI, OpenAI, Together, Mistral).
"""

from knowledge.embeddings import (
    EMBEDDING_DIMENSIONS,
    Embedder,
    EmbeddingError,
    HashingEmbedder,
    RemoteEmbedder,
    build_embedder,
    cosine_similarity,
    describe_embedder,
    embed_many,
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
    "EmbeddingError",
    "HashingEmbedder",
    "KnowledgeError",
    "RemoteEmbedder",
    "build_embedder",
    "chunk_text",
    "cosine_similarity",
    "delete",
    "describe_embedder",
    "embed_many",
    "get_embedder",
    "ingest",
    "list_handles",
    "retrieve",
    "set_embedder",
]
