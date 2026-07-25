"""Embedding backends.

The default is a deterministic hashing embedder: no network, no API key, no
model download. It is genuinely useful for lexical retrieval (it is a hashed
bag-of-words with sublinear term weighting, i.e. cosine over a hashing
vectoriser) and it makes the RAG tests assert on exact scores.

Swapping in a real embedding model is a change to this file only — the vector
store and the retrieval API are written against the `Embedder` protocol.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

#: Small enough to keep JSON-column storage cheap, wide enough that hash
#: collisions between distinct terms stay rare for document-sized corpora.
EMBEDDING_DIMENSIONS = 512

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Extremely common words carry no retrieval signal and would otherwise dominate
#: the cosine score of every chunk.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "how", "i", "if", "in", "is", "it", "its", "of", "on",
        "or", "that", "the", "this", "to", "was", "were", "what", "when",
        "where", "which", "who", "will", "with", "you", "your",
    }
)


class Embedder(Protocol):
    dimensions: int
    #: Stamped onto every indexed corpus. Retrieval refuses to compare vectors
    #: produced by different models, which would otherwise return confident
    #: nonsense after someone changes `CWAP_EMBEDDING_MODEL`.
    identity: str

    def embed(self, text: str) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    return [
        stem(token)
        for token in _TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS
    ]


def stem(token: str) -> str:
    """Crude suffix stripping so "meals" and "meal" land in the same bucket.

    A hashing embedder has no notion of morphology, so without this a user
    searching for "meal allowance" scores zero against a chunk that says
    "meals". This is not linguistics — it is the minimum that stops obvious
    misses. Swapping `HashingEmbedder` for a real embedding model makes it
    irrelevant, which is the intended production path.
    """
    for suffix, minimum in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if token.endswith(suffix) and len(token) >= minimum:
            trimmed = token[: -len(suffix)]
            # Never strip into a double consonant ("ss" -> "s") or a stub.
            if len(trimmed) >= 3 and not trimmed.endswith(suffix[-1]):
                return trimmed
    return token


class HashingEmbedder:
    """Deterministic, dependency-free embeddings.

    Lexical, not semantic: it matches wording rather than meaning. That is a
    deliberate default — it needs no model, no credentials and no network, so
    the platform demos and tests offline — but a deployment that cares about
    retrieval quality should point `CWAP_EMBEDDING_PROVIDER` at a real model.
    """

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.identity = f"hashing:{dimensions}"

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            # Sublinear scaling: a term repeated 10 times is more relevant than
            # one seen once, but not 10x more.
            vector[self._bucket(token)] += 1.0 + math.log(count)

        return normalize(vector)


def normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return vector
    return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Both inputs are unit vectors, so this is a plain dot product.

    Clamped to [0, 1] because the contract declares that range and floating
    point can otherwise nudge a perfect match to 1.0000000000000002.
    """
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} vs {len(right)}")
    score = sum(a * b for a, b in zip(left, right, strict=True))
    return max(0.0, min(1.0, score))


class EmbeddingError(RuntimeError):
    """The embedding backend could not be reached or is misconfigured."""


class RemoteEmbedder:
    """Embeddings from any server exposing `/embeddings`.

    The same endpoint shape is served by Ollama, vLLM, LM Studio, LocalAI,
    OpenAI, Together and Mistral, so one client covers open-weight models running
    on a laptop and hosted APIs alike.

    Dimensionality is discovered from the first response rather than configured,
    because it is a property of the model and getting it wrong silently corrupts
    a corpus.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: int = 60,
        provider: str = "openai_compatible",
    ) -> None:
        if not base_url:
            raise EmbeddingError(
                "embedding provider needs a base URL; set CWAP_EMBEDDING_BASE_URL "
                "(for example http://localhost:11434/v1 for Ollama)"
            )
        if not model:
            raise EmbeddingError(
                "embedding provider needs a model id; set CWAP_EMBEDDING_MODEL "
                "(for example nomic-embed-text)"
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self.identity = f"{provider}:{model}"
        #: Unknown until the first call answers.
        self.dimensions = 0

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several chunks in one request.

        Indexing a document is the hot path — one HTTP round trip per chunk
        would make a modest upload take minutes against a local server.
        """
        import httpx  # noqa: PLC0415 - already a dependency

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/embeddings",
                    json={"model": self._model, "input": texts},
                    headers=headers,
                )
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                f"could not reach the embedding server at {self._base_url} ({exc})"
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                f"embedding request timed out after {self._timeout}s; "
                "raise CWAP_EMBEDDING_TIMEOUT or index smaller documents"
            ) from exc

        if response.status_code >= 400:
            raise EmbeddingError(
                f"embedding server returned {response.status_code}: {response.text[:300]}"
            )

        try:
            payload = response.json()["data"]
            # Some servers return results out of order; `index` is authoritative.
            ordered = sorted(payload, key=lambda item: item.get("index", 0))
            vectors = [normalize([float(value) for value in item["embedding"]]) for item in ordered]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError(
                f"embedding server returned an unexpected shape: {response.text[:300]}"
            ) from exc

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"asked for {len(texts)} embeddings and got {len(vectors)} back"
            )

        self.dimensions = len(vectors[0]) if vectors else 0
        return vectors


def build_embedder(settings=None) -> Embedder:
    """Construct the embedder named by configuration."""
    from cwap_common.settings import get_settings  # noqa: PLC0415 - avoid import cycle

    settings = settings or get_settings()
    provider = settings.embedding_provider.strip().lower()

    if provider in {"hashing", "", "stub", "local"}:
        return HashingEmbedder()

    from llm_proxy.presets import resolve  # noqa: PLC0415

    preset = resolve(provider)
    base_url = settings.embedding_base_url or (preset.base_url if preset else "")
    model = settings.embedding_model or (preset.default_embedding_model if preset else "")
    api_key = settings.embedding_api_key or settings.llm_api_key

    return RemoteEmbedder(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=settings.embedding_timeout_seconds,
        provider=provider,
    )


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-wide embedder, built from configuration on first use."""
    global _embedder
    if _embedder is None:
        _embedder = build_embedder()
    return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """Swap the embedder. Used by tests and by the dev runner."""
    global _embedder
    _embedder = embedder


def embed_many(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Embed a batch, using the backend's batch endpoint when it has one."""
    batch = getattr(embedder, "embed_batch", None)
    if callable(batch):
        return batch(texts)
    return [embedder.embed(text) for text in texts]


def describe_embedder() -> dict[str, object]:
    """Embedding configuration, for the API and the canvas."""
    from cwap_common.settings import get_settings  # noqa: PLC0415

    settings = get_settings()
    try:
        embedder = get_embedder()
    except EmbeddingError as exc:
        return {
            "configured": False,
            "provider": settings.embedding_provider,
            "error": str(exc),
        }
    return {
        "configured": True,
        "provider": settings.embedding_provider,
        "identity": embedder.identity,
        "dimensions": embedder.dimensions,
        "semantic": not embedder.identity.startswith("hashing:"),
    }
