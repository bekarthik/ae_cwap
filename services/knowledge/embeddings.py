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
    """
    a an and are as at be but by for from has have how i if in is it its of on or
    that the this to was were what when where which who will with you your
    """.split()
)


class Embedder(Protocol):
    dimensions: int

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
    """Deterministic, dependency-free embeddings."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

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
    score = sum(a * b for a, b in zip(left, right))
    return max(0.0, min(1.0, score))


_embedder: Embedder = HashingEmbedder()


def get_embedder() -> Embedder:
    return _embedder


def set_embedder(embedder: Embedder) -> None:
    global _embedder
    _embedder = embedder
