"""Memory Management System (Epic 3, User Story 3)."""

from __future__ import annotations

import pytest
from cwap_contracts.v2 import IngestRequest, RetrievalRequest
from knowledge.embeddings import HashingEmbedder, cosine_similarity, tokenize
from knowledge.service import KnowledgeError, chunk_text, delete, ingest, list_handles, retrieve

HANDBOOK = """
Expense policy. Employees may claim meals up to 45 GBP per day when travelling.
Receipts must be submitted within 30 days of the expense being incurred.

Travel policy. Flights under four hours are booked in economy class. Rail travel
is preferred for journeys within the United Kingdom.

Equipment policy. Every engineer is issued a laptop on their first day. Requests
for a second monitor go through the facilities team.
"""


def ingest_handbook(tenant_id: str = "tenant-a", title: str = "Employee handbook"):
    return ingest(
        IngestRequest(
            tenant_id=tenant_id, title=title, content=HANDBOOK, chunk_size=200, chunk_overlap=40
        )
    )


class TestChunking:
    def test_short_text_is_a_single_chunk(self):
        assert chunk_text("hello world", chunk_size=800, overlap=100) == ["hello world"]

    def test_long_text_is_split(self):
        chunks = chunk_text("word " * 500, chunk_size=200, overlap=20)
        assert len(chunks) > 1

    def test_chunks_overlap_so_facts_are_not_lost_at_a_cut(self):
        text = " ".join(f"sentence{i}." for i in range(80))
        chunks = chunk_text(text, chunk_size=200, overlap=60)
        joined = "".join(chunks)
        assert len(joined) > len(text) - 200  # overlap means near-total coverage

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(KnowledgeError, match="must be smaller"):
            chunk_text("x" * 1000, chunk_size=100, overlap=100)

    def test_empty_text_yields_nothing(self):
        assert chunk_text("   ", chunk_size=100, overlap=10) == []


class TestEmbeddings:
    def test_embeddings_are_deterministic(self):
        embedder = HashingEmbedder()
        assert embedder.embed("expense policy") == embedder.embed("expense policy")

    def test_identical_text_scores_one(self):
        embedder = HashingEmbedder()
        vector = embedder.embed("laptop on their first day")
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_related_text_scores_above_unrelated(self):
        embedder = HashingEmbedder()
        query = embedder.embed("how much can I claim for meals")
        related = embedder.embed("Employees may claim meals up to 45 GBP per day")
        unrelated = embedder.embed("Requests for a second monitor go to facilities")
        assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)

    def test_stopwords_are_dropped(self):
        assert "the" not in tokenize("the laptop and the monitor")


class TestIngestAndRetrieve:
    def test_ingest_returns_a_usable_handle(self):
        handle = ingest_handbook()
        assert handle.handle.startswith("kb_")
        assert handle.chunk_count > 1

    def test_retrieval_finds_the_relevant_chunk(self):
        handle = ingest_handbook()
        result = retrieve(
            RetrievalRequest(
                handle=handle.handle,
                tenant_id="tenant-a",
                query="how much can I claim for meals while travelling",
                top_k=2,
            )
        )
        assert result.chunks
        assert "45 GBP" in result.as_context()

    def test_results_are_ordered_by_score(self):
        handle = ingest_handbook()
        result = retrieve(
            RetrievalRequest(
                handle=handle.handle, tenant_id="tenant-a", query="laptop equipment", top_k=5
            )
        )
        scores = [chunk.score for chunk in result.chunks]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_respected(self):
        handle = ingest_handbook()
        result = retrieve(
            RetrievalRequest(
                handle=handle.handle, tenant_id="tenant-a", query="policy", top_k=1
            )
        )
        assert len(result.chunks) <= 1

    def test_empty_document_is_rejected(self):
        with pytest.raises(Exception):
            ingest(IngestRequest(tenant_id="tenant-a", title="empty", content="   "))


class TestTenantIsolation:
    def test_another_tenant_cannot_read_a_corpus_by_handle(self):
        """A handle is an identifier, never an authorisation."""
        handle = ingest_handbook(tenant_id="tenant-a")
        with pytest.raises(KnowledgeError, match="not available to tenant"):
            retrieve(
                RetrievalRequest(
                    handle=handle.handle, tenant_id="tenant-b", query="meals", top_k=2
                )
            )

    def test_listing_is_scoped_to_the_tenant(self):
        ingest_handbook(tenant_id="tenant-a", title="A handbook")
        ingest_handbook(tenant_id="tenant-b", title="B handbook")
        assert [item.title for item in list_handles("tenant-a")] == ["A handbook"]

    def test_delete_is_scoped_to_the_tenant(self):
        handle = ingest_handbook(tenant_id="tenant-a")
        assert delete(handle.handle, "tenant-b") is False
        assert delete(handle.handle, "tenant-a") is True
        assert list_handles("tenant-a") == []
