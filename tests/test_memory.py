"""Memory at three scopes: workflow, agent, skill.

The behaviours worth defending are not "text goes in, text comes out". They are
the ones that decide whether a memory is an asset or a liability after fifty
runs: does it accumulate duplicates, does recall still work when the embedder is
gone, does an unhelpful lesson eventually stop being recalled, and is one
tenant's memory reachable from another's.
"""

from __future__ import annotations

import pytest
from cwap_contracts.v4 import (
    MemoryKind,
    MemoryScope,
    MemoryWriteRequest,
    RecallRequest,
)
from knowledge.embeddings import set_embedder
from memory import service as memory
from pydantic import ValidationError


def write(
    text: str,
    *,
    scope: MemoryScope = MemoryScope.AGENT,
    scope_id: str = "agt_1",
    tenant: str = "tenant-a",
    kind: MemoryKind = MemoryKind.LEARNING,
):
    return memory.remember(
        MemoryWriteRequest(
            tenant_id=tenant, scope=scope, scope_id=scope_id, kind=kind, text=text
        )
    )


def read(
    query: str,
    *,
    scope: MemoryScope = MemoryScope.AGENT,
    scope_id: str = "agt_1",
    tenant: str = "tenant-a",
    limit: int = 5,
):
    return memory.recall(
        RecallRequest(
            tenant_id=tenant, scope=scope, scope_id=scope_id, query=query, limit=limit
        )
    )


class TestRememberAndRecall:
    def test_what_was_written_can_be_recalled(self):
        write("The Denver office closes at 4pm on Fridays.")
        result = read("When does the Denver office close?")
        assert any("Denver" in entry.text for entry in result.entries)

    def test_an_empty_memory_is_unrepresentable(self):
        """Rejected at the contract, not in the service — a blank lesson can
        never reach storage from any caller."""
        with pytest.raises(ValidationError):
            write("   ")

    def test_recall_is_scoped_to_the_tenant(self):
        write("Tenant A's private pricing rule.", tenant="tenant-a")
        assert read("pricing rule", tenant="tenant-b").entries == []

    def test_recall_is_scoped_to_the_owner(self):
        """Agent memory must not leak into another agent's prompt."""
        write("Researcher lesson.", scope_id="agt_researcher")
        assert read("lesson", scope_id="agt_writer").entries == []

    def test_scopes_do_not_bleed_into_each_other(self):
        """The same id under a different scope is a different memory."""
        write("A skill-level lesson.", scope=MemoryScope.SKILL, scope_id="shared_id")
        assert read("lesson", scope=MemoryScope.AGENT, scope_id="shared_id").entries == []

    def test_recall_respects_the_limit(self):
        for index in range(10):
            write(f"Distinct lesson number {index} about scheduling.")
        assert len(read("scheduling", limit=3).entries) == 3


class TestDeduplication:
    def test_the_same_lesson_twice_is_stored_once(self):
        """Otherwise fifty runs produce fifty copies, and recall returns nothing
        else."""
        first = write("Always confirm the departure city before booking.")
        second = write("Always confirm the departure city before booking.")

        assert first.id == second.id
        assert len(memory.list_memories("tenant-a", MemoryScope.AGENT, "agt_1")) == 1

    def test_relearning_strengthens_rather_than_duplicates(self):
        first = write("Always confirm the departure city before booking.")
        second = write("Always confirm the departure city before booking.")
        assert second.usefulness > first.usefulness

    def test_a_genuinely_different_lesson_is_kept_separately(self):
        write("Always confirm the departure city.")
        write("Hotel quotes exclude local tax.")
        assert len(memory.list_memories("tenant-a", MemoryScope.AGENT, "agt_1")) == 2

    def test_dedupe_does_not_cross_kinds(self):
        """The same sentence as a failure and as a learning are different claims."""
        write("Retrieval returned nothing.", kind=MemoryKind.FAILURE)
        write("Retrieval returned nothing.", kind=MemoryKind.LEARNING)
        assert len(memory.list_memories("tenant-a", MemoryScope.AGENT, "agt_1")) == 2


class TestReinforcement:
    def test_a_good_outcome_makes_a_memory_more_trusted(self):
        entry = write("Check the corpus before answering.")
        memory.reinforce([entry.id], delta=0.5)
        [stored] = memory.list_memories("tenant-a", MemoryScope.AGENT, "agt_1")
        assert stored.usefulness == pytest.approx(entry.usefulness + 0.5)

    def test_a_bad_outcome_makes_it_less_trusted(self):
        entry = write("Check the corpus before answering.")
        memory.reinforce([entry.id], delta=-0.5)
        [stored] = memory.list_memories("tenant-a", MemoryScope.AGENT, "agt_1")
        assert stored.usefulness < entry.usefulness

    def test_a_repeatedly_unhelpful_memory_drops_out_of_recall(self):
        """This is the mechanism that stops a plausible-but-wrong lesson being
        recalled forever."""
        entry = write("Denver is in Texas.")
        memory.reinforce([entry.id], delta=-10.0)
        assert read("Where is Denver?").entries == []

    def test_usefulness_is_bounded(self):
        entry = write("A lesson.")
        memory.reinforce([entry.id], delta=100.0)
        [stored] = memory.list_memories("tenant-a", MemoryScope.AGENT, "agt_1")
        assert stored.usefulness <= 10.0

    def test_reinforcing_nothing_is_not_an_error(self):
        assert memory.reinforce([], delta=1.0) == 0


class TestRecallWithoutEmbeddings:
    def test_recall_still_works_when_the_embedder_is_gone(self):
        """A memory worth keeping is worth keeping when the embedding service is
        down; recall degrades to lexical overlap rather than returning nothing."""
        write("The quarterly review happens in March.")

        set_embedder(_BrokenEmbedder())
        try:
            result = read("When is the quarterly review?")
        finally:
            set_embedder(None)

        assert any("quarterly review" in entry.text for entry in result.entries)


class _BrokenEmbedder:
    identity = "broken"

    def embed(self, text: str):
        raise RuntimeError("embedding service unavailable")

    def embed_many(self, texts):
        raise RuntimeError("embedding service unavailable")


class TestStandingKnowledge:
    def test_proven_lessons_surface_even_without_word_overlap(self):
        """An agent's most-proven lesson is worth injecting even when it shares
        no vocabulary with today's objective — that is what makes it standing
        knowledge rather than a search index."""
        entry = write("Always state the currency alongside any figure.")
        memory.reinforce([entry.id], delta=3.0)

        result = read("Draft the onboarding email for new hires")
        assert any("currency" in item.text for item in result.entries)

    def test_relevance_hits_outrank_the_top_up(self):
        standing = write("Always state the currency alongside any figure.")
        memory.reinforce([standing.id], delta=5.0)
        write("Onboarding emails go out on the Monday before the start date.")

        result = read("When do onboarding emails go out?", limit=2)
        assert "Onboarding emails" in result.entries[0].text


class TestPruning:
    def test_a_scope_is_bounded(self):
        for index in range(12):
            write(f"Lesson {index} about a completely different topic {index}.")
        memory._prune_if_needed("tenant-a", MemoryScope.AGENT, "agt_1", capacity=5)
        assert len(memory.list_memories("tenant-a", MemoryScope.AGENT, "agt_1")) == 5

    def test_pruning_drops_the_least_useful_not_the_oldest(self):
        """An early lesson that keeps proving correct outranks yesterday's noise."""
        proven = write("Lesson zero, which has repeatedly proved correct.")
        memory.reinforce([proven.id], delta=8.0)
        for index in range(1, 12):
            write(f"Lesson {index}, unproven filler about topic {index}.")

        memory._prune_if_needed("tenant-a", MemoryScope.AGENT, "agt_1", capacity=3)

        survivors = memory.list_memories("tenant-a", MemoryScope.AGENT, "agt_1")
        assert proven.id in {entry.id for entry in survivors}


class TestForgetting:
    def test_a_scope_can_be_wiped(self):
        write("One.")
        write("Two, about something else entirely.")
        assert memory.forget("tenant-a", MemoryScope.AGENT, "agt_1") == 2
        assert read("One").entries == []

    def test_a_single_entry_can_be_removed(self):
        entry = write("A lesson a user disagrees with.")
        assert memory.forget_entry("tenant-a", entry.id) is True
        assert memory.forget_entry("tenant-a", entry.id) is False

    def test_one_tenant_cannot_delete_anothers_memory(self):
        entry = write("Tenant A's lesson.", tenant="tenant-a")
        assert memory.forget_entry("tenant-b", entry.id) is False
        assert memory.list_memories("tenant-a", MemoryScope.AGENT, "agt_1")


class TestPromptBlock:
    def test_recalled_memory_renders_as_a_labelled_block(self):
        write("The Denver office closes at 4pm on Fridays.")
        block = read("Denver office").as_prompt_block("What you have learned")
        assert "What you have learned" in block
        assert "Denver office closes" in block

    def test_an_empty_recall_renders_as_nothing(self):
        """So an agent with no memory yet gets a clean prompt rather than an
        empty heading implying it forgot something."""
        assert read("anything").as_prompt_block("What you have learned") == ""
