"""The agent loop.

What distinguishes an agent from a step is the loop: it is given an objective and
a set of skills, decides which to use, sees the result, and decides again. These
tests script the model's decisions so that loop is observable — a stub that never
requests a tool would exercise exactly one iteration and prove nothing.

The reflection half matters as much as the acting half. An agent that does not
write down what it learned starts every run from zero, and per-agent and
per-skill memory is what makes the second run better than the first.
"""

from __future__ import annotations

import pytest
from agents import registry as agent_registry
from agents import runtime
from cwap_contracts.v2 import MemoryScope, SkillKind, SkillParameter, SkillProposal
from llm_proxy.client import (
    LLMCompletion,
    LLMProxyError,
    LLMRefusal,
    ProviderCapabilities,
    ToolCallRequest,
    reset_provider_cache,
)
from memory import service as memory_service
from skills import registry as skill_registry

TENANT = "tenant-a"


class ScriptedAgentProvider:
    """A model whose every decision is written down in advance.

    Each script entry is either a string (a final answer) or a list of
    `(skill_name, arguments)` pairs (tool calls to request).
    """

    def __init__(self, *script) -> None:
        self._script = list(script)
        self.conversations: list[list] = []
        self.systems: list[str] = []
        self.tools_offered: list[list] = []
        self.capabilities = ProviderCapabilities(
            provider="scripted",
            label="Scripted",
            model="scripted-model",
            supports_effort=True,
            supports_temperature=True,
            supports_top_p=True,
            supports_stop_sequences=True,
            supports_system_prompt=True,
        )

    def complete(self, prompt, *, system=None, options=None):
        return LLMCompletion(text="completed", model="scripted-model")

    def converse(self, messages, *, system=None, tools=None, options=None):
        self.conversations.append(list(messages))
        self.systems.append(system or "")
        self.tools_offered.append([tool["name"] for tool in (tools or [])])

        step = self._script.pop(0) if self._script else "I have what I need."
        if isinstance(step, str):
            return LLMCompletion(text=step, model="scripted-model", output_tokens=5)

        calls = tuple(
            ToolCallRequest(id=f"call_{index}", name=name, arguments=arguments)
            for index, (name, arguments) in enumerate(step)
        )
        return LLMCompletion(
            text="Let me use a capability.",
            model="scripted-model",
            tool_calls=calls,
            output_tokens=5,
        )


class FailingProvider:
    capabilities = ProviderCapabilities(
        provider="failing",
        label="Failing",
        model="none",
        supports_effort=False,
        supports_temperature=False,
        supports_top_p=False,
        supports_stop_sequences=False,
        supports_system_prompt=True,
    )

    def __init__(self, error) -> None:
        self._error = error

    def complete(self, prompt, *, system=None, options=None):
        raise self._error

    def converse(self, messages, *, system=None, tools=None, options=None):
        raise self._error


@pytest.fixture
def skills():
    return {skill.name: skill for skill in skill_registry.ensure_builtins(TENANT)}


@pytest.fixture
def researcher(skills):
    return agent_registry.create(
        TENANT,
        name="Researcher",
        role="a careful researcher",
        skill_ids=[skills["summarise"].id, skills["critique"].id],
        max_iterations=3,
    )


def context(**overrides) -> runtime.AgentRunContext:
    base = dict(tenant_id=TENANT, run_id="run_1", step_execution_id="se_1")
    base.update(overrides)
    return runtime.AgentRunContext(**base)


class TestTheLoop:
    def test_an_agent_that_needs_nothing_answers_in_one_iteration(self, researcher):
        reset_provider_cache(ScriptedAgentProvider("Denver is in Colorado."))
        result = runtime.run_agent(researcher, "Where is Denver?", context())

        assert result.text == "Denver is in Colorado."
        assert result.status == runtime.OBJECTIVE_MET
        assert result.iterations == 1

    def test_an_agent_uses_a_skill_then_answers(self, researcher):
        provider = ScriptedAgentProvider(
            [("summarise", {"text": "a long document"})],
            "Here is what the document says.",
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(researcher, "Summarise the document", context())

        assert result.skills_used == ["summarise"]
        assert result.iterations == 2
        assert result.status == runtime.OBJECTIVE_MET

    def test_the_skill_result_is_fed_back_to_the_model(self, researcher):
        """Without this it is not a loop — it is two unrelated calls."""
        provider = ScriptedAgentProvider(
            [("format_output", {"content": "THE CONDENSED TEXT"})], "Done."
        )
        skill = skill_registry.find_by_name(TENANT, "format_output")
        researcher = agent_registry.update(TENANT, researcher.id, skill_ids=[skill.id])
        reset_provider_cache(provider)

        runtime.run_agent(researcher, "Shape the text", context())

        second_turn = provider.conversations[1]
        assert any(
            message.role == "tool" and "THE CONDENSED TEXT" in message.content
            for message in second_turn
        )

    def test_an_agent_can_chain_several_skills(self, researcher):
        provider = ScriptedAgentProvider(
            [("summarise", {"text": "a document"})],
            [("critique", {"work": "the summary", "objective": "be accurate"})],
            "Final answer.",
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(researcher, "Summarise and check", context())
        assert result.skills_used == ["summarise", "critique"]

    def test_parallel_tool_calls_in_one_turn_all_run(self, researcher):
        provider = ScriptedAgentProvider(
            [
                ("summarise", {"text": "document one"}),
                ("summarise", {"text": "document two"}),
            ],
            "Both read.",
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(researcher, "Read both", context())
        assert result.skills_used == ["summarise", "summarise"]

    def test_only_the_agents_own_skills_are_offered(self, researcher, skills):
        reset_provider_cache(ScriptedAgentProvider("Done."))
        provider = ScriptedAgentProvider("Done.")
        reset_provider_cache(provider)

        runtime.run_agent(researcher, "Do the thing", context())
        assert provider.tools_offered[0] == ["summarise", "critique"]

    def test_token_usage_accumulates_across_iterations(self, researcher):
        provider = ScriptedAgentProvider(
            [("summarise", {"text": "a document"})], "Final answer."
        )
        reset_provider_cache(provider)
        result = runtime.run_agent(researcher, "Summarise", context())
        assert result.output_tokens == 10


class TestBudget:
    def test_an_agent_stops_at_its_iteration_budget(self, researcher):
        """Otherwise a model that keeps calling tools runs forever, on someone's
        bill."""
        provider = ScriptedAgentProvider(
            *[[("summarise", {"text": "again"})] for _ in range(20)]
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(researcher, "Summarise repeatedly", context())

        assert result.status == runtime.BUDGET_EXHAUSTED
        assert result.iterations == researcher.max_iterations

    def test_a_budget_exhausted_agent_still_returns_its_best_answer(self, researcher):
        provider = ScriptedAgentProvider(
            *[[("summarise", {"text": "again"})] for _ in range(3)],
            "This is as far as I got; the totals are still missing.",
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(researcher, "Summarise repeatedly", context())
        assert "as far as I got" in result.text


class TestFailureHandling:
    def test_an_agent_with_no_objective_is_refused(self, researcher):
        with pytest.raises(runtime.AgentError, match="no objective"):
            runtime.run_agent(researcher, "   ", context())

    def test_an_unreachable_model_is_a_typed_error(self, researcher):
        reset_provider_cache(FailingProvider(LLMProxyError("connection refused")))
        with pytest.raises(runtime.AgentError, match="could not reach"):
            runtime.run_agent(researcher, "Do the thing", context())

    def test_a_refusal_names_the_agent(self, researcher):
        reset_provider_cache(FailingProvider(LLMRefusal("declined", category="policy")))
        with pytest.raises(runtime.AgentError, match="Researcher"):
            runtime.run_agent(researcher, "Do the thing", context())

    def test_an_invented_skill_name_is_answered_with_the_real_ones(self, researcher):
        """Models occasionally invent a tool name. Naming the real ones usually
        recovers the turn; failing the run certainly does not."""
        provider = ScriptedAgentProvider(
            [("teleport", {"destination": "Denver"})], "Understood, I will summarise."
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(researcher, "Do the thing", context())

        [outcome] = result.turns[0].tool_results
        assert outcome.is_error
        assert "summarise" in outcome.output
        assert result.status == runtime.OBJECTIVE_MET

    def test_a_failing_skill_does_not_fail_the_run(self, researcher):
        provider = ScriptedAgentProvider(
            [("summarise", {})],  # missing the required argument
            "I could not summarise, so here is what I know directly.",
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(researcher, "Summarise it", context())
        assert result.status == runtime.OBJECTIVE_MET
        assert result.turns[0].tool_results[0].is_error


class TestAgentMemory:
    def test_a_successful_run_records_what_worked(self, researcher):
        provider = ScriptedAgentProvider(
            [("summarise", {"text": "a document"})], "Final answer."
        )
        reset_provider_cache(provider)

        runtime.run_agent(researcher, "Summarise the quarterly report", context())

        stored = memory_service.list_memories(TENANT, MemoryScope.AGENT, researcher.id)
        assert any("summarise" in entry.text for entry in stored)

    def test_a_lesson_records_the_subject_not_the_template(self):
        """Found by reading real memory after a browser run. A designed objective
        is mostly boilerplate identical on every run of that role, so storing it
        raw spends the whole excerpt on the part that never varies and truncates
        away the thing that identifies the run."""
        objective = (
            "Gather everything needed to address this goal. Note what you could not "
            "establish rather than filling the gap. Constraints: none stated\n\n"
            "The goal is:\nResearch our competitors and write a short comparison\n\n"
            "The previous agent produced:\nA long block of upstream work that belongs "
            "to that run rather than to this lesson."
        )
        condensed = runtime._condense(objective)

        assert condensed == "Research our competitors and write a short comparison"

    def test_a_hand_written_objective_still_condenses(self):
        """A canvas-authored objective has no markers, so it must degrade to
        plain single-line truncation rather than losing everything."""
        assert runtime._condense("  Do\n  the   thing  ") == "Do the thing"

    def test_a_very_long_objective_is_excerpted(self):
        condensed = runtime._condense("word " * 200)
        assert len(condensed) <= runtime._OBJECTIVE_EXCERPT + 1
        assert condensed.endswith("…")

    def test_the_lesson_reaches_the_next_run(self, researcher):
        """The self-improvement loop: run two starts with run one's lesson already
        in the prompt."""
        reset_provider_cache(
            ScriptedAgentProvider([("summarise", {"text": "a document"})], "Done.")
        )
        runtime.run_agent(researcher, "Summarise the quarterly report", context())

        second = ScriptedAgentProvider("Done again.")
        reset_provider_cache(second)
        runtime.run_agent(researcher, "Summarise the quarterly report", context())

        assert "summarise" in second.systems[0]

    def test_a_skill_failure_is_recorded_against_the_skill(self, researcher, skills):
        """So every agent that reaches for the capability inherits the lesson,
        not just this one."""
        reset_provider_cache(ScriptedAgentProvider([("summarise", {})], "Done."))
        runtime.run_agent(researcher, "Summarise it", context())

        stored = memory_service.list_memories(
            TENANT, MemoryScope.SKILL, skills["summarise"].id
        )
        assert stored

    def test_budget_exhaustion_is_remembered_as_a_failure(self, researcher):
        reset_provider_cache(
            ScriptedAgentProvider(*[[("summarise", {"text": "x"})] for _ in range(20)])
        )
        runtime.run_agent(researcher, "An objective that never finishes", context())

        stored = memory_service.list_memories(TENANT, MemoryScope.AGENT, researcher.id)
        assert any("more than" in entry.text for entry in stored)

    def test_recalled_memory_is_strengthened_by_a_good_run(self, researcher):
        reset_provider_cache(
            ScriptedAgentProvider([("summarise", {"text": "a doc"})], "Done.")
        )
        runtime.run_agent(researcher, "Summarise the report", context())
        [before] = memory_service.list_memories(TENANT, MemoryScope.AGENT, researcher.id)

        reset_provider_cache(ScriptedAgentProvider("Done."))
        runtime.run_agent(researcher, "Summarise the report", context())
        after = {
            entry.id: entry
            for entry in memory_service.list_memories(TENANT, MemoryScope.AGENT, researcher.id)
        }

        assert after[before.id].usefulness > before.usefulness

    def test_workflow_memory_is_shared_between_agents(self, researcher, skills):
        """A fact one agent established is available to the next, which is what
        makes it the *workflow's* memory rather than an agent's."""
        runtime.remember_for_workflow(TENANT, "wf_trip", "The traveller departs from Denver.")

        provider = ScriptedAgentProvider("Done.")
        reset_provider_cache(provider)
        runtime.run_agent(
            researcher, "Book the flights", context(workflow_memory_scope="wf_trip")
        )

        assert "Denver" in provider.systems[0]

    def test_an_agent_can_be_told_not_to_write_learnings(self, skills):
        from cwap_contracts.v2 import AgentMemoryConfig

        quiet = agent_registry.create(
            TENANT,
            name="Quiet",
            role="a agent that keeps no notes",
            skill_ids=[skills["summarise"].id],
            memory=AgentMemoryConfig(write_learnings=False),
        )
        reset_provider_cache(
            ScriptedAgentProvider([("summarise", {"text": "a doc"})], "Done.")
        )
        runtime.run_agent(quiet, "Summarise", context())

        assert memory_service.list_memories(TENANT, MemoryScope.AGENT, quiet.id) == []

    def test_one_agents_memory_does_not_reach_another(self, researcher, skills):
        reset_provider_cache(
            ScriptedAgentProvider([("summarise", {"text": "a doc"})], "Done.")
        )
        runtime.run_agent(researcher, "Summarise the quarterly report", context())

        writer = agent_registry.create(
            TENANT, name="Writer", role="an editor", skill_ids=[skills["draft_text"].id]
        )
        provider = ScriptedAgentProvider("Done.")
        reset_provider_cache(provider)
        runtime.run_agent(writer, "Write the introduction", context())

        assert "summarise" not in provider.systems[0]


class TestObservability:
    def test_the_loop_reports_what_it_is_doing(self, researcher):
        events: list[tuple[str, str]] = []

        def emit(event, *, message, data):
            events.append((event, message))

        reset_provider_cache(
            ScriptedAgentProvider([("summarise", {"text": "a doc"})], "Done.")
        )
        runtime.run_agent(researcher, "Summarise", context(emit=emit))

        names = [event for event, _message in events]
        assert "agent.thinking" in names
        assert "agent.skill_used" in names

    def test_every_turn_is_recorded_for_the_run_report(self, researcher):
        reset_provider_cache(
            ScriptedAgentProvider([("summarise", {"text": "a doc"})], "Done.")
        )
        result = runtime.run_agent(researcher, "Summarise", context())

        assert [turn.iteration for turn in result.turns] == [1, 2]
        assert result.turns[0].tool_calls[0].skill_name == "summarise"


class TestAgentRegistry:
    def test_an_agent_is_stored_and_reloaded(self, researcher):
        assert agent_registry.get(TENANT, researcher.id).name == "Researcher"

    def test_an_unknown_agent_is_a_typed_error(self):
        with pytest.raises(agent_registry.AgentNotFound):
            agent_registry.get(TENANT, "agt_missing")

    def test_one_tenant_cannot_read_anothers_agent(self, researcher):
        with pytest.raises(agent_registry.AgentNotFound):
            agent_registry.get("tenant-b", researcher.id)

    def test_editing_bumps_the_version(self, researcher):
        updated = agent_registry.update(TENANT, researcher.id, role="a thorough researcher")
        assert updated.version == researcher.version + 1

    def test_an_agent_can_be_deleted(self, researcher):
        assert agent_registry.delete(TENANT, researcher.id) is True
        assert agent_registry.delete(TENANT, researcher.id) is False


class TestSkillsAreAttachedNotEmbedded:
    def test_improving_a_skill_improves_every_agent_using_it(self, researcher, skills):
        """Agents reference skills by id, so a lesson learned through one agent is
        in place for every other agent that holds the same skill."""
        from skills import execution

        execution.teach_skill(skills["summarise"], "Always ask for an explicit focus.")

        second = agent_registry.create(
            TENANT, name="Other", role="another role", skill_ids=[skills["summarise"].id]
        )
        provider = ScriptedAgentProvider(
            [("summarise", {"text": "a document"})], "Done."
        )
        reset_provider_cache(provider)
        runtime.run_agent(second, "Summarise", context())

        # The lesson rides in the *skill's* prompt, which the agent never sees.
        assert skill_registry.find_by_name(TENANT, "summarise").invocations == 1

    def test_a_synthesised_skill_can_be_attached_to_an_agent(self, skills):
        """Item 4 end to end: a capability that did not exist is created and
        becomes usable by an agent in the same run."""
        skill, created = skill_registry.ensure_capability(
            TENANT, "reconcile expense claims against receipts"
        )
        assert created

        agent = agent_registry.create(
            TENANT, name="Reconciler", role="a finance assistant", skill_ids=[skill.id]
        )
        provider = ScriptedAgentProvider(
            [(skill.name, {"input": "the claims"})], "Reconciled."
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(agent, "Reconcile the claims", context())
        assert result.skills_used == [skill.name]


class TestCompositeAgentSkills:
    def test_an_agent_can_hold_a_composite_skill(self, skills):
        skill_registry.create(
            TENANT,
            SkillProposal(
                name="condense_and_shape",
                description="Condense then shape.",
                kind=SkillKind.COMPOSITE,
                parameters=[SkillParameter(name="text", description="Source.")],
                definition={"steps": [{"skill": "summarise"}, {"skill": "format_output"}]},
            ),
        )
        composite = skill_registry.find_by_name(TENANT, "condense_and_shape")
        agent = agent_registry.create(
            TENANT, name="Shaper", role="a shaper", skill_ids=[composite.id]
        )
        provider = ScriptedAgentProvider(
            [("condense_and_shape", {"text": "a long document"})], "Done."
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(agent, "Condense and shape it", context())
        assert not result.turns[0].tool_results[0].is_error
