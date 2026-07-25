"""The agent loop.

This is what separates an agent from a step. A step calls a model once and
returns the text. An agent is given an objective and a set of skills, and then:

    recall what it has learned  →  decide  →  invoke a skill  →  see the result
                                      ↑                              │
                                      └──────────────────────────────┘
                                        until the objective is met
                                        or the iteration budget runs out

Afterwards it reflects: what it learned about its role goes to agent memory, what
it learned about a capability goes to that skill's memory, and what it learned
about the job goes to workflow memory. That is the self-improvement loop — the
next run starts with the previous run's lessons already in the prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cwap_contracts.v4 import (
    AgentDefinition,
    AgentTurn,
    MemoryKind,
    MemoryScope,
    MemoryWriteRequest,
    RecallRequest,
    SkillDefinition,
    ToolCall,
    ToolResult,
)
from llm_proxy.client import (
    ChatMessage,
    GenerationOptions,
    LLMProxyError,
    LLMRefusal,
    get_provider,
    provider_for_choice,
)
from llm_proxy.live import LiveText
from memory import service as memory_service
from skills import execution as skill_execution
from skills import registry as skill_registry

#: A budget-exhausted agent has produced *something*; the caller decides whether
#: it is usable. Distinct from failure on purpose.
BUDGET_EXHAUSTED = "budget_exhausted"
OBJECTIVE_MET = "objective_met"


class AgentError(RuntimeError):
    """The agent could not run at all — a missing model, an unusable definition."""


@dataclass
class AgentRunContext:
    tenant_id: str
    run_id: str = ""
    step_execution_id: str = ""
    workflow_memory_scope: str = ""
    allow_side_effects: bool = False
    emit: Any = None
    #: Live-only publisher for output as the model produces it. Separate from
    #: `emit` because these are never persisted: the finished answer is on the
    #: step, and a row per fragment would bury the run report in its own tokens.
    stream: Any = None


@dataclass
class AgentResult:
    text: str
    status: str
    turns: list[AgentTurn] = field(default_factory=list)
    skills_used: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    #: Memory entries consulted, so the outcome can reinforce or weaken them.
    recalled_memory_ids: list[str] = field(default_factory=list)
    #: What the model thought before each turn, when it is a reasoning model and
    #: the backend returned it. Carried beside `turns` rather than on `AgentTurn`,
    #: which is a published contract and cannot grow a field without a version.
    reasoning: list[dict[str, Any]] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        return len(self.turns)


def run_agent(
    agent: AgentDefinition, objective: str, context: AgentRunContext
) -> AgentResult:
    """Run one agent to completion or to its budget."""
    if not objective.strip():
        raise AgentError(f"agent '{agent.name}' was given no objective")

    skills = skill_registry.get_many(agent.tenant_id, agent.skill_ids)
    by_name = {skill.name: skill for skill in skills}
    tools = [skill.tool_schema() for skill in skills]

    recalled = _recall(agent, objective, context)
    system = _build_system_prompt(agent, recalled.text, skills=skills)

    provider = _provider_for(agent)
    messages: list[ChatMessage] = [ChatMessage(role="user", content=objective)]

    turns: list[AgentTurn] = []
    used: list[str] = []
    thought: list[dict[str, Any]] = []
    totals = [0, 0]

    for iteration in range(1, agent.max_iterations + 1):
        completion = _think(provider, messages, system, tools, agent, iteration, context)
        totals[0] += completion.input_tokens
        totals[1] += completion.output_tokens
        if completion.reasoning:
            thought.append({"iteration": iteration, "text": completion.reasoning})

        if not completion.tool_calls:
            turns.append(
                AgentTurn(
                    iteration=iteration,
                    text=completion.text,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                )
            )
            result = AgentResult(
                text=completion.text,
                status=OBJECTIVE_MET,
                turns=turns,
                skills_used=used,
                input_tokens=totals[0],
                output_tokens=totals[1],
                recalled_memory_ids=recalled.ids,
                reasoning=thought,
            )
            _reflect(agent, objective, result, context)
            return result

        calls = [
            ToolCall(id=call.id, skill_name=call.name, arguments=call.arguments)
            for call in completion.tool_calls
        ]
        results = _invoke(calls, by_name, agent, context, used)

        turns.append(
            AgentTurn(
                iteration=iteration,
                text=completion.text,
                tool_calls=calls,
                tool_results=results,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            )
        )

        messages.append(
            ChatMessage(
                role="assistant", content=completion.text, tool_calls=completion.tool_calls
            )
        )
        for outcome in results:
            messages.append(
                ChatMessage(
                    role="tool",
                    content=outcome.output,
                    tool_call_id=outcome.call_id,
                    name=outcome.skill_name,
                )
            )

    # Budget spent. Ask once for the best answer it can give from what it has,
    # rather than returning an empty string or a half-finished tool call.
    final = _final_answer(provider, messages, system, agent, context, turns)
    result = AgentResult(
        text=final,
        status=BUDGET_EXHAUSTED,
        turns=turns,
        skills_used=used,
        input_tokens=totals[0],
        output_tokens=totals[1],
        recalled_memory_ids=recalled.ids,
        reasoning=thought,
    )
    _reflect(agent, objective, result, context)
    return result


# ---------------------------------------------------------------------------
# steps of the loop
# ---------------------------------------------------------------------------


def _provider_for(agent: AgentDefinition):
    """The model this agent runs on.

    An agent that names one gets it; an agent that names nothing gets the
    workspace's. That is what "mix providers freely within one workflow" means
    in practice — a triage step on a small local model handing to a synthesis
    step on a hosted one, in the same run.
    """
    if agent.model_provider or agent.model_override:
        return provider_for_choice(agent.model_provider, agent.model_override or "")
    return get_provider()


def _options(agent: AgentDefinition) -> GenerationOptions:
    """What this agent asks the model for.

    Effort is per agent because that is where the decision belongs: triage wants
    `low`, synthesis wants `high`, and one deployment-wide setting cannot be
    both. Blank inherits the deployment's, and a backend without a reasoning
    mode reports it as ignored rather than silently dropping it.
    """
    return GenerationOptions(effort=agent.thinking_effort or None, max_tokens=None)


def _think(provider, messages, system, tools, agent, iteration, context):
    _emit(
        context,
        "agent.thinking",
        f"'{agent.name}' iteration {iteration}/{agent.max_iterations}",
        {"iteration": iteration, "skills_offered": len(tools)},
    )
    live = _live_sink(agent, iteration, context)
    try:
        completion = provider.converse(
            messages,
            system=system,
            tools=tools or None,
            options=_options(agent),
            on_delta=live,
        )
    except LLMRefusal as exc:
        raise AgentError(
            f"the model declined to act as '{agent.name}'"
            + (f" (category: {exc.category})" if exc.category else "")
        ) from exc
    except LLMProxyError as exc:
        raise AgentError(f"agent '{agent.name}' could not reach the model: {exc}") from exc
    finally:
        # Even on a failure: what the model managed to say before it broke is
        # the most useful thing on the screen at that moment.
        if live is not None:
            live.flush()

    if completion.reasoning:
        _emit(
            context,
            "agent.reasoned",
            f"'{agent.name}' thought first: {_excerpt(completion.reasoning)}",
            {"iteration": iteration, "reasoning": completion.reasoning},
        )
    return completion


def _live_sink(agent, iteration: int, context: AgentRunContext):
    """Publish this turn's output as it is written, if anyone is watching.

    Returns None when the run has no live channel — a design-time call, a test,
    a workflow executed outside the orchestrator — so the provider does the
    ordinary thing and nothing pays for a stream nobody is reading.
    """
    if context is None or getattr(context, "stream", None) is None:
        return None

    def publish(kind: str, text: str) -> None:
        # The fragment travels in `data`, never in `message`: every contract
        # model strips leading and trailing whitespace from its string fields,
        # which would silently glue "Here" and "is" into "Hereis" at every
        # chunk boundary. `data` values are untyped and arrive as sent.
        context.stream(
            "agent.streaming",
            message=f"{agent.name} is {'thinking' if kind == 'reasoning' else 'writing'}",
            data={
                "agent": agent.name,
                "iteration": iteration,
                "kind": kind,
                "text": text,
            },
        )

    return LiveText(publish)


#: Reasoning runs long — often longer than the answer. The log line carries
#: enough to see the direction the agent took; the whole of it is in the event
#: data and on the step, where it can be read without flooding the stream.
_REASONING_EXCERPT = 300


def _excerpt(text: str) -> str:
    condensed = " ".join(text.split())
    if len(condensed) <= _REASONING_EXCERPT:
        return condensed
    return condensed[:_REASONING_EXCERPT].rstrip() + "…"


def _invoke(
    calls: list[ToolCall],
    by_name: dict[str, SkillDefinition],
    agent: AgentDefinition,
    context: AgentRunContext,
    used: list[str],
) -> list[ToolResult]:
    results: list[ToolResult] = []

    for call in calls:
        skill = by_name.get(call.skill_name)
        if skill is None:
            # Tell the agent rather than failing: models occasionally invent a
            # tool name, and naming the real ones usually recovers the turn.
            results.append(
                ToolResult(
                    call_id=call.id,
                    skill_name=call.skill_name,
                    output=(
                        f"No skill named '{call.skill_name}'. Available: "
                        f"{sorted(by_name)}"
                    ),
                    is_error=True,
                )
            )
            continue

        started = time.perf_counter()
        outcome = skill_execution.execute(
            skill,
            call.arguments,
            skill_execution.SkillContext(
                tenant_id=agent.tenant_id,
                run_id=context.run_id,
                step_execution_id=context.step_execution_id,
                allow_side_effects=context.allow_side_effects,
                emit=context.emit,
            ),
        )
        used.append(skill.name)
        _emit(
            context,
            "agent.skill_used",
            f"'{agent.name}' used {skill.name}"
            + (" — it failed" if outcome.is_error else ""),
            {
                "skill": skill.name,
                "arguments": call.arguments,
                "error": outcome.is_error,
                "duration_ms": outcome.duration_ms,
            },
        )
        results.append(
            ToolResult(
                call_id=call.id,
                skill_name=skill.name,
                output=outcome.output,
                is_error=outcome.is_error,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        )

    return results


def _final_answer(provider, messages, system, agent, context, turns) -> str:
    _emit(
        context,
        "agent.budget_exhausted",
        f"'{agent.name}' used all {agent.max_iterations} iterations; "
        "asking for its best answer so far",
        {"max_iterations": agent.max_iterations},
    )
    closing = list(messages) + [
        ChatMessage(
            role="user",
            content=(
                "You have no further tool calls available. Give your best answer from "
                "what you have gathered, and state plainly what is still missing."
            ),
        )
    ]
    try:
        return provider.converse(
            closing, system=system, tools=None, options=_options(agent)
        ).text
    except LLMProxyError:
        # Losing the closing summary is bad; losing the whole run is worse. Hand
        # back what the agent actually said along the way.
        return "\n\n".join(turn.text for turn in turns if turn.text) or (
            "The agent ran out of iterations before reaching a conclusion."
        )


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


@dataclass
class _Recalled:
    text: str = ""
    ids: list[str] = field(default_factory=list)


def _recall(
    agent: AgentDefinition, objective: str, context: AgentRunContext
) -> _Recalled:
    """Read agent memory, and the workflow's, before deciding anything."""
    if not agent.memory.recall:
        return _Recalled()

    blocks: list[str] = []
    ids: list[str] = []

    own = memory_service.recall(
        RecallRequest(
            tenant_id=agent.tenant_id,
            scope=MemoryScope.AGENT,
            scope_id=agent.id,
            query=objective,
            limit=agent.memory.recall_limit,
        )
    )
    if own.entries:
        blocks.append(own.as_prompt_block("What you have learned in this role"))
        ids.extend(entry.id for entry in own.entries)

    if agent.memory.use_workflow_memory and context.workflow_memory_scope:
        shared = memory_service.recall(
            RecallRequest(
                tenant_id=agent.tenant_id,
                scope=MemoryScope.WORKFLOW,
                scope_id=context.workflow_memory_scope,
                query=objective,
                limit=agent.memory.recall_limit,
            )
        )
        if shared.entries:
            blocks.append(shared.as_prompt_block("What this workflow has established"))
            ids.extend(entry.id for entry in shared.entries)

    if ids:
        _emit(
            context,
            "agent.memory_recalled",
            f"'{agent.name}' recalled {len(ids)} prior learning(s)",
            {"count": len(ids)},
        )

    return _Recalled(text="\n\n".join(blocks), ids=ids)


def _reflect(
    agent: AgentDefinition, objective: str, result: AgentResult, context: AgentRunContext
) -> None:
    """Write down what this run taught, at the scope that owns the lesson.

    Deliberately mechanical rather than asking the model to introspect: an
    LLM-authored "lesson" after every run fills memory with plausible platitudes,
    and the observations worth keeping — which skills failed, whether the budget
    was enough — are things the runtime already knows for certain.
    """
    if not agent.memory.write_learnings:
        return

    failures = [
        outcome
        for turn in result.turns
        for outcome in turn.tool_results
        if outcome.is_error
    ]

    for outcome in failures:
        skill = skill_registry.find_by_name(agent.tenant_id, outcome.skill_name)
        if skill is None:
            continue
        # The lesson belongs to the *skill*, so every agent that uses it benefits.
        skill_execution.teach_skill(
            skill,
            f"Invoking this with arguments that produced: {outcome.output[:300]}",
            kind=MemoryKind.FAILURE,
            run_id=context.run_id or None,
        )

    if result.status == BUDGET_EXHAUSTED:
        _remember(
            agent,
            f"Objective of this shape needed more than {agent.max_iterations} iterations: "
            f"{_condense(objective)}",
            MemoryKind.FAILURE,
            context,
        )
    elif result.skills_used:
        ordered = " then ".join(dict.fromkeys(result.skills_used))
        _remember(
            agent,
            f"For an objective like '{_condense(objective)}', {ordered} worked in "
            f"{result.iterations} iteration(s).",
            MemoryKind.LEARNING,
            context,
        )

    # Memories that preceded a good run become slightly more trusted; ones that
    # preceded a budget blow-out become slightly less.
    if result.recalled_memory_ids:
        memory_service.reinforce(
            result.recalled_memory_ids,
            delta=0.2 if result.status == OBJECTIVE_MET else -0.3,
        )


#: Long enough to identify what the run was about, short enough that a dozen
#: recalled lessons do not crowd out the task itself.
_OBJECTIVE_EXCERPT = 160

#: The section markers a designed objective template uses. Kept here rather than
#: imported from the design service, because an objective written by hand on the
#: canvas will not have them and must degrade to plain truncation.
_GOAL_MARKER = "The goal is:"
_HANDOFF_MARKER = "The previous agent produced:"


def _condense(objective: str) -> str:
    """One line, no template scaffolding.

    An objective is a rendered template, so it arrives with newlines and a
    boilerplate preamble that is identical on every run of that role. Storing it
    raw wastes the excerpt on the part that never varies — the actual subject
    gets truncated away, and lexical recall then matches on the boilerplate.

    A designed objective marks the goal with "The goal is:" and the upstream
    agent's work with "The previous agent produced:". Keeping the span between
    them isolates what this run was actually about; the predecessor's output is
    dropped because it belongs to that run, not to the lesson.
    """
    text = objective
    if _GOAL_MARKER in text:
        text = text.split(_GOAL_MARKER, 1)[1]
    text = text.split(_HANDOFF_MARKER, 1)[0]

    condensed = " ".join(text.split())
    if len(condensed) <= _OBJECTIVE_EXCERPT:
        return condensed
    return condensed[:_OBJECTIVE_EXCERPT].rstrip() + "…"


def _remember(
    agent: AgentDefinition, text: str, kind: MemoryKind, context: AgentRunContext
) -> None:
    memory_service.remember(
        MemoryWriteRequest(
            tenant_id=agent.tenant_id,
            scope=MemoryScope.AGENT,
            scope_id=agent.id,
            kind=kind,
            text=text,
            source_run_id=context.run_id or None,
        )
    )


def remember_for_workflow(
    tenant_id: str, scope_id: str, text: str, *, run_id: str | None = None
) -> None:
    """Record something the whole workflow should know next time."""
    memory_service.remember(
        MemoryWriteRequest(
            tenant_id=tenant_id,
            scope=MemoryScope.WORKFLOW,
            scope_id=scope_id,
            kind=MemoryKind.FACT,
            text=text,
            source_run_id=run_id,
        )
    )


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


def _build_system_prompt(
    agent: AgentDefinition, recalled: str, *, skills: list[SkillDefinition] | None = None
) -> str:
    """The agent's instructions, written for what it can actually do.

    An agent with tools is told to establish things rather than recall them; an
    agent with none is told the opposite, because instructing a model to "use
    your tools" when it has none produces apologies for tools it cannot see, or
    invented calls. The same prompt cannot honestly serve both.
    """
    parts = [f"You are {agent.name}. {agent.role}", ""]

    if skills:
        named = ", ".join(skill.name for skill in skills)
        parts += [
            "Work by using the tools available to you. Call one, read its result, and "
            "decide what to do next. When you have what you need, answer directly "
            "instead of calling another tool.",
            "",
            f"Your tools: {named}.",
            "",
            "Check rather than recall. If something the answer depends on could be "
            "established with one of these tools, use it — even when you believe you "
            "already know. What you remember may be out of date or may not apply here; "
            "what a tool returns is about this case.",
            "",
            "If a tool fails, read the error and try different arguments or a different "
            "tool. Do not repeat a call that just failed the same way.",
        ]
    else:
        parts += [
            "You have no tools for this task. Work from the material you have been "
            "given, and say plainly where an answer would need something you cannot "
            "reach from here.",
        ]

    parts += [
        "",
        "State plainly what you could not determine. Never invent a fact to fill a gap.",
    ]
    if agent.instructions:
        parts += ["", agent.instructions]
    if recalled:
        parts += ["", recalled]
    return "\n".join(parts)


def _emit(context: AgentRunContext, event: str, message: str, data: dict | None = None) -> None:
    if context.emit is not None:
        context.emit(event, message=message, data=data or {})
