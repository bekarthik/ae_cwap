"""Designing a workflow from a goal.

The v1 behaviour was one-shot: a goal in, a graph out. That is fine for a goal
that is already specific and wrong for everything else — "sort out our
onboarding" does not contain enough information to design anything, and guessing
produces a confident, useless workflow.

So designing is a short conversation:

    goal ──► clarifying questions (only what cannot be inferred)
              │
              answers
              ▼
            design: agents, the skills each needs, memory wiring, the graph
              │
              └─► skills that do not exist yet are created

The design is deterministic at its core, with the model used to *improve* the
result rather than to produce it. That is deliberate: the platform must design
something sensible when it is running on a small local model, or on the offline
stub, and a design that varies run to run is impossible to review.
"""

from __future__ import annotations

import json
import re
import uuid

from agents import registry as agent_registry
from cwap_contracts.v2 import (
    AgentMemoryConfig,
    ClarifyingQuestion,
    DesignRequest,
    DesignResponse,
    DesignStage,
    NodeType,
    PlannedAgent,
    Position,
    SkillGap,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
)
from llm_proxy.client import GenerationOptions, LLMProxyError, get_provider
from skills import registry as skill_registry

from design import blueprints

#: More than this and the conversation stops feeling like help.
MAX_QUESTIONS = 4

#: More than this and the user is reviewing a system, not a workflow.
MAX_AGENTS = 4

_X_STEP = 300
_Y_BASE = 140


def design(request: DesignRequest, tenant_id: str) -> DesignResponse:
    """One turn of the design conversation."""
    intent = blueprints.classify(request.goal)
    questions = _questions_for(intent, request)

    unanswered = [q for q in questions if q.id not in request.answers]
    if unanswered and not request.skip_questions:
        return DesignResponse(
            stage=DesignStage.CLARIFYING,
            understanding=blueprints.restate(request.goal, intent),
            questions=unanswered,
            notes=[
                "Answer what you can — anything you skip uses a sensible default.",
            ],
        )

    return _build(request, tenant_id, intent, questions)


# ---------------------------------------------------------------------------
# asking
# ---------------------------------------------------------------------------


def _questions_for(intent: blueprints.Intent, request: DesignRequest) -> list[ClarifyingQuestion]:
    """Only ask what the goal does not already answer.

    A question the goal has already answered reads as though the system was not
    listening, so each is suppressed when the text makes it redundant.
    """
    goal = request.goal.lower()
    questions: list[ClarifyingQuestion] = []

    if not _mentions_deliverable(goal):
        questions.append(
            ClarifyingQuestion(
                id="deliverable",
                question="What should this produce when it finishes?",
                why="It decides what the final step builds and how we check it is done.",
                options=intent.deliverable_options,
                default=intent.default_deliverable,
            )
        )

    if not _mentions_audience(goal):
        questions.append(
            ClarifyingQuestion(
                id="audience",
                question="Who is the result for?",
                why="It sets the tone and level of detail the agents write at.",
                options=["Me", "My team", "A customer or client", "A public audience"],
                default="Me",
            )
        )

    if request.knowledge_handles:
        questions.append(
            ClarifyingQuestion(
                id="grounding",
                question="Should this rely on your uploaded documents?",
                why="If so, agents get a search capability over them instead of "
                "answering from general knowledge.",
                options=["Yes, use my documents", "No, general knowledge is fine"],
                default="Yes, use my documents",
            )
        )

    if intent.may_need_external:
        questions.append(
            ClarifyingQuestion(
                id="external",
                question="Does this need to send or fetch anything from another system?",
                why="External calls need a credential and an approved host, so we "
                "flag them rather than building them silently.",
                options=["No", "Yes — I will supply the details"],
                default="No",
            )
        )

    if len(questions) < MAX_QUESTIONS:
        questions.append(
            ClarifyingQuestion(
                id="constraints",
                question="Anything the result must respect?",
                why="Constraints are carried into every agent's instructions and "
                "remembered for future runs.",
                default="",
            )
        )

    return questions[:MAX_QUESTIONS]


def _mentions_deliverable(goal: str) -> bool:
    return any(
        word in goal
        for word in (
            "report", "email", "summary", "plan", "list", "table", "draft",
            "document", "post", "answer", "itinerary", "comparison", "spreadsheet",
        )
    )


def _mentions_audience(goal: str) -> bool:
    return any(
        word in goal
        for word in ("for my", "for our", "for the team", "for a client", "for customers", "for me")
    )


# ---------------------------------------------------------------------------
# designing
# ---------------------------------------------------------------------------


def _build(
    request: DesignRequest,
    tenant_id: str,
    intent: blueprints.Intent,
    questions: list[ClarifyingQuestion],
) -> DesignResponse:
    answers = {
        question.id: (request.answers.get(question.id) or question.default).strip()
        for question in questions
    }

    use_documents = bool(request.knowledge_handles) and not answers.get(
        "grounding", ""
    ).lower().startswith("no")

    planned = _plan_agents(request.goal, intent, answers, use_documents)
    planned = _refine_with_model(request.goal, planned, answers)

    skill_registry.ensure_builtins(tenant_id)
    resolved, gaps = _resolve_skills(
        tenant_id, planned, request.goal, request.knowledge_handles if use_documents else []
    )

    if answers.get("external", "").lower().startswith("yes"):
        gaps.append(
            SkillGap(
                needed_by=planned[0].name,
                capability="call the external system you mentioned",
                blocked_reason=(
                    "External calls need a credential and an allow-listed host. Add the "
                    "host to CWAP_HTTP_ALLOWLIST and create the skill once you have the "
                    "credential — we will not invent one."
                ),
            )
        )

    graph = _build_graph(request.goal, planned, resolved, tenant_id, answers)

    return DesignResponse(
        stage=DesignStage.DESIGNED,
        understanding=blueprints.restate(request.goal, intent, answers),
        agents=planned,
        skill_gaps=gaps,
        graph=graph,
        notes=_notes(planned, gaps, use_documents),
    )


def _plan_agents(
    goal: str,
    intent: blueprints.Intent,
    answers: dict[str, str],
    use_documents: bool,
) -> list[PlannedAgent]:
    """Decompose the goal into roles.

    Deterministic, from the intent's blueprint. The model refines wording in
    `_refine_with_model`; it does not get to invent the structure, because a
    structure that changes between identical requests cannot be reviewed.
    """
    audience = answers.get("audience") or "the requester"
    deliverable = answers.get("deliverable") or intent.default_deliverable
    constraints = answers.get("constraints") or "none stated"

    planned: list[PlannedAgent] = []
    for template in intent.agents[:MAX_AGENTS]:
        skills = list(template.skills)
        if use_documents and template.wants_documents:
            skills.insert(0, "search_my_documents")

        planned.append(
            PlannedAgent(
                name=template.name,
                role=template.role,
                objective=template.objective.format(
                    goal=goal, audience=audience, deliverable=deliverable,
                    constraints=constraints,
                ),
                skills=skills,
                rationale=template.rationale,
            )
        )
    return planned


REFINE_SYSTEM = """\
You improve the wording of a workflow design. You do NOT change its structure.

Return one JSON object: {"agents": [{"name": "...", "role": "...", "objective": "..."}]}
with exactly the same number of agents, in the same order. Keep each name short.
Make each role and objective specific to the user's actual goal. Return only JSON.
"""


def _refine_with_model(
    goal: str, planned: list[PlannedAgent], answers: dict[str, str]
) -> list[PlannedAgent]:
    """Let the model sharpen the wording. Structure is fixed.

    Any failure — no model, bad JSON, wrong agent count — keeps the deterministic
    design, so this can only improve the result, never break it.
    """
    if not planned:
        return planned

    sketch = [{"name": a.name, "role": a.role, "objective": a.objective} for a in planned]
    prompt = (
        f"Goal:\n{goal}\n\nAnswers:\n{json.dumps(answers, indent=2)}\n\n"
        f"Design to reword:\n{json.dumps(sketch, indent=2)}"
    )

    try:
        completion = get_provider().complete(
            prompt,
            system=REFINE_SYSTEM,
            options=GenerationOptions(max_tokens=1500, temperature=0.3, effort="medium"),
        )
        payload = _extract_json(completion.text)
        refined = (payload or {}).get("agents")
        if not isinstance(refined, list) or len(refined) != len(planned):
            return planned

        return [
            original.model_copy(
                update={
                    "name": str(item.get("name") or original.name)[:120],
                    "role": str(item.get("role") or original.role)[:500],
                    "objective": str(item.get("objective") or original.objective)[:2000],
                }
            )
            for original, item in zip(planned, refined, strict=True)
        ]
    except (LLMProxyError, ValueError, TypeError):
        return planned


def _extract_json(text: str) -> dict | None:
    candidate = text.strip()
    if "```" in candidate:
        blocks = [part for part in candidate.split("```") if "{" in part]
        if blocks:
            candidate = blocks[0].lstrip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:]
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _resolve_skills(
    tenant_id: str,
    planned: list[PlannedAgent],
    goal: str,
    knowledge_handles: list[str],
) -> tuple[dict[str, list[str]], list[SkillGap]]:
    """Find or create every skill the design calls for.

    This is the "if a skill does not exist, create it" step. Retrieval skills are
    built directly rather than synthesised, because their definition is fully
    determined by which corpus they search.
    """
    resolved: dict[str, list[str]] = {}
    gaps: list[SkillGap] = []

    for agent in planned:
        skill_ids: list[str] = []
        for wanted in agent.skills:
            if wanted == "search_my_documents":
                if not knowledge_handles:
                    continue
                skill = _ensure_retrieval_skill(tenant_id, knowledge_handles[0])
                skill_ids.append(skill.id)
                continue

            existing = skill_registry.find_by_name(tenant_id, wanted)
            if existing is not None:
                skill_ids.append(existing.id)
                continue

            skill, created = skill_registry.ensure_capability(
                tenant_id,
                wanted.replace("_", " "),
                context=f"agent '{agent.name}' working on: {goal}",
                knowledge_handles=knowledge_handles,
            )
            skill_ids.append(skill.id)
            if created:
                gaps.append(
                    SkillGap(
                        needed_by=agent.name,
                        capability=wanted.replace("_", " "),
                        proposal=None,
                    )
                )

        resolved[agent.name] = skill_ids

    return resolved, gaps


def _ensure_retrieval_skill(tenant_id: str, handle: str):
    """A search capability over one corpus, created once and reused."""
    from cwap_contracts.v2 import SkillKind, SkillParameter, SkillProposal  # noqa: PLC0415

    name = f"search_documents_{handle[-8:]}"
    existing = skill_registry.find_by_name(tenant_id, name)
    if existing is not None:
        return existing

    return skill_registry.create(
        tenant_id,
        SkillProposal(
            name=name,
            description=(
                "Search the user's uploaded documents for passages relevant to a "
                "question. Use this before answering anything that the documents "
                "might cover."
            ),
            kind=SkillKind.RETRIEVAL,
            parameters=[
                SkillParameter(name="query", description="What to look for.")
            ],
            definition={"knowledge_handle": handle, "top_k": 4},
        ),
    )


def _build_graph(
    goal: str,
    planned: list[PlannedAgent],
    resolved: dict[str, list[str]],
    tenant_id: str,
    answers: dict[str, str],
) -> WorkflowGraph:
    """Create the agents and lay them out as a runnable workflow.

    There is no separate retrieval node: an agent that needs documents is given a
    search *skill*, so it decides when to look rather than always looking first.
    """
    workflow_id = f"wf_{uuid.uuid4().hex[:16]}"
    nodes: list[WorkflowNode] = [
        WorkflowNode(
            id="input",
            type=NodeType.INPUT,
            label="Goal",
            params={"fields": ["goal"], "defaults": {"goal": goal}},
            position=Position(x=0, y=_Y_BASE),
        )
    ]
    edges: list[WorkflowEdge] = []
    previous = "input"

    for index, agent in enumerate(planned, start=1):
        stored = agent_registry.create(
            tenant_id,
            name=agent.name,
            role=agent.role,
            objective=agent.objective,
            instructions=_instructions(answers),
            skill_ids=resolved.get(agent.name, []),
            max_iterations=6,
            memory=AgentMemoryConfig(),
        )
        node_id = f"agent_{index}"
        nodes.append(
            WorkflowNode(
                id=node_id,
                type=NodeType.AGENT,
                label=agent.name,
                params={"objective_template": _objective_template(index, agent)},
                position=Position(x=float(index * _X_STEP), y=_Y_BASE),
                agent_id=stored.id,
            )
        )
        edges.append(
            WorkflowEdge(
                id=f"e_{previous}_{node_id}",
                source=previous,
                target=node_id,
                bindings=(
                    {"goal": "$run.input.goal"}
                    if previous == "input"
                    else {"goal": "$run.input.goal", "previous": "$output.text"}
                ),
            )
        )
        previous = node_id

    nodes.append(
        WorkflowNode(
            id="output",
            type=NodeType.OUTPUT,
            label="Result",
            params={"result_template": "{{previous}}"},
            position=Position(x=float((len(planned) + 1) * _X_STEP), y=_Y_BASE),
        )
    )
    edges.append(
        WorkflowEdge(
            id=f"e_{previous}_output",
            source=previous,
            target="output",
            bindings={"previous": "$output.text"},
        )
    )

    return WorkflowGraph(
        id=workflow_id,
        name=_name_for(goal),
        nodes=nodes,
        edges=edges,
        # Every agent in this workflow shares one memory scope, so what one
        # establishes is available to the next — and to every future run.
        memory_scope_id=workflow_id,
    )


def _objective_template(index: int, agent: PlannedAgent) -> str:
    if index == 1:
        return f"{agent.objective}\n\nThe goal is:\n{{{{goal}}}}"
    return (
        f"{agent.objective}\n\nThe goal is:\n{{{{goal}}}}\n\n"
        "The previous agent produced:\n{{previous}}"
    )


def _instructions(answers: dict[str, str]) -> str:
    parts = []
    if answers.get("audience"):
        parts.append(f"The result is for: {answers['audience']}.")
    if answers.get("deliverable"):
        parts.append(f"The deliverable is: {answers['deliverable']}.")
    if answers.get("constraints"):
        parts.append(f"Constraints to respect: {answers['constraints']}.")
    return " ".join(parts)


def _notes(planned: list[PlannedAgent], gaps: list[SkillGap], use_documents: bool) -> list[str]:
    notes = [
        f"{len(planned)} agent(s), each with its own memory. They improve across runs.",
    ]
    created = [gap for gap in gaps if gap.blocked_reason == ""]
    if created:
        notes.append(
            f"Created {len(created)} new skill(s) that did not exist yet: "
            + ", ".join(gap.capability for gap in created)
        )
    blocked = [gap for gap in gaps if gap.blocked_reason]
    if blocked:
        notes.append(f"{len(blocked)} capability needs your input before it can be built.")
    if use_documents:
        notes.append("Agents can search your documents; they decide when it is worth doing.")
    return notes


def _name_for(goal: str) -> str:
    condensed = re.sub(r"\s+", " ", goal).strip()
    return condensed[:60] + ("…" if len(condensed) > 60 else "")
