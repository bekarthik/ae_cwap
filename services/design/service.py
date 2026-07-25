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
from cwap_contracts.v4 import (
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

#: A bound on review effort and cost, not an opinion about structure.
#:
#: Every agent is a stored object with its own memory and its own chain of model
#: calls, so a runaway design is both expensive and unreviewable. Eight leaves
#: room for a full delivery pipeline — specify, break down, build, test, review,
#: land — plus a couple the goal turns out to need. How many are actually used is
#: the model's decision from the goal; this only says where it stops.
MAX_AGENTS = 8

_X_STEP = 300
_Y_BASE = 140


def design(request: DesignRequest, tenant_id: str) -> DesignResponse:
    """One turn of the design conversation."""
    intent, _confident = blueprints.classify_with_confidence(request.goal)
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

    _intent, confident = blueprints.classify_with_confidence(request.goal)

    skill_registry.ensure_builtins(tenant_id)
    connectors = _connector_skills(tenant_id)

    # The model decides the structure. How many agents a goal needs, and what
    # each hands to the next, is a property of the goal — it cannot be settled
    # before reading it. The blueprint that classification found is passed in as
    # a *hint*, because a shape that usually works for this kind of request is
    # worth knowing, and it stays the fallback for when there is no usable model.
    plan = _plan_with_model(
        request.goal,
        answers,
        connectors,
        hint=intent if confident else None,
        use_documents=use_documents,
    )
    proposed = plan is not None
    if plan is None:
        planned = _plan_agents(request.goal, intent, answers, use_documents)
        budgets = {t.name: t.max_iterations for t in intent.agents}
    else:
        planned, budgets = plan

    resolved, gaps = _resolve_skills(
        tenant_id, planned, request.goal, request.knowledge_handles if use_documents else []
    )
    _attach_connectors(intent, planned, resolved, connectors, gaps, proposed)

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

    _note_write_scope(planned, resolved, tenant_id, gaps)
    planned = _mark_reuse(tenant_id, planned)
    graph = _build_graph(request.goal, planned, resolved, tenant_id, answers, budgets)

    return DesignResponse(
        stage=DesignStage.DESIGNED,
        understanding=blueprints.restate(request.goal, intent, answers),
        agents=planned,
        skill_gaps=gaps,
        graph=graph,
        notes=_notes(planned, gaps, use_documents, proposed),
    )


def _mark_reuse(tenant_id: str, planned: list[PlannedAgent]) -> list[PlannedAgent]:
    """Say which of these the workspace already has.

    Read before anything is stored, because afterwards every agent exists and
    the answer is always "reused". A plan that silently reuses is as opaque as
    one that silently duplicates — the review screen should be able to show
    which of these are yours already.
    """
    marked: list[PlannedAgent] = []
    for agent in planned:
        existing = agent_registry.find_by_name(tenant_id, agent.name)
        marked.append(agent.model_copy(update={"reused": existing is not None}))
    return marked


def _plan_agents(
    goal: str,
    intent: blueprints.Intent,
    answers: dict[str, str],
    use_documents: bool,
) -> list[PlannedAgent]:
    """Decompose the goal into roles.

    The fallback, used when there is no usable model or its plan did not
    validate. Deterministic, so a deployment running the offline stub — or a
    small local model that cannot return clean JSON — still gets a coherent
    workflow rather than an error.
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
                    previous="{{previous}}",
                ),
                skills=skills,
                rationale=template.rationale,
            )
        )
    return planned


PLAN_SYSTEM = """\
You design a team of AI agents that will run one after another to accomplish a
goal. You decide the structure.

Return ONE JSON object:

{"agents": [{"name": "...", "role": "...", "objective": "...",
             "skills": ["..."], "max_iterations": 6, "rationale": "..."}]}

Deciding the structure means deciding:
- HOW MANY agents. Use as many as the work genuinely needs and no more. A simple
  question needs one. A delivery pipeline may need five or six. Do not pad the
  team with agents that only pass work along.
- WHAT each one does, and what it hands to the next. Each agent runs once, in
  order, and receives the previous agent's output.
- HOW MUCH each may iterate. `max_iterations` is how many times an agent may use
  a skill and reconsider before it must answer. There is no loop *between*
  agents, so an agent that must produce, check and correct its own work needs a
  larger budget — 10 or more. One that summarises once needs 3 or 4.

Field rules:
- "name": one or two words, unique within the team.
- "role": who this agent is and how it works, in one sentence.
- "objective": what THIS agent must achieve. Use {{goal}} for the user's request
  and {{previous}} for the previous agent's output. The first agent has no
  {{previous}}.
- "skills": capability names this agent needs, lower_snake_case. Prefer the ones
  listed as available. Anything else you name will be built for it.
- "rationale": why this agent exists, for a human reviewing the design.

Checking beats recalling. Wherever a fact the work depends on could be looked up
with one of the available capabilities — searching, fetching, reading a
repository or a document — give that agent the capability and say in its
objective that it must check rather than assume. An agent with no way to look
anything up can only restate what the model already believed, which is the wrong
answer whenever the goal turns on something specific or current.

Between 1 and {max_agents} agents. Return only the JSON object.
""".replace("{max_agents}", str(MAX_AGENTS))


def _plan_with_model(
    goal: str,
    answers: dict[str, str],
    connectors: list,
    *,
    hint: blueprints.Intent | None,
    use_documents: bool,
) -> tuple[list[PlannedAgent], Budgets] | None:
    """Ask the model to design the team for this goal.

    The primary path. The structure of a workflow — how many agents, what each
    does, how much each may iterate — is a property of the goal, and cannot be
    decided before reading it. Fixing it in advance and asking the model only to
    reword produced exactly the failure that made this change necessary: a
    request for a software organisation staffed as three writers, because the
    blueprint said three.

    A blueprint the classifier is confident about is offered as a hint, not a
    cage: a shape that usually works for this kind of request is worth knowing,
    and the model may use more agents, fewer, or entirely different ones.

    Returns None when there is no usable model or the proposal cannot be
    validated, and the caller falls back to the blueprint — so the platform still
    designs something sensible with no model at all.
    """
    prompt = _plan_prompt(goal, answers, connectors, hint=hint, use_documents=use_documents)

    try:
        completion = get_provider().complete(
            prompt,
            system=PLAN_SYSTEM,
            options=GenerationOptions(max_tokens=3000, temperature=0.2, effort="high"),
        )
    except LLMProxyError:
        return None

    payload = _extract_json(completion.text) or {}
    raw = payload.get("agents")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_AGENTS:
        return None

    planned: list[PlannedAgent] = []
    budgets: Budgets = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            return None
        agent = _validated_agent(item, index)
        if agent is None:
            return None
        planned.append(agent)
        budgets[agent.name] = _budget(item.get("max_iterations"))

    # Two agents with the same name would collide when stored and would make the
    # canvas ambiguous.
    if len({agent.name for agent in planned}) != len(planned):
        return None
    return planned, budgets


def _plan_prompt(
    goal: str,
    answers: dict[str, str],
    connectors: list,
    *,
    hint: blueprints.Intent | None,
    use_documents: bool,
) -> str:
    """Everything the model needs to decide a structure.

    Notably including what capabilities already exist. A model that knows the
    workspace has `repo_create_pull_request` designs an agent that opens pull
    requests; one that does not describes opening one.
    """
    parts = [f"Goal:\n{goal}\n"]

    stated = {key: value for key, value in answers.items() if value}
    if stated:
        parts.append(f"What the user said about it:\n{json.dumps(stated, indent=2)}\n")

    builtin = ", ".join(skill.name for skill in skill_registry.BUILTIN_SKILLS)
    parts.append(f"Capabilities always available:\n{builtin}\n")

    if connectors:
        tools = ", ".join(sorted({skill.name for skill in connectors})[:30])
        parts.append(
            "Tools connected to this workspace, which agents can be given:\n"
            f"{tools}\n"
        )
        # Named separately because they are the ones an agent can be pointed at
        # freely: they cannot change anything, so an agent that should establish
        # facts can be given them without the design demanding a write scope.
        research = ", ".join(
            sorted({skill.name for skill in connectors if skill.definition.get("read_only")})[:20]
        )
        if research:
            parts.append(
                "Of those, these only read, so any agent that needs to establish "
                f"facts can be given them:\n{research}\n"
            )
    else:
        parts.append("No external systems are connected to this workspace.\n")

    if use_documents:
        parts.append(
            "The user has uploaded documents. An agent that should consult them "
            "can ask for the skill `search_my_documents`.\n"
        )

    if hint is not None:
        shape = " → ".join(template.name for template in hint.agents)
        parts.append(
            f"This reads like: {hint.label.lower()}. A shape that often works for "
            f"that is {shape}. Use it only if it genuinely fits — use more agents, "
            "fewer, or different ones as this goal actually requires.\n"
        )

    return "\n".join(parts)


#: Iteration budgets the model asked for, keyed by agent name.
#:
#: Carried beside the plan rather than on `PlannedAgent`, which is a published
#: contract: the budget is how the graph is built, not part of the design a
#: caller is promised, and the stored `AgentDefinition` records the real value.
Budgets = dict[str, int]


def _validated_agent(item: dict, index: int) -> PlannedAgent | None:
    """One proposed agent, or None if it is not usable.

    Rejecting the whole plan on one bad agent is deliberate: half a design is not
    a design, and the blueprint fallback produces something coherent.
    """
    name = str(item.get("name") or "").strip()[:60]
    role = str(item.get("role") or "").strip()[:500]
    objective = str(item.get("objective") or "").strip()[:2000]
    if not name or not role or not objective:
        return None

    raw_skills = item.get("skills")
    skills = [
        _slug(str(skill))
        for skill in (raw_skills if isinstance(raw_skills, list) else [])
        if str(skill).strip()
    ][:8]

    # An objective that references nothing it will be given renders as literal
    # braces in the prompt. The first agent gets the goal; the rest also get
    # their predecessor's output.
    if "{{goal}}" not in objective and "{{previous}}" not in objective:
        objective = f"{objective}\n\nThe request:\n{{{{goal}}}}"
    if index > 0 and "{{previous}}" not in objective:
        objective = f"{objective}\n\nThe previous agent produced:\n{{{{previous}}}}"

    return PlannedAgent(
        name=name,
        role=role,
        objective=objective,
        skills=skills or ["summarise"],
        rationale=str(item.get("rationale") or "Proposed for this goal.").strip()[:1000],
    )


def _budget(raw: object) -> int:
    """An agent's iteration budget, clamped to something a person would accept.

    The model is asked for this because it is a structural decision — an agent
    that must produce, check and correct needs more attempts than one that
    summarises — but an unbounded value is somebody's bill.
    """
    try:
        wanted = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 6
    return max(1, min(20, wanted))


def _slug(text: str, limit: int = 48) -> str:
    words = "".join(c if c.isalnum() else " " for c in text.lower()).split()[:5]
    slug = "_".join(words)[:limit].strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"skill_{slug}" if slug else "generated_skill"
    return slug


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


# ---------------------------------------------------------------------------
# connected systems
# ---------------------------------------------------------------------------


def _connector_skills(tenant_id: str) -> list:
    """Skills that came from a connected MCP server, for agents that need one."""
    from cwap_contracts.v4 import SkillOrigin  # noqa: PLC0415

    return [
        skill
        for skill in skill_registry.list_all(tenant_id)
        if skill.origin is SkillOrigin.MCP and not skill.definition.get("unavailable")
    ]


#: A connector skill is offered to at most this many agents' worth of tools, so
#: one busy server cannot bury an agent's own capabilities in its tool list.
MAX_CONNECTOR_SKILLS = 8


def _attach_connectors(
    intent: blueprints.Intent,
    planned: list[PlannedAgent],
    resolved: dict[str, list[str]],
    connectors: list,
    gaps: list[SkillGap],
    model_proposed: bool,
) -> None:
    """Give agents that need to touch another system the tools to do it.

    An agent whose job is to open a pull request cannot do it with a prompt. If
    the tenant has connected a server offering the right tools, they are
    attached; if not, that is reported as a gap — a workflow that looks complete
    and cannot reach anything is worse than one that says what is missing.
    """
    templates = {template.name: template for template in intent.agents}

    for agent in planned:
        template = templates.get(agent.name)
        if template is not None:
            wanted, writes = template.wants_connectors, template.connector_writes
        elif model_proposed:
            # A model-designed team has names nothing here anticipated, so what
            # it needs is read from what it *said* — the skills it asked for and
            # the job it described.
            wanted, writes = _inferred_connector_need(agent)
        else:
            continue

        if not wanted:
            continue

        matched = _matching_connectors(connectors, wanted, writes=writes)
        if matched:
            resolved.setdefault(agent.name, [])
            for skill in matched[:MAX_CONNECTOR_SKILLS]:
                if skill.id not in resolved[agent.name]:
                    resolved[agent.name].append(skill.id)
                    agent.skills.append(skill.name)
        elif wanted:
            gaps.append(
                SkillGap(
                    needed_by=agent.name,
                    capability=f"access to {wanted[0]}",
                    blocked_reason=(
                        f"{agent.name} needs to reach a system this workspace has "
                        "not connected. Add an MCP server under Connected systems "
                        "and re-run the design, or this agent will describe the "
                        "work instead of doing it."
                    ),
                )
            )


def _note_write_scope(
    planned: list[PlannedAgent],
    resolved: dict[str, list[str]],
    tenant_id: str,
    gaps: list[SkillGap],
) -> None:
    """Say up front when this design will need permission to act outward.

    The design knows it just attached a tool that can change something outside
    the platform, and the run authoriser will refuse without the WRITE_EXTERNAL
    scope. Discovering that after placing five agents on a canvas and pressing
    Run is the same mistake as a template that fails on its third step: the
    prerequisite is knowable now, so it is reported now.
    """
    from cwap_contracts.v4 import SIDE_EFFECTING_KINDS  # noqa: PLC0415

    for agent in planned:
        writers = [
            skill
            for skill in skill_registry.get_many(tenant_id, resolved.get(agent.name, []))
            if skill.kind in SIDE_EFFECTING_KINDS and not skill.definition.get("read_only")
        ]
        if not writers:
            continue

        gaps.append(
            SkillGap(
                needed_by=agent.name,
                capability=f"permission to use {writers[0].name}",
                blocked_reason=(
                    f"{agent.name} can change things outside the platform "
                    f"({', '.join(skill.name for skill in writers[:3])}), so running "
                    "this workflow needs the WRITE_EXTERNAL scope. Ask an "
                    "administrator to grant it — everything else will run without it."
                ),
            )
        )
        return


def _matching_connectors(
    connectors: list, wanted: tuple[str, ...], *, writes: bool = True
) -> list:
    """Connector skills whose tool name or description mentions what is wanted.

    An agent that only reads is offered only the tools the server itself marks
    read-only. Least privilege, and it keeps the workflow's permission
    requirement honest: a five-agent pipeline should not demand a write scope
    because a reviewer was handed the ability to merge.
    """
    matched = []
    for skill in connectors:
        if not writes and not skill.definition.get("read_only"):
            continue
        haystack = f"{skill.name} {skill.description}".lower()
        if any(keyword == "" or keyword.lower() in haystack for keyword in wanted):
            matched.append(skill)
    return matched


#: Words in a proposed agent's own description that mean it has to touch
#: something outside the platform, and whether doing so changes anything there.
_READ_SIGNALS = (
    "repository", "repo", "codebase", "source", "file", "read", "search",
    "inspect", "review", "fetch", "look up",
    # An agent whose job is to find things out is useless without something to
    # find them out *with*, so researching counts as needing the outside world.
    "research", "investigate", "verify", "check", "gather", "find out",
    "documentation", "evidence",
)
_WRITE_SIGNALS = (
    "pull request", "merge", "commit", "push", "branch", "deploy", "publish",
    "create", "write", "update", "open a pr", "land",
)


def _inferred_connector_need(agent: PlannedAgent) -> tuple[tuple[str, ...], bool]:
    """What a model-designed agent needs from connected systems.

    Read from what the model said about the agent, because the platform has no
    template for a role it did not write. An agent that describes opening a pull
    request is offered the tools to open one; one that only reads is offered only
    read tools, so a reviewer does not end up able to merge.
    """
    haystack = f"{agent.role} {agent.objective} {' '.join(agent.skills)}".lower()

    writes = any(signal in haystack for signal in _WRITE_SIGNALS)
    needs = writes or any(signal in haystack for signal in _READ_SIGNALS)
    if not needs:
        return (), False

    # No keyword filter: the agent said it needs the outside world, and the
    # platform has no basis for guessing which of the tenant's tools it means.
    return ("",), writes


def _ensure_retrieval_skill(tenant_id: str, handle: str):
    """A search capability over one corpus, created once and reused."""
    from cwap_contracts.v4 import SkillKind, SkillParameter, SkillProposal  # noqa: PLC0415

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
    budgets: Budgets,
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
        stored = _agent_for(tenant_id, agent, answers, resolved, budgets)
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
    """The objective as the node will render it.

    Composed from the bindings the node actually receives, and only appending
    what the objective does not already reference — some templates embed
    `{{goal}}` and `{{previous}}` where they read best, and blindly appending
    would hand the model the same text twice.
    """
    objective = agent.objective
    if "{{goal}}" not in objective:
        objective = f"{objective}\n\nThe goal is:\n{{{{goal}}}}"
    if index > 1 and "{{previous}}" not in objective:
        objective = f"{objective}\n\nThe previous agent produced:\n{{{{previous}}}}"
    return objective


def _agent_for(
    tenant_id: str,
    planned: PlannedAgent,
    answers: dict[str, str],
    resolved: dict[str, list[str]],
    budgets: Budgets,
):
    """The stored agent this step will run — reused where one already fits.

    Creating unconditionally is what made a workspace fill with near-duplicates:
    every design minted a new "Researcher", each with its own empty memory, and
    the roster grew by the size of the team on every plan. That is the opposite
    of the promise — an agent is supposed to get better across runs, and it
    cannot if each run gets a fresh one.

    So a name that already exists is reused, and given whatever skills this plan
    asks for on top of what it already had. What is deliberately *not* touched
    is its role: an existing agent's description is something a person may have
    edited, and a planner silently rewriting it would undo that edit without
    saying so. A reused agent keeps its identity and gains a capability.
    """
    wanted = resolved.get(planned.name, [])
    existing = agent_registry.find_by_name(tenant_id, planned.name)
    if existing is None:
        return agent_registry.create(
            tenant_id,
            name=planned.name,
            role=planned.role,
            objective=planned.objective,
            instructions=_instructions(answers),
            skill_ids=wanted,
            # The graph is acyclic, so an agent that must write, check and
            # correct needs the budget to do it inside its own turn.
            max_iterations=budgets.get(planned.name, 6),
            memory=AgentMemoryConfig(),
        )

    gained = [skill_id for skill_id in wanted if skill_id not in existing.skill_ids]
    if not gained:
        return existing
    return agent_registry.update(
        tenant_id, existing.id, skill_ids=[*existing.skill_ids, *gained]
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


def _notes(
    planned: list[PlannedAgent],
    gaps: list[SkillGap],
    use_documents: bool,
    model_proposed: bool = False,
) -> list[str]:
    notes = [
        f"{len(planned)} agent(s), each with its own memory. They improve across runs.",
    ]
    reused = [agent.name for agent in planned if agent.reused]
    if reused:
        # Worth saying plainly: these arrive with everything they have already
        # learned, which is the reason reuse beats a fresh copy.
        notes.append(
            f"Reusing {len(reused)} agent(s) you already have — "
            + ", ".join(reused)
            + " — so they bring what they have learned with them."
        )
    if model_proposed:
        notes.append(
            "This goal did not match a known shape, so the team was proposed for it. "
            "Re-running the design may produce a different one — adjust it here and "
            "save it once it looks right."
        )
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
