"""Intelligent Requirement Diagnosis & Setup (Epic 1).

User Story 2: the user types a high-level goal ("Plan my weekend trip to
Denver") and the system decides which skills the goal needs and hands back a
draft workflow to review on the canvas.

Two properties this module is built around:

* **Deterministic.** The same goal always produces the same plan, so onboarding
  is testable and a user who retypes their goal does not get a different graph.
* **Honest about gaps.** A skill it recognises but cannot wire up — a corpus
  that has not been uploaded, an API credential it does not have — is reported
  as a `gap`, never scaffolded as a node that would fail at run time.
"""

from __future__ import annotations

import uuid

from cwap_contracts import (
    DiagnosisResult,
    GoalIntakeRequest,
    NodeType,
    Position,
    ScaffoldResponse,
    SkillRequirement,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
)
from nlp import skills as catalogue
from nlp.skills import Skill

#: Node types that produce prose and can be chained.
_CONTENT_SKILLS = frozenset({"research", "plan", "draft"})

#: Cap on chained LLM steps in a scaffold. More than this stops being a helpful
#: starting point and starts being a graph the user has to prune.
MAX_CONTENT_STEPS = 2

#: Canvas layout constants — the draft should open readable, not as a pile.
_X_STEP = 260
_Y_BASE = 120
_Y_BRANCH_OFFSET = 130


def diagnose(request: GoalIntakeRequest) -> DiagnosisResult:
    """Classify intent and decide which skills the goal requires."""
    matches = catalogue.match(request.goal)

    required: list[SkillRequirement] = []
    suggestions: list[str] = []
    gaps: list[str] = []

    for skill, _hits in matches:
        if skill.requires and not _can_satisfy(skill, request):
            gaps.append(f"{skill.label}: needs {skill.requires}")
            continue
        required.append(
            SkillRequirement(
                skill=skill.key, node_type=skill.node_type, rationale=skill.rationale
            )
        )

    content = [req for req in required if req.skill in _CONTENT_SKILLS]
    if not content:
        # Every goal needs at least one reasoning step, even when no keyword hit.
        required.insert(
            0,
            SkillRequirement(
                skill="research",
                node_type=NodeType.LLM,
                rationale="No specific skill matched, so the goal is handled as a single "
                "reasoning step you can refine.",
            ),
        )

    if request.knowledge_handles and not any(
        req.node_type is NodeType.RAG_RETRIEVE for req in required
    ):
        suggestions.append(
            "You have uploaded documents — link a Knowledge Context node if the answer "
            "should be grounded in them."
        )
    if not any(req.node_type is NodeType.BRANCH for req in required):
        suggestions.append(
            "Add a Decision node if this workflow should take different paths depending "
            "on a result."
        )
    suggestions.append("Review each step's parameters before running — they are drafts.")

    return DiagnosisResult(
        intent=_intent_label(matches, request.goal),
        required_steps=required,
        suggestions=suggestions,
        confidence=_confidence(matches),
        gaps=gaps,
    )


def scaffold(request: GoalIntakeRequest) -> ScaffoldResponse:
    """Diagnose the goal and build the draft graph the canvas opens with."""
    diagnosis = diagnose(request)
    graph = build_graph(diagnosis, request)
    return ScaffoldResponse(diagnosis=diagnosis, graph=graph)


def build_graph(diagnosis: DiagnosisResult, request: GoalIntakeRequest) -> WorkflowGraph:
    """Turn a diagnosis into an executable draft.

    The result is a real `WorkflowGraph`, so it goes through the same structural
    validation as a hand-built one: if the scaffolder ever emitted something
    unexecutable, construction would raise here rather than at run time.
    """
    nodes: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []
    column = 0

    def place() -> Position:
        nonlocal column
        position = Position(x=float(column * _X_STEP), y=float(_Y_BASE))
        column += 1
        return position

    nodes.append(
        WorkflowNode(
            id="input",
            type=NodeType.INPUT,
            label="Goal",
            params={"fields": ["goal"], "defaults": {"goal": request.goal}},
            position=place(),
        )
    )
    previous = "input"

    wanted = {req.skill for req in diagnosis.required_steps}

    if NodeType.RAG_RETRIEVE in {req.node_type for req in diagnosis.required_steps}:
        skill = catalogue.by_key("ground_in_documents")
        nodes.append(
            WorkflowNode(
                id="knowledge",
                type=NodeType.RAG_RETRIEVE,
                label="Knowledge Context",
                params=dict(skill.default_params),
                position=place(),
                knowledge_handle=request.knowledge_handles[0],
            )
        )
        edges.append(
            WorkflowEdge(
                id="e_input_knowledge",
                source=previous,
                target="knowledge",
                bindings={"query": "$run.input.goal"},
            )
        )
        previous = "knowledge"

    content_steps = [
        req for req in diagnosis.required_steps if req.skill in _CONTENT_SKILLS
    ][:MAX_CONTENT_STEPS]

    for index, requirement in enumerate(content_steps, start=1):
        skill = catalogue.by_key(requirement.skill)
        node_id = f"step_{index}"
        nodes.append(
            WorkflowNode(
                id=node_id,
                type=NodeType.LLM,
                label=skill.label,
                params=_llm_params(skill, index),
                position=place(),
            )
        )
        edges.append(
            WorkflowEdge(
                id=f"e_{previous}_{node_id}",
                source=previous,
                target=node_id,
                bindings=_content_bindings(previous),
            )
        )
        previous = node_id

    if "format" in wanted:
        skill = catalogue.by_key("format")
        nodes.append(
            WorkflowNode(
                id="format",
                type=NodeType.TRANSFORM,
                label=skill.label,
                params={"template": "{{previous}}"},
                position=place(),
            )
        )
        edges.append(
            WorkflowEdge(
                id=f"e_{previous}_format",
                source=previous,
                target="format",
                bindings={"previous": "$output.text"},
            )
        )
        previous = "format"

    if "classify" in wanted:
        nodes.extend(_branch_nodes(column))
        edges.extend(_branch_edges(previous))
    else:
        nodes.append(
            WorkflowNode(
                id="output",
                type=NodeType.OUTPUT,
                label="Result",
                params={"result_template": "{{previous}}"},
                position=place(),
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
        id=f"wf_{uuid.uuid4().hex[:16]}",
        name=_workflow_name(request.goal),
        nodes=nodes,
        edges=edges,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _can_satisfy(skill: Skill, request: GoalIntakeRequest) -> bool:
    if skill.node_type is NodeType.RAG_RETRIEVE:
        return bool(request.knowledge_handles)
    # Anything else with a `requires` needs a credential we cannot invent.
    return False


def _intent_label(matches: list[tuple[Skill, int]], goal: str) -> str:
    if not matches:
        return "general_assistance"
    top = matches[0][0]
    return f"{top.key}:{_slug(goal)}"


def _slug(goal: str, words: int = 5) -> str:
    tokens = [tok for tok in goal.lower().split() if tok.isalnum()][:words]
    return "_".join(tokens) or "goal"


def _confidence(matches: list[tuple[Skill, int]]) -> float:
    """Rough but monotonic: more distinct skills matched, more confident.

    Capped below 1.0 — a keyword planner should never claim certainty.
    """
    if not matches:
        return 0.25
    distinct = len(matches)
    total_hits = sum(hits for _, hits in matches)
    return round(min(0.95, 0.4 + 0.1 * distinct + 0.02 * total_hits), 2)


def _llm_params(skill: Skill, index: int) -> dict[str, object]:
    params = dict(skill.default_params)
    if index > 1:
        params["prompt_template"] = (
            "Build on the previous step's result.\n\n"
            "Previous result:\n{{previous}}\n\nOriginal goal:\n{{goal}}"
        )
    return params


def _content_bindings(previous: str) -> dict[str, str]:
    if previous == "input":
        return {"goal": "$run.input.goal"}
    if previous == "knowledge":
        return {"goal": "$run.input.goal", "context": "$output.context"}
    return {"goal": "$run.input.goal", "previous": "$output.text"}


def _branch_nodes(column: int) -> list[WorkflowNode]:
    return [
        WorkflowNode(
            id="decide",
            type=NodeType.BRANCH,
            label="Decision",
            params={"left": "$output.text", "operator": "contains", "right": ""},
            position=Position(x=float(column * _X_STEP), y=float(_Y_BASE)),
        ),
        WorkflowNode(
            id="output",
            type=NodeType.OUTPUT,
            label="Result (condition met)",
            params={"result_template": "{{previous}}"},
            position=Position(
                x=float((column + 1) * _X_STEP), y=float(_Y_BASE - _Y_BRANCH_OFFSET)
            ),
        ),
        WorkflowNode(
            id="output_alt",
            type=NodeType.OUTPUT,
            label="Result (condition not met)",
            params={"result_template": "{{previous}}"},
            position=Position(
                x=float((column + 1) * _X_STEP), y=float(_Y_BASE + _Y_BRANCH_OFFSET)
            ),
        ),
    ]


def _branch_edges(previous: str) -> list[WorkflowEdge]:
    return [
        WorkflowEdge(
            id=f"e_{previous}_decide",
            source=previous,
            target="decide",
            bindings={"previous": "$output.text"},
        ),
        WorkflowEdge(
            id="e_decide_true",
            source="decide",
            target="output",
            condition=True,
            bindings={"previous": "$steps.decide.evaluated"},
        ),
        WorkflowEdge(
            id="e_decide_false",
            source="decide",
            target="output_alt",
            condition=False,
            bindings={"previous": "$steps.decide.evaluated"},
        ),
    ]


def _workflow_name(goal: str) -> str:
    condensed = " ".join(goal.split())
    return condensed[:60] + ("…" if len(condensed) > 60 else "")
