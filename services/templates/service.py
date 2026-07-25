"""Turning a template into a workflow this tenant owns.

Instantiation is a copy, not a reference. Each agent is created for the tenant,
each skill is found or built, and the graph gets its own memory scope — so
editing a copied template changes your workflow and nobody else's, and the
agents accumulate their own memory from your runs.

It reuses the design service's graph shape deliberately. A workflow that came
from a template and one the system designed from a goal should be the same kind
of object, or half the product's behaviour would depend on where a workflow came
from.
"""

from __future__ import annotations

import uuid

from agents import registry as agent_registry
from cwap_contracts.v3 import (
    AgentMemoryConfig,
    NodeType,
    PlannedAgent,
    Position,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
)
from design import service as design_service
from pydantic import BaseModel, ConfigDict, Field
from skills import registry as skill_registry

from templates.catalogue import WorkflowTemplate, all_templates, find

_X_STEP = 300
_Y_BASE = 140


class TemplateNotFound(LookupError):
    """No such template."""


class TemplateSummary(BaseModel):
    """What the launcher shows before anything is created."""

    model_config = ConfigDict(extra="forbid")

    key: str
    title: str
    summary: str
    detail: str
    category: str
    input_label: str
    example_input: str
    agents: list[dict] = Field(default_factory=list)
    #: Prerequisites, phrased as things to set up rather than errors.
    requires: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class InstantiatedTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: WorkflowGraph
    agents: list[PlannedAgent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def catalogue(tenant_id: str) -> list[TemplateSummary]:
    """Every template, with its prerequisites checked against this tenant."""
    return [_summarise(template, tenant_id) for template in all_templates()]


def instantiate(key: str, tenant_id: str) -> InstantiatedTemplate:
    """Create this tenant's own copy of a template."""
    template = find(key)
    if template is None:
        raise TemplateNotFound(key)

    skill_registry.ensure_builtins(tenant_id)
    handles = _knowledge_handles(tenant_id)
    use_documents = bool(handles) and template.wants_documents

    planned: list[PlannedAgent] = [
        PlannedAgent(
            name=agent.name,
            role=agent.role,
            objective=agent.objective,
            skills=list(agent.skills),
            rationale=agent.rationale,
        )
        for agent in template.agents
    ]

    resolved, gaps = design_service._resolve_skills(  # noqa: SLF001 - one owner of this logic
        tenant_id,
        planned,
        template.title,
        handles if use_documents else [],
    )

    graph = _build_graph(template, planned, resolved, tenant_id)
    return InstantiatedTemplate(
        graph=graph,
        agents=planned,
        notes=_notes(template, gaps, use_documents, tenant_id),
    )


def _build_graph(
    template: WorkflowTemplate,
    planned: list[PlannedAgent],
    resolved: dict[str, list[str]],
    tenant_id: str,
) -> WorkflowGraph:
    workflow_id = f"wf_{uuid.uuid4().hex[:16]}"
    nodes: list[WorkflowNode] = [
        WorkflowNode(
            id="input",
            type=NodeType.INPUT,
            label="Start",
            params={"fields": ["goal"], "defaults": {"goal": template.example_input}},
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
                params={"objective_template": agent.objective},
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
        name=template.title,
        nodes=nodes,
        edges=edges,
        memory_scope_id=workflow_id,
    )


def _summarise(template: WorkflowTemplate, tenant_id: str) -> TemplateSummary:
    return TemplateSummary(
        key=template.key,
        title=template.title,
        summary=template.summary,
        detail=template.detail,
        category=template.category,
        input_label=template.input_label,
        example_input=template.example_input,
        agents=[
            {"name": agent.name, "rationale": agent.rationale, "skills": list(agent.skills)}
            for agent in template.agents
        ],
        requires=_requirements(template, tenant_id),
        tags=list(template.tags),
    )


def _requirements(template: WorkflowTemplate, tenant_id: str) -> list[str]:
    """What is missing before this template would do what it says.

    Reported up front rather than at run time: a template that quietly produces a
    workflow which fails on its third step is worse than one that says "upload
    something first".
    """
    missing: list[str] = []

    if template.wants_documents and not _knowledge_handles(tenant_id):
        missing.append("Upload a document — this one answers from what you give it.")

    if template.wants_connectors and not _has_any_connector(tenant_id):
        missing.extend(f"Connect {need}." for need in template.wants_connectors)

    return missing


def _knowledge_handles(tenant_id: str) -> list[str]:
    from knowledge.service import list_handles  # noqa: PLC0415 - avoid a cycle

    return [item.handle for item in list_handles(tenant_id)]


def _has_any_connector(tenant_id: str) -> bool:
    from mcp_connect import registry as mcp_registry  # noqa: PLC0415

    return any(server.enabled for server in mcp_registry.list_all(tenant_id))


def _notes(
    template: WorkflowTemplate, gaps: list, use_documents: bool, tenant_id: str
) -> list[str]:
    notes = [
        f"{len(template.agents)} agent(s), created for you. Editing them changes "
        "your copy and nothing else.",
    ]
    created = [gap for gap in gaps if not gap.blocked_reason]
    if created:
        notes.append(
            "Built " + ", ".join(gap.capability for gap in created) + " for this workflow."
        )
    if use_documents:
        notes.append("Agents can search your documents; they decide when it is worth it.")
    if template.wants_connectors:
        connectors = _connector_skill_names(tenant_id)
        if connectors:
            notes.append(
                "Give an agent one of your connected tools to let it reach out: "
                + ", ".join(connectors[:6])
            )
    return notes


def _connector_skill_names(tenant_id: str) -> list[str]:
    from cwap_contracts.v3 import SkillOrigin  # noqa: PLC0415

    return [
        skill.name
        for skill in skill_registry.list_all(tenant_id)
        if skill.origin is SkillOrigin.MCP
    ]
