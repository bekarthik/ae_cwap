"""The whole workspace as one graph.

Every other endpoint answers a question about one kind of thing: what agents do I
have, what skills, which servers are connected. None of them answers the question
a person actually has when they open the product — *what is in here, and what is
wired to what*. The pieces are already related to each other (an agent holds
skills, a skill came from a connected server, a workflow staffs its steps with
agents, a memory belongs to one of them); those relationships simply had no
endpoint that showed them together.

So this returns nodes and links, deliberately shaped for a renderer rather than
for a form: an id, a kind, a label, a size that means "how much has this been
used", and edges between them. What draws it is not this module's business — the
same payload feeds a 3D map, a list, or a diagnostic in a terminal.

Nothing here is authoritative. It is a read-only projection assembled from the
registries that own each kind, which is why it can be cheap and why it can be
wrong for a moment without anything breaking.
"""

from __future__ import annotations

from typing import Any

from agents import registry as agent_registry
from cwap_common.db import read_only_session
from cwap_common.models import Memory as MemoryRow
from cwap_common.models import Workflow as WorkflowRow
from cwap_contracts.v4 import SkillOrigin, WorkflowGraph
from fastapi import APIRouter, Depends
from mcp_connect import registry as mcp_registry
from skills import registry as skill_registry

from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/workspace", tags=["workspace"])

#: What a node's size means, by kind. Everything is "how much has this been
#: used", because the interesting thing about a workspace is where the work is.
_BASE_SIZE = 1.0


@router.get("/map")
def workspace_map(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    """Everything in this workspace, and what connects to what."""
    tenant = principal.tenant_id

    agents = agent_registry.list_all(tenant)
    skills = skill_registry.list_all(tenant)
    servers = mcp_registry.list_all(tenant)
    workflows = _workflows(tenant)
    memories = _memory_counts(tenant)

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    for agent in agents:
        nodes.append(
            {
                "id": agent.id,
                "kind": "agent",
                "label": agent.name,
                "detail": agent.role[:160],
                "size": _BASE_SIZE + 0.25 * len(agent.skill_ids),
                "memories": memories.get(("agent", agent.id), 0),
            }
        )
        for skill_id in agent.skill_ids:
            links.append({"source": agent.id, "target": skill_id, "kind": "holds"})
        if agent.model_provider:
            links.append(
                {"source": agent.id, "target": f"provider:{agent.model_provider}", "kind": "runs_on"}
            )

    for skill in skills:
        nodes.append(
            {
                "id": skill.id,
                "kind": "skill",
                "label": skill.name.replace("_", " "),
                "detail": skill.description[:160],
                # Invocations, so a skill everything leans on is visibly bigger
                # than one nobody has called.
                "size": _BASE_SIZE + min(skill.invocations, 40) * 0.05,
                "memories": memories.get(("skill", skill.id), 0),
            }
        )
        if skill.origin is SkillOrigin.MCP:
            server_id = (skill.definition or {}).get("server_id")
            if server_id:
                links.append({"source": skill.id, "target": server_id, "kind": "from"})

    for server in servers:
        nodes.append(
            {
                "id": server.id,
                "kind": "server",
                "label": server.name,
                "detail": server.description[:160] or server.config.transport.value,
                "size": _BASE_SIZE + 0.1 * len(server.tools),
                "memories": 0,
            }
        )

    for provider in sorted({agent.model_provider for agent in agents if agent.model_provider}):
        nodes.append(
            {
                "id": f"provider:{provider}",
                "kind": "provider",
                "label": provider,
                "detail": "model backend",
                "size": _BASE_SIZE,
                "memories": 0,
            }
        )

    for workflow_id, name, graph in workflows:
        nodes.append(
            {
                "id": workflow_id,
                "kind": "workflow",
                "label": name,
                "detail": f"{len(graph.nodes)} step(s)",
                "size": _BASE_SIZE + 0.15 * len(graph.nodes),
                "memories": memories.get(("workflow", graph.memory_scope), 0),
            }
        )
        for node in graph.nodes:
            if node.agent_id:
                links.append({"source": workflow_id, "target": node.agent_id, "kind": "staffs"})
            if node.review is not None:
                links.append(
                    {"source": workflow_id, "target": node.review.agent_id, "kind": "reviews"}
                )

    known = {node["id"] for node in nodes}
    # A link to something that is not here would be an edge into empty space —
    # a skill deleted while a workflow still mentions it, most often.
    links = [link for link in links if link["source"] in known and link["target"] in known]

    return {
        "nodes": nodes,
        "links": links,
        "counts": {
            "agents": len(agents),
            "skills": len(skills),
            "servers": len(servers),
            "workflows": len(workflows),
            "memories": sum(memories.values()),
        },
    }


def _workflows(tenant_id: str) -> list[tuple[str, str, WorkflowGraph]]:
    found: list[tuple[str, str, WorkflowGraph]] = []
    with read_only_session() as session:
        rows = session.query(WorkflowRow).filter_by(tenant_id=tenant_id).all()
        for row in rows:
            try:
                graph = WorkflowGraph.model_validate(row.graph)
            except Exception:  # noqa: BLE001 - a stored graph that will not parse
                continue
            found.append((row.id, row.name, graph))
    return found


def _memory_counts(tenant_id: str) -> dict[tuple[str, str], int]:
    """How much each thing has learned, keyed by (scope, scope id)."""
    with read_only_session() as session:
        rows = session.query(MemoryRow.scope, MemoryRow.scope_id).filter_by(
            tenant_id=tenant_id
        )
        counts: dict[tuple[str, str], int] = {}
        for scope, scope_id in rows:
            key = (str(scope), str(scope_id))
            counts[key] = counts.get(key, 0) + 1
        return counts
