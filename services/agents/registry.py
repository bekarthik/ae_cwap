"""Agent storage.

Agents are stored, not embedded in graphs, so improving one improves every
workflow that uses it — and so its memory has a stable identity to accumulate
against. An agent embedded in a node would start from zero in every workflow.
"""

from __future__ import annotations

import uuid

from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import Agent as AgentRow
from cwap_contracts.v2 import AgentDefinition, AgentMemoryConfig


class AgentNotFound(LookupError):
    """The referenced agent does not exist for this tenant."""


def create(
    tenant_id: str,
    *,
    name: str,
    role: str,
    objective: str = "",
    instructions: str = "",
    skill_ids: list[str] | None = None,
    max_iterations: int = 6,
    model_override: str | None = None,
    memory: AgentMemoryConfig | None = None,
) -> AgentDefinition:
    agent = AgentDefinition(
        id=f"agt_{uuid.uuid4().hex[:20]}",
        tenant_id=tenant_id,
        name=name,
        role=role,
        objective=objective,
        instructions=instructions,
        skill_ids=list(skill_ids or []),
        max_iterations=max_iterations,
        model_override=model_override,
        memory=memory or AgentMemoryConfig(),
    )
    with unit_of_work() as session:
        session.add(_to_row(agent))
    return agent


def update(tenant_id: str, agent_id: str, **changes) -> AgentDefinition:
    with unit_of_work() as session:
        row = session.query(AgentRow).filter_by(tenant_id=tenant_id, id=agent_id).one_or_none()
        if row is None:
            raise AgentNotFound(agent_id)

        for key in ("name", "role", "objective", "instructions", "max_iterations", "model_override"):
            if key in changes and changes[key] is not None:
                setattr(row, key, changes[key])
        if changes.get("skill_ids") is not None:
            row.skill_ids = list(changes["skill_ids"])
        if changes.get("memory") is not None:
            row.memory_config = changes["memory"].model_dump(mode="json")

        # Bumped on every edit, so a run report can say which version of an
        # agent produced a result.
        row.version += 1
        return _to_definition(row)


def get(tenant_id: str, agent_id: str) -> AgentDefinition:
    with read_only_session() as session:
        row = session.query(AgentRow).filter_by(tenant_id=tenant_id, id=agent_id).one_or_none()
        if row is None:
            raise AgentNotFound(
                f"agent '{agent_id}' does not exist for tenant '{tenant_id}'"
            )
        return _to_definition(row)


def find_by_name(tenant_id: str, name: str) -> AgentDefinition | None:
    with read_only_session() as session:
        row = session.query(AgentRow).filter_by(tenant_id=tenant_id, name=name).one_or_none()
        return _to_definition(row) if row else None


def list_all(tenant_id: str) -> list[AgentDefinition]:
    with read_only_session() as session:
        rows = (
            session.query(AgentRow).filter_by(tenant_id=tenant_id).order_by(AgentRow.name).all()
        )
        return [_to_definition(row) for row in rows]


def delete(tenant_id: str, agent_id: str) -> bool:
    with unit_of_work() as session:
        return bool(
            session.query(AgentRow).filter_by(tenant_id=tenant_id, id=agent_id).delete()
        )


def _to_row(agent: AgentDefinition) -> AgentRow:
    return AgentRow(
        id=agent.id,
        tenant_id=agent.tenant_id,
        name=agent.name,
        role=agent.role,
        objective=agent.objective,
        instructions=agent.instructions,
        skill_ids=list(agent.skill_ids),
        max_iterations=agent.max_iterations,
        memory_config=agent.memory.model_dump(mode="json"),
        model_override=agent.model_override,
        version=agent.version,
    )


def _to_definition(row: AgentRow) -> AgentDefinition:
    return AgentDefinition(
        id=row.id,
        tenant_id=row.tenant_id,
        name=row.name,
        role=row.role,
        objective=row.objective or "",
        instructions=row.instructions or "",
        skill_ids=list(row.skill_ids or []),
        max_iterations=row.max_iterations,
        memory=AgentMemoryConfig.model_validate(row.memory_config or {}),
        model_override=row.model_override,
        version=row.version,
    )
