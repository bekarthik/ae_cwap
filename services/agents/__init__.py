"""Agents — a role, the skills it can use, a loop, and memory.

The loop is the difference from a step. A step calls a model once; an agent
decides, invokes a skill, reads the result, and decides again, until it judges
the objective met or its iteration budget runs out. Afterwards it reflects, and
what it learned is waiting in the prompt on the next run.
"""

from agents.registry import (
    AgentNotFound,
    create,
    delete,
    find_by_name,
    get,
    list_all,
    update,
)
from agents.runtime import (
    BUDGET_EXHAUSTED,
    OBJECTIVE_MET,
    AgentError,
    AgentResult,
    AgentRunContext,
    remember_for_workflow,
    run_agent,
)

__all__ = [
    "BUDGET_EXHAUSTED",
    "OBJECTIVE_MET",
    "AgentError",
    "AgentNotFound",
    "AgentResult",
    "AgentRunContext",
    "create",
    "delete",
    "find_by_name",
    "get",
    "list_all",
    "remember_for_workflow",
    "run_agent",
    "update",
]
