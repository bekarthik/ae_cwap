"""Skills — the capabilities an agent can invoke.

A skill is declarative, never code: a prompt template, a retrieval over a corpus
the tenant owns, a text transform, an allow-listed HTTP call, or a composition of
those. That boundary is what makes it safe for the platform to synthesise a
capability an agent needs but does not have.

Each skill carries its own memory, so a lesson about using the capability accrues
to the skill and every agent that reaches for it inherits the lesson.
"""

from skills.execution import (
    MAX_SKILL_OUTPUT_CHARS,
    SkillContext,
    SkillExecutionError,
    SkillOutcome,
    execute,
    recall_skill_memory,
    teach_skill,
)
from skills.registry import (
    BUILTIN_SKILLS,
    SkillError,
    create,
    delete,
    ensure_builtins,
    ensure_capability,
    find_by_name,
    get,
    get_many,
    list_all,
    record_invocation,
    synthesize,
)

__all__ = [
    "BUILTIN_SKILLS",
    "MAX_SKILL_OUTPUT_CHARS",
    "SkillContext",
    "SkillError",
    "SkillExecutionError",
    "SkillOutcome",
    "create",
    "delete",
    "ensure_builtins",
    "ensure_capability",
    "execute",
    "find_by_name",
    "get",
    "get_many",
    "list_all",
    "recall_skill_memory",
    "record_invocation",
    "synthesize",
    "teach_skill",
]
