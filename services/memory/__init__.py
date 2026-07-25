"""Memory — what the platform carries forward, at three scopes.

Workflow memory answers "what do I know about this job?", agent memory answers
"what have I learned about my role?", and skill memory answers "what have I
learned about using this capability?". Keeping them separate is what makes a
lesson transferable: a lesson about phrasing a retrieval query belongs to the
skill, so every agent that uses it benefits.
"""

from memory.service import (
    DEDUPE_THRESHOLD,
    DEFAULT_CAPACITY,
    MIN_USEFULNESS,
    forget,
    forget_entry,
    list_memories,
    recall,
    reinforce,
    remember,
)

__all__ = [
    "DEDUPE_THRESHOLD",
    "DEFAULT_CAPACITY",
    "MIN_USEFULNESS",
    "forget",
    "forget_entry",
    "list_memories",
    "recall",
    "reinforce",
    "remember",
]
