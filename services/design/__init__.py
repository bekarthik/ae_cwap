"""Designing a workflow from a goal.

The system gathers what it needs, decides which agents the job requires, gives
each the skills it needs — creating any that do not exist — and wires up memory.
The user reviews a design, rather than being handed an empty canvas.
"""

from design.blueprints import GENERALIST, INTENTS, AgentTemplate, Intent, classify, restate
from design.service import MAX_AGENTS, MAX_QUESTIONS, design

__all__ = [
    "GENERALIST",
    "INTENTS",
    "MAX_AGENTS",
    "MAX_QUESTIONS",
    "AgentTemplate",
    "Intent",
    "classify",
    "design",
    "restate",
]
