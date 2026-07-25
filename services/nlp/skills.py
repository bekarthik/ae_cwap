"""The skill catalogue the diagnosis layer matches goals against.

Deliberately data, not code. Adding a capability the onboarding layer can
recognise and scaffold is a new entry here — no changes to the planner, the
graph builder, or the executor registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cwap_contracts import NodeType


@dataclass(frozen=True)
class Skill:
    """One capability the platform can recognise in a goal and scaffold for."""

    key: str
    label: str
    node_type: NodeType
    #: Lowercase trigger terms. A match contributes to intent classification.
    keywords: tuple[str, ...]
    rationale: str
    #: Seed params for the scaffolded node, shown to the user for review.
    default_params: dict[str, Any] = field(default_factory=dict)
    #: Opening line of a generated prompt. The rest of the prompt is composed
    #: from the variables the incoming edge actually binds, so a scaffolded
    #: template can never reference something that will not be there.
    instruction: str = ""
    #: Set when the skill cannot run until the user supplies something we
    #: cannot invent — surfaced as a `gap` rather than silently scaffolded.
    requires: str | None = None


CATALOGUE: tuple[Skill, ...] = (
    Skill(
        key="research",
        label="Research & summarise",
        node_type=NodeType.LLM,
        keywords=(
            "research", "find out", "look up", "investigate", "compare", "explain",
            "summarise", "summarize", "summary", "explore", "learn about",
        ),
        rationale="The goal asks for information to be gathered and condensed.",
        default_params={
            "system": "You are a meticulous research assistant. Cite what you rely on.",
        },
        instruction="Research the following and summarise the key findings.",
    ),
    Skill(
        key="plan",
        label="Plan & sequence",
        node_type=NodeType.LLM,
        keywords=(
            "plan", "itinerary", "schedule", "organise", "organize", "arrange",
            "trip", "roadmap", "agenda", "steps",
        ),
        rationale="The goal asks for an ordered plan rather than a single answer.",
        default_params={
            "system": "You are a planner. Produce concrete, ordered, actionable steps.",
        },
        instruction="Produce a concrete, ordered, step-by-step plan.",
    ),
    Skill(
        key="ground_in_documents",
        label="Ground answers in my documents",
        node_type=NodeType.RAG_RETRIEVE,
        keywords=(
            "my documents", "our docs", "knowledge base", "internal", "company",
            "policy", "handbook", "corpus", "uploaded", "private data", "our data",
        ),
        rationale="The goal refers to private material the model has not been trained on.",
        default_params={"top_k": 4, "query_template": "{{query}}"},
        requires="a Knowledge Context — upload documents before running this step",
    ),
    Skill(
        key="draft",
        label="Draft written output",
        node_type=NodeType.LLM,
        keywords=(
            "write", "draft", "compose", "email", "post", "article", "report",
            "letter", "copy", "blog",
        ),
        rationale="The goal asks for a written artefact to be produced.",
        default_params={
            "system": "You are an editor. Write clearly and concisely; no filler.",
        },
        instruction="Write the requested piece.",
    ),
    Skill(
        key="classify",
        label="Classify or decide",
        node_type=NodeType.BRANCH,
        keywords=(
            "if ", "when ", "only if", "otherwise", "depending on", "decide",
            "route", "triage", "classify", "either",
        ),
        rationale="The goal describes different outcomes depending on a condition.",
        default_params={"left": "$output.text", "operator": "contains", "right": ""},
    ),
    Skill(
        key="call_external_api",
        label="Call an external service",
        node_type=NodeType.HTTP_REQUEST,
        keywords=(
            "api", "webhook", "http", "endpoint", "integrate", "crm", "salesforce",
            "send to", "post to", "sync with",
        ),
        rationale="The goal names an external system that must be called directly.",
        default_params={"method": "GET", "url": "", "headers": {}},
        requires=(
            "an API credential and an allow-listed host — this step also needs a write "
            "scope on the run, so it is left unconnected until you confirm it"
        ),
    ),
    Skill(
        key="format",
        label="Format the result",
        node_type=NodeType.TRANSFORM,
        keywords=("format", "table", "json", "csv", "bullet", "template", "structure"),
        rationale="The goal specifies a shape for the final output.",
        default_params={"template": "{{previous}}"},
    ),
)


def match(goal: str) -> list[tuple[Skill, int]]:
    """Score every catalogue entry against the goal text.

    Returns `(skill, hits)` for skills with at least one keyword hit, strongest
    first, with catalogue order as a deterministic tie-break.
    """
    haystack = f" {goal.lower()} "
    scored: list[tuple[Skill, int]] = []
    for skill in CATALOGUE:
        hits = sum(1 for keyword in skill.keywords if keyword in haystack)
        if hits:
            scored.append((skill, hits))
    scored.sort(key=lambda pair: (-pair[1], CATALOGUE.index(pair[0])))
    return scored


def by_key(key: str) -> Skill:
    for skill in CATALOGUE:
        if skill.key == key:
            return skill
    raise KeyError(f"no skill '{key}' in the catalogue")
