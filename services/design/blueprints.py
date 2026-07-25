"""Blueprints — the shapes a workflow takes, as data.

Decomposing a goal into roles is done from these templates rather than by asking
a model to invent a structure. Two reasons:

* **Reviewable.** The same goal produces the same design, so a user who changes
  one answer sees exactly what that answer changed.
* **Portable.** It must work on a small local model or the offline stub. A design
  step that only works on a frontier model would make the platform's central
  promise conditional on which model you happened to configure.

The model still improves the design — it rewords roles and objectives against the
user's actual goal in `service._refine_with_model`. It just does not get to
decide how many agents there are or what they hand to each other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentTemplate:
    name: str
    role: str
    #: `{goal}`, `{audience}`, `{deliverable}`, `{constraints}` are substituted.
    objective: str
    #: Skill names. Missing ones are created by the skill registry.
    skills: tuple[str, ...]
    rationale: str
    #: Whether to hand this agent a document-search capability when the tenant
    #: has corpora. A drafting agent rarely needs one; a researcher always does.
    wants_documents: bool = False


@dataclass(frozen=True)
class Intent:
    key: str
    label: str
    keywords: tuple[str, ...]
    agents: tuple[AgentTemplate, ...]
    default_deliverable: str
    deliverable_options: list[str] = field(default_factory=list)
    #: Goals of this shape often touch another system.
    may_need_external: bool = False


RESEARCHER = AgentTemplate(
    name="Researcher",
    role=(
        "a careful researcher who gathers what is actually known, separates fact "
        "from assumption, and says plainly when something cannot be established"
    ),
    objective=(
        "Gather everything needed to address this goal. Note what you could not "
        "establish rather than filling the gap. Constraints: {constraints}"
    ),
    skills=("summarise",),
    rationale="The goal needs information gathered before anything can be decided.",
    wants_documents=True,
)

PLANNER = AgentTemplate(
    name="Planner",
    role=(
        "a planner who turns findings into an ordered sequence of concrete, "
        "actionable steps"
    ),
    objective=(
        "Turn what has been gathered into an ordered plan for {audience}. Every step "
        "must be something they could actually do. Constraints: {constraints}"
    ),
    skills=("plan_steps",),
    rationale="The goal asks for a sequence, not a single answer.",
)

WRITER = AgentTemplate(
    name="Writer",
    role="an editor who writes clearly and concisely, and never invents detail",
    objective=(
        "Produce {deliverable} for {audience}, using only what the previous agents "
        "established. Constraints: {constraints}"
    ),
    skills=("draft_text", "format_output"),
    rationale="The deliverable is written output.",
)

REVIEWER = AgentTemplate(
    name="Reviewer",
    role=(
        "a demanding reviewer who checks work against its objective and reports "
        "what is actually wrong with it"
    ),
    objective=(
        "Check the work against the goal, fix what is wrong, and return the corrected "
        "version. If it is already sound, return it unchanged and say so."
    ),
    skills=("critique",),
    rationale="A checking pass catches the mistakes the producing agent cannot see.",
)

ANALYST = AgentTemplate(
    name="Analyst",
    role=(
        "an analyst who compares options against stated criteria and states a "
        "recommendation with its reasoning"
    ),
    objective=(
        "Compare the options against what matters here and recommend one, with the "
        "reasoning visible. Constraints: {constraints}"
    ),
    skills=("summarise", "critique"),
    rationale="The goal is a decision between alternatives.",
    wants_documents=True,
)


INTENTS: tuple[Intent, ...] = (
    Intent(
        key="research_and_write",
        label="Research, then write it up",
        keywords=(
            # No bare "post": it is genuinely ambiguous between publishing and
            # sending, and "blog post" is already covered by "blog". The
            # integrate blueprint owns the verb.
            "research", "write", "report", "article", "summarise", "summarize",
            "brief", "blog", "documentation", "explain",
        ),
        agents=(RESEARCHER, WRITER, REVIEWER),
        default_deliverable="a written summary",
        deliverable_options=["A written summary", "A short report", "An email", "A list of points"],
    ),
    Intent(
        key="plan",
        label="Plan something",
        keywords=(
            "plan", "itinerary", "trip", "schedule", "organise", "organize",
            "roadmap", "agenda", "prepare", "arrange", "weekend", "event",
        ),
        agents=(RESEARCHER, PLANNER, WRITER),
        default_deliverable="an ordered plan",
        deliverable_options=["An ordered plan", "A day-by-day itinerary", "A checklist"],
    ),
    Intent(
        key="decide",
        label="Compare and decide",
        keywords=(
            "compare", "decide", "choose", "evaluate", "assess", "versus", "vs",
            "options", "recommend", "which",
        ),
        agents=(RESEARCHER, ANALYST, WRITER),
        default_deliverable="a recommendation with reasoning",
        deliverable_options=["A recommendation", "A comparison table", "A short decision memo"],
    ),
    Intent(
        key="process",
        label="Process or transform material",
        keywords=(
            "extract", "classify", "categorise", "categorize", "clean", "convert",
            "transform", "triage", "sort", "tag", "label",
        ),
        agents=(RESEARCHER, WRITER),
        default_deliverable="the processed output",
        deliverable_options=["A structured list", "A table", "A cleaned document"],
        may_need_external=True,
    ),
    Intent(
        key="integrate",
        label="Connect systems",
        keywords=(
            "api", "webhook", "crm", "sync", "integrate", "send", "post",
            "notify", "slack", "salesforce", "endpoint", "upload",
        ),
        agents=(RESEARCHER, WRITER),
        default_deliverable="a message ready to send",
        deliverable_options=["A message to send", "A structured payload"],
        may_need_external=True,
    ),
)

#: Used when nothing matches. One capable generalist beats a guessed pipeline.
GENERALIST = Intent(
    key="general",
    label="General assistance",
    keywords=(),
    agents=(
        AgentTemplate(
            name="Assistant",
            role=(
                "a capable generalist who works out what is actually being asked and "
                "does it, saying clearly when something is out of reach"
            ),
            objective=(
                "Achieve this goal for {audience}, producing {deliverable}. "
                "Constraints: {constraints}"
            ),
            skills=("summarise", "draft_text"),
            rationale=(
                "The goal did not match a known shape, so it starts as one capable "
                "agent you can split up once you see how it behaves."
            ),
            wants_documents=True,
        ),
    ),
    default_deliverable="a direct answer",
    deliverable_options=["A direct answer", "A short write-up", "A list"],
)


def classify(goal: str) -> Intent:
    """Pick the closest blueprint. Ties break on declaration order, so the same
    goal always classifies the same way."""
    best, best_score = GENERALIST, 0

    for intent in INTENTS:
        score = sum(1 for keyword in intent.keywords if _mentions(goal, keyword))
        if score > best_score:
            best, best_score = intent, score

    return best


def _mentions(goal: str, keyword: str) -> bool:
    """Whole-word match.

    Substring matching reads "Compare **Post**gres and MySQL" as a request to
    write a blog post, and "**Post** new leads to our CRM" the same way — both
    then lose the tie-break to whichever intent is declared first. Word
    boundaries are the difference between classifying a goal and pattern-matching
    on its spelling.
    """
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", goal.lower()) is not None


def restate(goal: str, intent: Intent, answers: dict[str, str] | None = None) -> str:
    """Say back what we understood, so a misreading surfaces before the build."""
    answers = answers or {}
    condensed = " ".join(goal.split())
    sentence = f"You want to {condensed[0].lower() + condensed[1:] if condensed else 'do something'}"

    detail = []
    if answers.get("deliverable"):
        detail.append(f"producing {answers['deliverable']}")
    if answers.get("audience"):
        detail.append(f"for {answers['audience'].lower()}")
    if detail:
        sentence += ", " + " ".join(detail)

    return f"{sentence}. Read as: {intent.label.lower()}."
