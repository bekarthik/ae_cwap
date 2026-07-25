"""Workflow templates — a starting point that already works.

The third way into the canvas. Describing a goal is the most powerful route and a
blank canvas is the most direct, but both start from nothing, and "what can this
thing actually do?" is the first question a new user has. A template answers it
by example: here is a real workflow, these are its agents, run it and then change
what you disagree with.

Templates are **data, not saved workflows**. Each is instantiated fresh per
tenant — agents created, skills resolved or built, memory scope assigned — so
copying one gives you your own agents to edit rather than a shared object that
changes under other people. Adding a template is an entry here.

Each declares what it needs. A template whose agents want a connected system says
so, and the platform reports that as something to set up rather than producing a
workflow that fails on its third step.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TemplateAgent:
    name: str
    role: str
    objective: str
    #: Skill names. Missing ones are found or built at instantiation.
    skills: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class WorkflowTemplate:
    key: str
    title: str
    #: One line, in the user's terms, describing what running it does.
    summary: str
    #: Who this is for and when to reach for it.
    detail: str
    category: str
    agents: tuple[TemplateAgent, ...]
    #: What the workflow is given when it runs, and a worked example.
    input_label: str = "What do you want done?"
    example_input: str = ""
    #: Names of MCP servers whose tools this template's agents want. Reported as
    #: a prerequisite rather than silently omitted.
    wants_connectors: tuple[str, ...] = ()
    #: True when the template expects uploaded documents to search.
    wants_documents: bool = False
    tags: list[str] = field(default_factory=list)


RESEARCH_BRIEF = WorkflowTemplate(
    key="research_brief",
    title="Research and write a briefing",
    summary="Gathers what is known on a topic, writes it up, then checks its own work.",
    detail=(
        "Three agents in sequence: one establishes the facts and says plainly what "
        "it could not, one turns that into readable prose, and one reviews the "
        "result against the original request. The reviewing pass is what catches "
        "the mistakes the writing agent cannot see in its own output."
    ),
    category="Writing",
    input_label="What should the briefing cover?",
    example_input="The current state of solar tariffs and what changed this year",
    agents=(
        TemplateAgent(
            name="Researcher",
            role=(
                "a careful researcher who gathers what is actually known, separates "
                "fact from assumption, and says plainly when something cannot be "
                "established"
            ),
            objective=(
                "Gather everything needed to address this request. Note what you "
                "could not establish rather than filling the gap.\n\nThe request "
                "is:\n{{goal}}"
            ),
            skills=("summarise",),
            rationale="Nothing can be written until the facts are in one place.",
        ),
        TemplateAgent(
            name="Writer",
            role="an editor who writes clearly and concisely, and never invents detail",
            objective=(
                "Write the briefing using only what the researcher established.\n\n"
                "The request was:\n{{goal}}\n\nThe research:\n{{previous}}"
            ),
            skills=("draft_text", "format_output"),
            rationale="The deliverable is written output, which is its own skill.",
        ),
        TemplateAgent(
            name="Reviewer",
            role="a demanding reviewer who reports what is actually wrong with a draft",
            objective=(
                "Check the briefing against the request, fix what is wrong, and "
                "return the corrected version.\n\nThe request was:\n{{goal}}\n\n"
                "The draft:\n{{previous}}"
            ),
            skills=("critique",),
            rationale="A checking pass catches what the producing agent cannot see.",
        ),
    ),
    tags=["research", "writing"],
)

DOCUMENT_QA = WorkflowTemplate(
    key="document_qa",
    title="Answer questions from your documents",
    summary="Searches what you have uploaded and answers only from it.",
    detail=(
        "One agent with a search capability over your uploaded documents. It "
        "decides when a lookup is worth doing rather than always searching first, "
        "and it is told to say when the documents do not cover something instead "
        "of answering from general knowledge."
    ),
    category="Knowledge",
    input_label="What do you want to know?",
    example_input="What is our policy on expense claims over £500?",
    wants_documents=True,
    agents=(
        TemplateAgent(
            name="Analyst",
            role=(
                "an analyst who answers strictly from the documents provided and "
                "says clearly when they do not cover the question"
            ),
            objective=(
                "Answer this using the documents. Quote what supports your answer. "
                "If the documents do not cover it, say so rather than answering "
                "from general knowledge.\n\nThe question is:\n{{goal}}"
            ),
            skills=("search_my_documents", "summarise"),
            rationale="Grounded answers need search, and searching is a capability.",
        ),
    ),
    tags=["rag", "documents"],
)

DECISION_MEMO = WorkflowTemplate(
    key="decision_memo",
    title="Compare options and recommend one",
    summary="Lays out the alternatives against what matters, then commits to a recommendation.",
    detail=(
        "An analyst compares the options against the criteria that actually apply "
        "and makes a recommendation with its reasoning visible, then a writer turns "
        "that into something you could send to someone else."
    ),
    category="Decisions",
    input_label="What are you deciding between?",
    example_input="Postgres or MySQL for a write-heavy analytics workload",
    agents=(
        TemplateAgent(
            name="Analyst",
            role=(
                "an analyst who compares options against stated criteria and commits "
                "to a recommendation with the reasoning visible"
            ),
            objective=(
                "Compare the options against what matters here and recommend one. "
                "Make the trade-offs explicit; do not hedge.\n\nThe decision "
                "is:\n{{goal}}"
            ),
            skills=("summarise", "critique"),
            rationale="A decision needs the alternatives held against each other.",
        ),
        TemplateAgent(
            name="Writer",
            role="an editor who writes a decision memo someone else can act on",
            objective=(
                "Turn this into a short memo: the recommendation, why, and what "
                "would change it.\n\nThe decision was:\n{{goal}}\n\nThe "
                "analysis:\n{{previous}}"
            ),
            skills=("draft_text", "format_output"),
            rationale="A recommendation nobody can read is a recommendation nobody takes.",
        ),
    ),
    tags=["analysis", "decisions"],
)

CODE_REVIEW = WorkflowTemplate(
    key="code_review",
    title="Review code in a repository",
    summary="Reads the code through a connected system and reports what is actually wrong.",
    detail=(
        "Needs a connected code host — an MCP server for GitHub or similar. The "
        "agent reads the files it needs through that connection and reviews them, "
        "rather than being pasted a diff. Because the connection is a skill, the "
        "agent decides what to read; you do not have to know in advance."
    ),
    category="Engineering",
    input_label="What should be reviewed?",
    example_input="The authentication changes in src/auth/",
    wants_connectors=("a code host, connected under Connected systems",),
    agents=(
        TemplateAgent(
            name="Reviewer",
            role=(
                "a demanding code reviewer who reads the code before judging it and "
                "reports concrete problems with what to change, not style opinions"
            ),
            objective=(
                "Review this. Read whatever files you need before commenting. "
                "Report real problems with the change to make; if it is sound, say "
                "so plainly.\n\nWhat to review:\n{{goal}}"
            ),
            skills=("critique",),
            rationale=(
                "Reading the code is the review. Give this agent your code host's "
                "read tools and it will fetch what it needs."
            ),
        ),
    ),
    tags=["engineering", "mcp"],
)

TRIAGE = WorkflowTemplate(
    key="triage",
    title="Sort and summarise incoming items",
    summary="Takes a pile of text, groups it, and tells you what needs attention.",
    detail=(
        "For support tickets, feedback, survey responses — anything that arrives as "
        "many small pieces of text. One agent categorises and summarises, a second "
        "shapes the result into something scannable."
    ),
    category="Operations",
    input_label="What needs sorting?",
    example_input="Last week's support tickets, pasted below",
    agents=(
        TemplateAgent(
            name="Triager",
            role=(
                "an operations analyst who groups incoming items by what they are "
                "actually about and flags what needs a person"
            ),
            objective=(
                "Group these by theme, count each group, and flag anything that "
                "needs attention now.\n\nThe items:\n{{goal}}"
            ),
            skills=("summarise",),
            rationale="Grouping before summarising is what makes the summary useful.",
        ),
        TemplateAgent(
            name="Writer",
            role="an editor who produces a scannable summary",
            objective=(
                "Turn this into a short scannable report: the groups, the counts, "
                "and what needs attention first.\n\nThe analysis:\n{{previous}}"
            ),
            skills=("format_output", "draft_text"),
            rationale="A wall of text is not a triage report.",
        ),
    ),
    tags=["operations"],
)


TEMPLATES: tuple[WorkflowTemplate, ...] = (
    RESEARCH_BRIEF,
    DOCUMENT_QA,
    DECISION_MEMO,
    CODE_REVIEW,
    TRIAGE,
)


def all_templates() -> tuple[WorkflowTemplate, ...]:
    return TEMPLATES


def find(key: str) -> WorkflowTemplate | None:
    return next((template for template in TEMPLATES if template.key == key), None)
