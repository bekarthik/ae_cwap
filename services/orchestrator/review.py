"""Quality loops: a second agent has to be satisfied before an answer is used.

An agent checking its own work is a weak check — it has already decided the
answer is good, which is why it stopped. A *different* agent, with its own role,
its own skills and its own memory of reviewing, is a real one.

The loop is bounded and lives **inside** one step:

    draft ──► reviewer ──► approved?  ──yes──► done
                 │            │
                 │            no
                 ▼            ▼
            feedback ──► author revises ──► (up to `max_rounds`)

Two decisions worth stating.

**Repetition is inside a step, not an edge on the canvas.** An edge back to an
earlier node would make the graph cyclic, and then whether a run terminates would
depend on a model's judgement rather than on the structure. Bounding the rounds
here keeps the graph acyclic, keeps every run finite, and keeps the cost of a
workflow knowable before it starts — at most `max_rounds` extra pairs of calls.

**The round limit is not a failure.** A reviewer that still has notes when the
budget runs out has improved the draft several times over; discarding that would
be worse than shipping it. The draft stands, and the run report says the loop
ended unapproved rather than pretending it passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cwap_contracts.v4 import LogLevel, ReviewConfig

#: What the reviewer is told to do. Deliberately asks for the approval word on a
#: line of its own: a reviewer that buries "approved" in a paragraph of caveats
#: has not approved anything, and the prefix match is what keeps that honest.
REVIEWER_BRIEF = """\
You are reviewing another agent's work before it is used.

The objective it was given:
{objective}

What it produced:
{draft}

Judge it against that objective and nothing else. If it genuinely meets the
objective, reply with exactly {approval} on the first line and stop. Otherwise
list what is wrong and what specifically to change — be concrete, quote the part
you mean, and do not rewrite it yourself.
"""

#: What the author is told when the reviewer asked for changes.
REVISION_BRIEF = """\
{objective}

You produced this draft:
{draft}

A reviewer asked for these changes:
{feedback}

Produce the corrected version in full. Address every point, or say plainly why a
point should not be applied.
"""


@dataclass
class ReviewOutcome:
    """What the loop did, for the run report."""

    text: str
    rounds: int = 0
    approved: bool = False
    #: One entry per round: the reviewer's verdict, so a reader can see what
    #: changed and why, not merely that something did.
    notes: list[dict[str, Any]] = field(default_factory=list)


def run_loop(
    *,
    config: ReviewConfig,
    objective: str,
    draft: str,
    author,
    reviewer,
    run_agent,
    context,
    emit,
) -> ReviewOutcome:
    """Critique and revise until the reviewer approves or the rounds run out.

    Takes its collaborators as arguments rather than importing them: the agent
    runtime already imports the orchestrator's executors, and reaching back the
    other way at module scope would close that circle.
    """
    outcome = ReviewOutcome(text=draft)

    for round_number in range(1, config.max_rounds + 1):
        emit(
            "review.started",
            message=f"'{reviewer.name}' is reviewing round {round_number}/{config.max_rounds}",
            data={"reviewer": reviewer.name, "round": round_number},
        )

        verdict = run_agent(
            reviewer,
            REVIEWER_BRIEF.format(
                objective=objective,
                draft=outcome.text,
                approval=config.approval_phrase,
            ),
            context,
        ).text

        outcome.rounds = round_number
        if approves(verdict, config.approval_phrase):
            outcome.approved = True
            outcome.notes.append({"round": round_number, "verdict": "approved"})
            emit(
                "review.approved",
                message=f"'{reviewer.name}' approved it after {round_number} round(s)",
                data={"reviewer": reviewer.name, "round": round_number},
            )
            return outcome

        outcome.notes.append(
            {"round": round_number, "verdict": "changes", "feedback": verdict}
        )
        emit(
            "review.changes",
            message=f"'{reviewer.name}' asked for changes: {_excerpt(verdict)}",
            data={"reviewer": reviewer.name, "round": round_number},
        )

        outcome.text = run_agent(
            author,
            REVISION_BRIEF.format(objective=objective, draft=outcome.text, feedback=verdict),
            context,
        ).text

    emit(
        "review.unapproved",
        level=LogLevel.WARN,
        message=(
            f"'{reviewer.name}' still had notes after {config.max_rounds} round(s); "
            "the last revision stands"
        ),
        data={"reviewer": reviewer.name, "rounds": config.max_rounds},
    )
    return outcome


def approves(verdict: str, phrase: str) -> bool:
    """Whether this verdict ends the loop.

    A prefix match on the first line, case-insensitively. Matching anywhere in
    the text would let "this is not approved" and "approved, but…" both pass,
    which is the opposite of what a reviewer meant in each case.
    """
    first = (verdict or "").strip().splitlines()[0] if (verdict or "").strip() else ""
    return first.strip().lower().startswith(phrase.strip().lower())


def _excerpt(text: str, limit: int = 160) -> str:
    condensed = " ".join((text or "").split())
    return condensed[:limit] + ("…" if len(condensed) > limit else "")
