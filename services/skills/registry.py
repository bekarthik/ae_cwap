"""Skill storage, built-ins, and synthesis of missing capabilities.

Synthesis is the part that needs care. When an agent needs a capability the
tenant does not have, the platform creates it — but a *declarative* skill, never
code. A synthesised skill is a prompt template, a retrieval over a corpus the
tenant already owns, a text transform, an allow-listed HTTP call, or an ordered
composition of those. Everything it can do, a user could already have built by
hand on the canvas.

That boundary is deliberate. If synthesis emitted Python, a model that had read a
hostile document could write arbitrary code into a privileged worker, and the
egress allow-list and scope checks would all be bypassable.
"""

from __future__ import annotations

import json
import uuid

from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import Skill as SkillRow
from cwap_contracts.v2 import (
    SkillDefinition,
    SkillKind,
    SkillOrigin,
    SkillParameter,
    SkillProposal,
)
from llm_proxy.client import GenerationOptions, LLMProxyError, get_provider


class SkillError(RuntimeError):
    """A skill could not be created, found or validated."""


#: Capabilities every tenant gets. Deliberately generic — they are the verbs
#: that show up in almost every workflow, and a specific one is better
#: synthesised against the actual task than guessed at here.
BUILTIN_SKILLS: tuple[SkillProposal, ...] = (
    SkillProposal(
        name="summarise",
        description=(
            "Condense text into its key points. Use when you have long source "
            "material and need the substance of it."
        ),
        kind=SkillKind.PROMPT,
        parameters=[
            SkillParameter(name="text", description="The text to condense."),
            SkillParameter(
                name="focus",
                description="What the summary should emphasise.",
                required=False,
            ),
        ],
        definition={
            "prompt_template": (
                "Summarise the following, keeping every fact that changes a decision "
                "and dropping everything that does not.\n\n"
                "Focus on: {{focus}}\n\nText:\n{{text}}"
            ),
            "system": "You summarise precisely. You never invent detail that is not present.",
        },
    ),
    SkillProposal(
        name="plan_steps",
        description=(
            "Break an objective into an ordered list of concrete steps. Use when "
            "the task needs sequencing before it can be executed."
        ),
        kind=SkillKind.PROMPT,
        parameters=[
            SkillParameter(name="objective", description="What needs to be achieved."),
            SkillParameter(
                name="constraints", description="Limits to respect.", required=False
            ),
        ],
        definition={
            "prompt_template": (
                "Produce an ordered, concrete plan for this objective. Each step must be "
                "something a person could actually do.\n\n"
                "Objective:\n{{objective}}\n\nConstraints:\n{{constraints}}"
            ),
            "system": "You are a planner. Concrete and ordered; no filler.",
        },
    ),
    SkillProposal(
        name="draft_text",
        description=(
            "Write a piece of prose to a brief — an email, a section, a report. "
            "Use when the deliverable is written output."
        ),
        kind=SkillKind.PROMPT,
        parameters=[
            SkillParameter(name="brief", description="What to write and for whom."),
            SkillParameter(name="material", description="Source material.", required=False),
        ],
        definition={
            "prompt_template": (
                "Write the piece described in the brief. Use only the material provided; "
                "if something is missing, say so rather than inventing it.\n\n"
                "Brief:\n{{brief}}\n\nMaterial:\n{{material}}"
            ),
            "system": "You are an editor. Clear and concise; no filler.",
        },
    ),
    SkillProposal(
        name="critique",
        description=(
            "Review a draft against its objective and list what is wrong with it. "
            "Use before returning work, to catch your own mistakes."
        ),
        kind=SkillKind.PROMPT,
        parameters=[
            SkillParameter(name="work", description="The draft to review."),
            SkillParameter(name="objective", description="What it was meant to achieve."),
        ],
        definition={
            "prompt_template": (
                "Review this work against its objective. List only real problems, each with "
                "what to change. If it is sound, say so plainly.\n\n"
                "Objective:\n{{objective}}\n\nWork:\n{{work}}"
            ),
            "system": "You are a demanding reviewer. Specific, not generic.",
        },
    ),
    SkillProposal(
        name="format_output",
        description=(
            "Reshape text into a required layout without a model call. Use for the "
            "final shaping of a deliverable."
        ),
        kind=SkillKind.TRANSFORM,
        parameters=[SkillParameter(name="content", description="The content to place.")],
        definition={"template": "{{content}}"},
    ),
)


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------


def ensure_builtins(tenant_id: str) -> list[SkillDefinition]:
    """Seed the built-in skills for a tenant, idempotently."""
    created: list[SkillDefinition] = []
    for proposal in BUILTIN_SKILLS:
        if find_by_name(tenant_id, proposal.name) is None:
            created.append(
                create(tenant_id, proposal, origin=SkillOrigin.BUILTIN)
            )
    return created


def create(
    tenant_id: str, proposal: SkillProposal, *, origin: SkillOrigin = SkillOrigin.USER
) -> SkillDefinition:
    """Store a proposal as a skill.

    The proposal type cannot carry an id, tenant or origin, so a synthesised
    skill can never claim to be a builtin or overwrite another skill's history.
    """
    existing = find_by_name(tenant_id, proposal.name)
    if existing is not None:
        raise SkillError(f"tenant already has a skill named '{proposal.name}'")

    skill = SkillDefinition(
        id=f"skl_{uuid.uuid4().hex[:20]}",
        tenant_id=tenant_id,
        name=proposal.name,
        description=proposal.description,
        kind=proposal.kind,
        parameters=proposal.parameters,
        definition=proposal.definition,
        origin=origin,
    )

    with unit_of_work() as session:
        session.add(
            SkillRow(
                id=skill.id,
                tenant_id=tenant_id,
                name=skill.name,
                description=skill.description,
                kind=skill.kind.value,
                parameters=[p.model_dump(mode="json") for p in skill.parameters],
                definition=skill.definition,
                origin=skill.origin.value,
            )
        )
    return skill


def find_by_name(tenant_id: str, name: str) -> SkillDefinition | None:
    with read_only_session() as session:
        row = session.query(SkillRow).filter_by(tenant_id=tenant_id, name=name).one_or_none()
        return _to_definition(row) if row else None


def get(tenant_id: str, skill_id: str) -> SkillDefinition | None:
    with read_only_session() as session:
        row = session.query(SkillRow).filter_by(tenant_id=tenant_id, id=skill_id).one_or_none()
        return _to_definition(row) if row else None


def get_many(tenant_id: str, skill_ids: list[str]) -> list[SkillDefinition]:
    if not skill_ids:
        return []
    with read_only_session() as session:
        rows = (
            session.query(SkillRow)
            .filter(SkillRow.tenant_id == tenant_id, SkillRow.id.in_(skill_ids))
            .all()
        )
        by_id = {row.id: _to_definition(row) for row in rows}
    # Preserve the agent's declared order: it is the order the model sees them.
    return [by_id[skill_id] for skill_id in skill_ids if skill_id in by_id]


def list_all(tenant_id: str) -> list[SkillDefinition]:
    with read_only_session() as session:
        rows = (
            session.query(SkillRow)
            .filter_by(tenant_id=tenant_id)
            .order_by(SkillRow.origin, SkillRow.name)
            .all()
        )
        return [_to_definition(row) for row in rows]


def record_invocation(tenant_id: str, skill_id: str, *, failed: bool) -> None:
    """Track reliability, so a synthesised skill that never works is visible."""
    with unit_of_work() as session:
        row = session.query(SkillRow).filter_by(tenant_id=tenant_id, id=skill_id).one_or_none()
        if row is None:
            return
        row.invocations += 1
        if failed:
            row.failures += 1


def delete(tenant_id: str, skill_id: str) -> bool:
    with unit_of_work() as session:
        return bool(
            session.query(SkillRow).filter_by(tenant_id=tenant_id, id=skill_id).delete()
        )


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """\
You design capabilities for an agent platform. A capability is DECLARATIVE — you \
never write code. You return exactly one JSON object and nothing else.

Allowed kinds:
- "prompt":    a parameterised model call. definition: {"prompt_template": "...", "system": "..."}
- "transform": pure text shaping, no model. definition: {"template": "..."}
- "retrieval": search a document corpus. definition: {"knowledge_handle": "<handle>", "top_k": 4}

Rules:
- name: lower_snake_case, a verb phrase, <= 64 chars.
- description: when an agent should reach for this, in one or two sentences.
- parameters: at most 4. Each {"name", "type", "description", "required"}.
- Every {{placeholder}} in a template MUST match a parameter name exactly.
- Prefer "prompt" unless the task is pure text shaping or a document lookup.
"""


def synthesize(
    tenant_id: str,
    capability: str,
    *,
    context: str = "",
    knowledge_handles: list[str] | None = None,
) -> SkillProposal:
    """Design a skill for a capability the tenant does not have.

    Falls back to a deterministic prompt skill when no model is configured or the
    model returns something unusable — an agent that needs a capability is better
    served by a plain templated one than by a failure.
    """
    handles = knowledge_handles or []
    instruction = (
        f"Design one capability for this need:\n{capability}\n\n"
        f"Where it will be used:\n{context or 'general workflow automation'}\n\n"
        + (
            f"Corpora available for retrieval kinds: {handles}\n"
            if handles
            else "No document corpora are available, so do not use the retrieval kind.\n"
        )
        + "Return only the JSON object."
    )

    try:
        completion = get_provider().complete(
            instruction,
            system=SYNTHESIS_SYSTEM,
            options=GenerationOptions(max_tokens=1200, temperature=0.2, effort="medium"),
        )
        proposal = _parse_proposal(completion.text)
        if proposal is not None:
            _validate_proposal(proposal, handles)
            return proposal
    except (LLMProxyError, SkillError):
        pass

    return _fallback_proposal(capability, context)


def _parse_proposal(text: str) -> SkillProposal | None:
    """Extract the JSON object, tolerating fenced or prefixed output.

    Small open models routinely wrap JSON in prose or a code fence. Refusing
    those responses would make synthesis work only on the largest models.
    """
    candidate = text.strip()
    if "```" in candidate:
        blocks = [part for part in candidate.split("```") if "{" in part]
        if blocks:
            candidate = blocks[0]
            if candidate.lstrip().startswith("json"):
                candidate = candidate.lstrip()[4:]

    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        return SkillProposal.model_validate(json.loads(candidate[start : end + 1]))
    except Exception:
        return None


def _validate_proposal(proposal: SkillProposal, handles: list[str]) -> None:
    """Reject a design that would fail the first time it ran."""
    parameter_names = {parameter.name for parameter in proposal.parameters}

    template = proposal.definition.get("prompt_template") or proposal.definition.get("template")
    if proposal.kind in (SkillKind.PROMPT, SkillKind.TRANSFORM):
        if not template:
            raise SkillError(f"'{proposal.name}' has no template")
        import re  # noqa: PLC0415 - only needed here

        referenced = set(re.findall(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}", template))
        missing = referenced - parameter_names
        if missing:
            raise SkillError(
                f"'{proposal.name}' references {sorted(missing)} but declares "
                f"{sorted(parameter_names)}"
            )

    if proposal.kind is SkillKind.RETRIEVAL:
        handle = proposal.definition.get("knowledge_handle")
        if handle not in handles:
            raise SkillError(
                f"'{proposal.name}' wants corpus '{handle}', which this tenant does not have"
            )

    if proposal.kind is SkillKind.HTTP:
        # Synthesis may not invent network calls. A skill that reaches outside
        # the platform is a decision for a human, with a credential and an
        # allow-list entry behind it.
        raise SkillError("synthesised skills may not make HTTP calls")


def _fallback_proposal(capability: str, context: str) -> SkillProposal:
    """A plain templated prompt skill. Always valid, never clever."""
    name = _slug(capability)
    return SkillProposal(
        name=name,
        description=f"{capability.strip().rstrip('.')}. Generated as a plain prompt capability.",
        kind=SkillKind.PROMPT,
        parameters=[
            SkillParameter(name="input", description="What to work on."),
            SkillParameter(name="context", description="Supporting detail.", required=False),
        ],
        definition={
            "prompt_template": (
                f"{capability.strip()}\n\nInput:\n{{{{input}}}}\n\nContext:\n{{{{context}}}}"
            ),
            "system": "You perform the requested task precisely and say when you cannot.",
        },
        rationale=(
            "Generated without a model, or after the model returned an unusable design. "
            f"Intended use: {context or 'general'}."
        ),
    )


def _slug(text: str, limit: int = 48) -> str:
    words = [word for word in "".join(c if c.isalnum() else " " for c in text.lower()).split()][:5]
    slug = "_".join(words)[:limit].strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"skill_{slug}" if slug else "generated_skill"
    return slug


def ensure_capability(
    tenant_id: str,
    capability: str,
    *,
    context: str = "",
    knowledge_handles: list[str] | None = None,
) -> tuple[SkillDefinition, bool]:
    """Find a skill for a capability, creating one if none exists.

    Returns `(skill, created)`. This is what item 4 asks for: an agent declares
    what it needs, and the platform either finds it or builds it.
    """
    wanted = _slug(capability)
    existing = find_by_name(tenant_id, wanted)
    if existing is not None:
        return existing, False

    proposal = synthesize(
        tenant_id, capability, context=context, knowledge_handles=knowledge_handles
    )
    already = find_by_name(tenant_id, proposal.name)
    if already is not None:
        return already, False

    return create(tenant_id, proposal, origin=SkillOrigin.SYNTHESIZED), True


def _to_definition(row: SkillRow) -> SkillDefinition:
    return SkillDefinition(
        id=row.id,
        tenant_id=row.tenant_id,
        name=row.name,
        description=row.description,
        kind=SkillKind(row.kind),
        parameters=[SkillParameter.model_validate(p) for p in (row.parameters or [])],
        definition=row.definition or {},
        origin=SkillOrigin(row.origin),
        version=row.version,
        invocations=row.invocations,
        failures=row.failures,
    )
