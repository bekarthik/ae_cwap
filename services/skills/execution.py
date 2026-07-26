"""Running a skill.

Each skill kind maps to a primitive the platform already trusts. Nothing here
evaluates code; a skill's `definition` is data all the way down.

Every invocation consults the skill's own memory first and can contribute back to
it afterwards. That is the per-skill learning loop: a lesson about *using this
capability* — how to phrase its input, what it does badly — accrues to the skill,
so every agent that reaches for it inherits the lesson.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cwap_contracts.v4 import (
    MemoryKind,
    MemoryScope,
    MemoryWriteRequest,
    RecallRequest,
    RetrievalRequest,
    SkillDefinition,
    SkillKind,
)
from knowledge.service import KnowledgeError, retrieve
from llm_proxy.client import GenerationOptions, LLMProxyError, LLMRefusal, get_provider
from memory import service as memory_service
from orchestrator.variables import BindingError, render_template

from skills import registry

#: Prevents one skill's output from swamping the agent's context window.
MAX_SKILL_OUTPUT_CHARS = 20_000

#: How many of the skill's own lessons to inject into a prompt skill.
SKILL_MEMORY_RECALL = 3


class SkillExecutionError(RuntimeError):
    """A skill ran and failed. Reported to the agent, which may try another way."""


@dataclass
class SkillContext:
    """Everything a skill needs that is not its own definition."""

    tenant_id: str
    run_id: str = ""
    step_execution_id: str = ""
    #: Set for HTTP skills, which need the same two-phase idempotency gate as an
    #: HTTP node — an agent retrying a loop must not re-fire a side effect.
    allow_side_effects: bool = False
    emit: Any = None
    #: Memory entry ids consulted during this invocation, so the outcome can
    #: reinforce or weaken them.
    consulted: list[str] = field(default_factory=list)


@dataclass
class SkillOutcome:
    output: str
    is_error: bool = False
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def execute(
    skill: SkillDefinition, arguments: dict[str, Any], context: SkillContext
) -> SkillOutcome:
    """Run a skill and record whether it worked."""
    started = time.perf_counter()
    try:
        output = _dispatch(skill, arguments, context)
        outcome = SkillOutcome(
            output=_truncate(output),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        registry.record_invocation(skill.tenant_id, skill.id, failed=False)
        return outcome
    except (SkillExecutionError, KnowledgeError, LLMProxyError, BindingError) as exc:
        registry.record_invocation(skill.tenant_id, skill.id, failed=True)
        # A failed skill is reported *to the agent* rather than failing the run:
        # the agent can try different arguments or a different capability, which
        # is the whole point of giving it a loop.
        return SkillOutcome(
            output=f"{skill.name} failed: {exc}",
            is_error=True,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


def _dispatch(
    skill: SkillDefinition, arguments: dict[str, Any], context: SkillContext
) -> str:
    _check_required(skill, arguments)
    values = {parameter.name: arguments.get(parameter.name, "") for parameter in skill.parameters}

    if skill.kind is SkillKind.TRANSFORM:
        return render_template(skill.definition["template"], values)

    if skill.kind is SkillKind.PROMPT:
        return _run_prompt(skill, values, context)

    if skill.kind is SkillKind.RETRIEVAL:
        return _run_retrieval(skill, values, context)

    if skill.kind is SkillKind.HTTP:
        return _run_http(skill, values, context)

    if skill.kind is SkillKind.MCP:
        return _run_mcp(skill, arguments, context)

    if skill.kind is SkillKind.COMPOSITE:
        return _run_composite(skill, values, context)

    raise SkillExecutionError(f"skill kind '{skill.kind}' has no implementation")


def _check_required(skill: SkillDefinition, arguments: dict[str, Any]) -> None:
    missing = [
        parameter.name
        for parameter in skill.parameters
        if parameter.required and not str(arguments.get(parameter.name, "")).strip()
    ]
    if missing:
        raise SkillExecutionError(
            f"missing required argument(s) {missing}; this skill needs "
            f"{[p.name for p in skill.parameters if p.required]}"
        )


def _run_prompt(
    skill: SkillDefinition, values: dict[str, Any], context: SkillContext
) -> str:
    prompt = render_template(skill.definition["prompt_template"], values)
    system = skill.definition.get("system") or ""

    lessons = recall_skill_memory(skill, prompt, context)
    if lessons:
        # The skill's accumulated lessons ride in the system prompt, so they
        # shape how the capability is applied without polluting the task text.
        system = f"{system}\n\n{lessons}".strip()

    try:
        completion = get_provider().complete(
            prompt,
            system=system or None,
            options=GenerationOptions(
                max_tokens=int(skill.definition.get("max_tokens") or 0) or None,
                temperature=skill.definition.get("temperature"),
                effort=skill.definition.get("effort"),
            ),
        )
    except LLMRefusal as exc:
        raise SkillExecutionError(
            "the model declined this request"
            + (f" (category: {exc.category})" if exc.category else "")
        ) from exc

    return completion.text


def _run_retrieval(
    skill: SkillDefinition, values: dict[str, Any], context: SkillContext
) -> str:
    handle = skill.definition["knowledge_handle"]
    query = next((str(v) for v in values.values() if str(v).strip()), "")
    if not query:
        raise SkillExecutionError("retrieval needs a query")

    result = retrieve(
        RetrievalRequest(
            handle=handle,
            tenant_id=context.tenant_id,
            query=query,
            top_k=int(skill.definition.get("top_k", 4)),
        )
    )
    if not result.chunks:
        return "No relevant passages found in that corpus."
    return result.as_context()


def _run_http(
    skill: SkillDefinition, values: dict[str, Any], context: SkillContext
) -> str:
    """An outbound call, with the same guards a canvas HTTP node gets.

    Agents get no privilege a user does not: the host must be allow-listed, the
    run must carry a write scope, and the call passes the idempotency gate so a
    retried loop iteration cannot fire it twice.
    """
    if not context.allow_side_effects:
        raise SkillExecutionError(
            "this run has no write scope, so external calls are not permitted"
        )

    from cwap_common.db import unit_of_work  # noqa: PLC0415 - narrow use
    from cwap_common.idempotency import IdempotencyGate  # noqa: PLC0415
    from orchestrator.executors import (  # noqa: PLC0415
        NodeExecutionError,
        _assert_host_allowed,
    )

    url = render_template(str(skill.definition["url"]), values)
    method = str(skill.definition.get("method", "GET")).upper()
    try:
        _assert_host_allowed(url, skill.name)
    except NodeExecutionError as exc:
        # An egress denial is reported to the agent like any other skill failure,
        # not raised: the policy is not something the agent can fix, but it can
        # still finish the objective another way, and the run should not die
        # because one capability turned out to be unreachable.
        raise SkillExecutionError(str(exc)) from exc

    operation = f"skill:{skill.name}:{hash(tuple(sorted(values.items())))}"
    with unit_of_work() as session:
        gate = IdempotencyGate(session, context.run_id, context.step_execution_id, operation)
        claim = gate.pre_check()
    if not claim.should_execute:
        return str((claim.prior_response or {}).get("body", "already performed"))

    import httpx  # noqa: PLC0415

    body = skill.definition.get("body")
    if isinstance(body, str) and body:
        body = render_template(body, values)

    try:
        with httpx.Client(timeout=20) as client:
            response = client.request(
                method,
                url,
                headers=skill.definition.get("headers") or {},
                content=body if isinstance(body, str) else None,
                json=body if isinstance(body, dict) else None,
            )
        payload = {"status": response.status_code, "body": response.text[:MAX_SKILL_OUTPUT_CHARS]}
    except Exception as exc:
        with unit_of_work() as session:
            IdempotencyGate(
                session, context.run_id, context.step_execution_id, operation
            ).abandon(str(exc))
        raise SkillExecutionError(f"request to {url} failed: {exc}") from exc

    with unit_of_work() as session:
        IdempotencyGate(session, context.run_id, context.step_execution_id, operation).commit(
            payload
        )
    return f"HTTP {payload['status']}\n{payload['body']}"


def _run_mcp(
    skill: SkillDefinition, arguments: dict[str, Any], context: SkillContext
) -> str:
    """Invoke one tool on a connected MCP server.

    The arguments go through **unflattened**: the model was given the server's
    own schema, so what it produced is what the server expects. Coercing them
    through the platform's parameter list would break every tool that takes a
    structured argument.

    A write-capable tool needs the same write scope an HTTP node does — an agent
    gets no privilege the user running it has. Tools the server itself marks
    read-only are exempt, because reading a repository is not a side effect.
    """
    from mcp_connect.client import MCPError, get_client  # noqa: PLC0415
    from mcp_connect.policy import MCPPolicyError  # noqa: PLC0415
    from mcp_connect.registry import (  # noqa: PLC0415
        MCPServerNotFound,  # noqa: PLC0415
        credentials_for,
    )
    from mcp_connect.registry import get as get_server

    if skill.definition.get("unavailable"):
        raise SkillExecutionError(
            f"'{skill.name}' is no longer offered by its server. Re-sync the "
            "connection, or use a different capability."
        )

    read_only = bool(skill.definition.get("read_only"))
    if not read_only and not context.allow_side_effects:
        raise SkillExecutionError(
            f"'{skill.name}' can change things outside the platform, and this run "
            "has no write scope"
        )

    server_id = str(skill.definition["server_id"])
    tool_name = str(skill.definition["tool_name"])

    try:
        server = get_server(context.tenant_id, server_id)
    except MCPServerNotFound as exc:
        raise SkillExecutionError(
            f"the server behind '{skill.name}' is no longer connected"
        ) from exc

    if not server.enabled:
        raise SkillExecutionError(f"the '{server.name}' connection is switched off")

    # A write goes through the same two-phase gate an HTTP node does, so a
    # redelivered job cannot open the same pull request twice.
    gate_operation = f"mcp:{server_id}:{tool_name}:{_argument_digest(arguments)}"
    claim = None
    if not read_only and context.run_id:
        from cwap_common.db import unit_of_work  # noqa: PLC0415
        from cwap_common.idempotency import IdempotencyGate  # noqa: PLC0415

        with unit_of_work() as session:
            claim = IdempotencyGate(
                session, context.run_id, context.step_execution_id, gate_operation
            ).pre_check()
        if not claim.should_execute:
            return str(
                (claim.prior_response or {}).get("output", "already performed")
            )

    try:
        output = get_client().call_tool(
            server_id,
            server.config,
            credentials_for(context.tenant_id, server_id),
            tool_name,
            arguments,
        )
    except (MCPError, MCPPolicyError) as exc:
        if claim is not None:
            from cwap_common.db import unit_of_work  # noqa: PLC0415
            from cwap_common.idempotency import IdempotencyGate  # noqa: PLC0415

            with unit_of_work() as session:
                IdempotencyGate(
                    session, context.run_id, context.step_execution_id, gate_operation
                ).abandon(str(exc))
        raise SkillExecutionError(str(exc)) from exc

    if claim is not None:
        from cwap_common.db import unit_of_work  # noqa: PLC0415
        from cwap_common.idempotency import IdempotencyGate  # noqa: PLC0415

        with unit_of_work() as session:
            IdempotencyGate(
                session, context.run_id, context.step_execution_id, gate_operation
            ).commit({"output": output[:MAX_SKILL_OUTPUT_CHARS]})

    return output or "(the tool returned nothing)"


def _argument_digest(arguments: dict[str, Any]) -> str:
    """A stable fingerprint of one call's arguments.

    Part of the idempotency key so that calling the same tool twice with
    *different* arguments in one step is two operations, while a redelivery of
    the same call is one.
    """
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    encoded = json.dumps(arguments or {}, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _run_composite(
    skill: SkillDefinition, values: dict[str, Any], context: SkillContext
) -> str:
    """Run named skills in order, threading each output into the next."""
    steps = skill.definition.get("steps") or []
    if not steps:
        raise SkillExecutionError(f"composite skill '{skill.name}' has no steps")

    carried = dict(values)
    output = ""
    for index, step in enumerate(steps, start=1):
        step_name = step.get("skill") if isinstance(step, dict) else str(step)
        child = registry.find_by_name(skill.tenant_id, str(step_name))
        if child is None:
            raise SkillExecutionError(
                f"composite '{skill.name}' step {index} names unknown skill '{step_name}'"
            )
        if child.id == skill.id:
            raise SkillExecutionError(f"composite '{skill.name}' cannot invoke itself")

        arguments = {
            parameter.name: carried.get(parameter.name, output if output else "")
            for parameter in child.parameters
        }
        result = execute(child, arguments, context)
        if result.is_error:
            raise SkillExecutionError(result.output)
        output = result.output
        carried["previous"] = output

    return output


# ---------------------------------------------------------------------------
# per-skill memory
# ---------------------------------------------------------------------------


def recall_skill_memory(
    skill: SkillDefinition, query: str, context: SkillContext | None = None
) -> str:
    """What this skill has learned about being used well."""
    # Through the shared helper rather than a local slice: the local slice here
    # was correct and the one missing from the agent path was not, and nothing
    # tied them together.
    trimmed = memory_service.query_text(query)
    if not trimmed:
        return ""

    result = memory_service.recall(
        RecallRequest(
            tenant_id=skill.tenant_id,
            scope=MemoryScope.SKILL,
            scope_id=skill.id,
            query=trimmed,
            limit=SKILL_MEMORY_RECALL,
        )
    )
    if context is not None:
        context.consulted.extend(entry.id for entry in result.entries)
    return result.as_prompt_block(f"Lessons from previous uses of {skill.name}")


def teach_skill(
    skill: SkillDefinition,
    lesson: str,
    *,
    kind: MemoryKind = MemoryKind.LEARNING,
    run_id: str | None = None,
) -> None:
    """Record something learned about using this capability."""
    memory_service.remember(
        MemoryWriteRequest(
            tenant_id=skill.tenant_id,
            scope=MemoryScope.SKILL,
            scope_id=skill.id,
            kind=kind,
            text=lesson,
            source_run_id=run_id,
        )
    )


def _truncate(text: str) -> str:
    if len(text) <= MAX_SKILL_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_SKILL_OUTPUT_CHARS]
        + f"\n\n[truncated at {MAX_SKILL_OUTPUT_CHARS} characters]"
    )
