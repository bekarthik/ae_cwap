"""Node executors — the actual work behind each block on the canvas.

One executor per `NodeType`, registered in `EXECUTORS`. Each is a pure function
of (node params, resolved inputs, run context) except `http_request`, which is
the only executor that causes an external side effect and therefore the only one
that runs the two-phase idempotency gate from mandate §3.A.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from cwap_common.db import unit_of_work
from cwap_common.idempotency import IdempotencyGate
from cwap_common.settings import get_settings
from cwap_contracts.v3 import LogLevel, NodeType, RetrievalRequest
from knowledge.service import KnowledgeError, retrieve
from llm_proxy.client import (
    GenerationOptions,
    LLMProxyError,
    LLMRefusal,
    get_provider,
)

from orchestrator.variables import BindingError, RunContext, render_template, resolve


class NodeExecutionError(RuntimeError):
    """A node failed in a way that should fail the run with a clear message."""


@dataclass
class ExecutionRequest:
    """Everything an executor is allowed to see."""

    node: Any  # WorkflowNode; typed loosely to avoid a circular import
    inputs: dict[str, Any]
    context: RunContext
    run_id: str
    step_execution_id: str
    tenant_id: str
    #: emit(event, message=..., level=..., data=...) -> streams a LogEvent
    emit: Callable[..., None]
    #: Memory shared by every reasoning node in this workflow.
    workflow_memory_scope: str = ""
    #: Whether this run carries a write scope. Gates skills that reach outward.
    allow_side_effects: bool = False


@dataclass
class ExecutionOutcome:
    output: dict[str, Any] = field(default_factory=dict)
    derived_context: dict[str, Any] = field(default_factory=dict)
    #: Set only by branch-style nodes; selects which outgoing edge is taken.
    branch: bool | None = None


class Executor(Protocol):
    def __call__(self, request: ExecutionRequest) -> ExecutionOutcome: ...


# ---------------------------------------------------------------------------
# input / output
# ---------------------------------------------------------------------------


def execute_input(request: ExecutionRequest) -> ExecutionOutcome:
    """Seed the run with its declared inputs.

    Declared defaults fill gaps but never override what the user actually
    supplied at run time.
    """
    params = request.node.params or {}
    defaults = dict(params.get("defaults") or {})
    resolved = {**defaults, **request.context.inputs}

    required = params.get("fields") or []
    missing = [name for name in required if not resolved.get(name)]
    if missing:
        raise NodeExecutionError(
            f"input node '{request.node.id}' is missing required field(s): {missing}"
        )

    return ExecutionOutcome(output=resolved, derived_context={"field_count": len(resolved)})


def execute_output(request: ExecutionRequest) -> ExecutionOutcome:
    """Produce the run's final result payload."""
    template = (request.node.params or {}).get("result_template")
    if template:
        result = render_template(template, request.inputs)
    else:
        result = request.inputs.get("previous") or request.inputs
    return ExecutionOutcome(output={"result": result})


# ---------------------------------------------------------------------------
# llm
# ---------------------------------------------------------------------------


def execute_llm(request: ExecutionRequest) -> ExecutionOutcome:
    params = request.node.params or {}
    template = params.get("prompt_template")

    if template:
        prompt = render_template(template, request.inputs)
    else:
        # A node the user dropped without configuring still needs to do
        # something sensible rather than fail.
        prompt = str(request.inputs.get("goal") or request.inputs.get("previous") or "").strip()
    if not prompt:
        raise NodeExecutionError(
            f"llm node '{request.node.id}' produced an empty prompt; set a prompt_template "
            "or connect an input that supplies 'goal'"
        )

    provider = get_provider()
    options = _generation_options(params)

    request.emit(
        "llm.request",
        message=f"calling {provider.capabilities.label} for node '{request.node.id}'",
        data={
            "prompt_chars": len(prompt),
            "model": provider.capabilities.model,
            "effort": options.effort,
            "temperature": options.temperature,
        },
    )

    try:
        completion = provider.complete(prompt, system=params.get("system"), options=options)
    except LLMRefusal as exc:
        # A safety decline is a content outcome, not a transport failure — say so
        # plainly instead of surfacing it as an opaque provider error.
        raise NodeExecutionError(
            f"the model declined the request at node '{request.node.id}'"
            + (f" (category: {exc.category})" if exc.category else "")
        ) from exc
    except LLMProxyError as exc:
        raise NodeExecutionError(f"node '{request.node.id}' could not reach the model: {exc}") from exc

    if completion.ignored_options:
        # Say so rather than letting a run report imply a knob took effect. A
        # temperature set on a node running against Claude, or an effort level
        # set against Llama, is silently meaningless otherwise.
        request.emit(
            "llm.options_ignored",
            level=LogLevel.WARN,
            message=(
                f"{provider.capabilities.label} does not support "
                f"{', '.join(completion.ignored_options)}; those settings had no effect"
            ),
            data={"ignored": list(completion.ignored_options)},
        )

    if completion.metadata.get("truncated"):
        request.emit(
            "llm.truncated",
            level=LogLevel.WARN,
            message=(
                f"'{request.node.id}' hit the output token limit; the answer is cut off. "
                "Raise max_tokens on this step."
            ),
        )

    request.emit(
        "llm.response",
        message=f"model returned {completion.output_tokens} output tokens",
        data={"model": completion.model, "stop_reason": completion.stop_reason},
    )

    return ExecutionOutcome(
        output={"text": completion.text},
        derived_context={
            "provider": provider.capabilities.provider,
            "model": completion.model,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "stop_reason": completion.stop_reason,
            "ignored_options": list(completion.ignored_options),
        },
    )


def _generation_options(params: dict[str, Any]) -> GenerationOptions:
    """Read the node's generation knobs.

    A node may carry settings for a backend it is not currently running against
    — that is deliberate, so a workflow stays portable between models. The
    provider decides what to honour.
    """
    stop = params.get("stop")
    if isinstance(stop, str):
        stop = [stop]

    return GenerationOptions(
        max_tokens=_optional_int(params.get("max_tokens")),
        effort=params.get("effort") or None,
        temperature=_optional_float(params.get("temperature")),
        top_p=_optional_float(params.get("top_p")),
        stop=tuple(stop or ()),
    )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None



# ---------------------------------------------------------------------------
# agent — a role with skills, a loop, and memory
# ---------------------------------------------------------------------------


def execute_agent(request: ExecutionRequest) -> ExecutionOutcome:
    """Run a stored agent against an objective built from this node's inputs.

    The node holds the objective template and nothing else. Everything that makes
    the agent what it is — its role, its skills, its memory — lives on the stored
    agent, so improving it improves every workflow that uses it.
    """
    from agents import registry as agent_registry  # noqa: PLC0415 - avoid import cycle
    from agents import runtime as agent_runtime  # noqa: PLC0415

    node = request.node
    if not node.agent_id:
        raise NodeExecutionError(f"agent node '{node.id}' names no agent")

    try:
        agent = agent_registry.get(request.tenant_id, node.agent_id)
    except agent_registry.AgentNotFound as exc:
        raise NodeExecutionError(
            f"node '{node.id}' references agent '{node.agent_id}', which no longer exists"
        ) from exc

    params = node.params or {}
    template = params.get("objective_template")
    objective = (
        render_template(template, request.inputs)
        if template
        else (agent.objective or str(request.inputs.get("goal") or "")).strip()
    )
    if not objective:
        raise NodeExecutionError(
            f"agent node '{node.id}' has no objective; set one on the node or the agent"
        )

    request.emit(
        "agent.started",
        message=f"'{agent.name}' starting — {len(agent.skill_ids)} skill(s) available",
        data={
            "agent": agent.name,
            "agent_id": agent.id,
            "max_iterations": agent.max_iterations,
        },
    )

    try:
        result = agent_runtime.run_agent(
            agent,
            objective,
            agent_runtime.AgentRunContext(
                tenant_id=request.tenant_id,
                run_id=request.run_id,
                step_execution_id=request.step_execution_id,
                workflow_memory_scope=request.workflow_memory_scope,
                allow_side_effects=request.allow_side_effects,
                emit=request.emit,
            ),
        )
    except agent_runtime.AgentError as exc:
        raise NodeExecutionError(str(exc)) from exc

    if result.status == agent_runtime.BUDGET_EXHAUSTED:
        # Say so rather than presenting a partial answer as finished.
        request.emit(
            "agent.incomplete",
            level=LogLevel.WARN,
            message=(
                f"'{agent.name}' used all {agent.max_iterations} iterations without "
                "concluding; the answer may be partial"
            ),
        )

    request.emit(
        "agent.finished",
        message=(
            f"'{agent.name}' finished in {result.iterations} iteration(s)"
            + (f", using {', '.join(dict.fromkeys(result.skills_used))}" if result.skills_used else "")
        ),
        data={"status": result.status, "skills_used": result.skills_used},
    )

    return ExecutionOutcome(
        output={"text": result.text},
        derived_context={
            "agent": agent.name,
            "agent_id": agent.id,
            "agent_version": agent.version,
            "status": result.status,
            "iterations": result.iterations,
            "skills_used": result.skills_used,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            # The reasoning path, so the run report shows how it got there.
            "turns": [turn.model_dump(mode="json") for turn in result.turns],
            # And, on a thinking model, what it worked through before each turn.
            "reasoning": result.reasoning,
        },
    )


# ---------------------------------------------------------------------------
# rag
# ---------------------------------------------------------------------------


def execute_rag(request: ExecutionRequest) -> ExecutionOutcome:
    node = request.node
    params = node.params or {}

    if not node.knowledge_handle:
        raise NodeExecutionError(f"rag node '{node.id}' has no knowledge_handle")

    query = request.inputs.get("query")
    if not query and params.get("query_template"):
        query = render_template(params["query_template"], request.inputs)
    if not query:
        raise NodeExecutionError(
            f"rag node '{node.id}' has no query; bind one to 'query' or set a query_template"
        )

    top_k = int(params.get("top_k", 4))
    try:
        result = retrieve(
            RetrievalRequest(
                handle=node.knowledge_handle,
                tenant_id=request.tenant_id,
                query=str(query),
                top_k=top_k,
            )
        )
    except KnowledgeError as exc:
        raise NodeExecutionError(f"node '{node.id}': {exc}") from exc

    request.emit(
        "rag.retrieved",
        message=f"retrieved {len(result.chunks)} chunk(s) from {node.knowledge_handle}",
        data={
            "handle": node.knowledge_handle,
            "top_score": result.chunks[0].score if result.chunks else 0.0,
        },
    )

    return ExecutionOutcome(
        output={
            "context": result.as_context(),
            "chunks": [chunk.model_dump(mode="json") for chunk in result.chunks],
            "query": str(query),
        },
        derived_context={"chunk_count": len(result.chunks), "handle": node.knowledge_handle},
    )


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------


def execute_transform(request: ExecutionRequest) -> ExecutionOutcome:
    template = (request.node.params or {}).get("template")
    if not template:
        raise NodeExecutionError(f"transform node '{request.node.id}' has no template")
    return ExecutionOutcome(output={"text": render_template(template, request.inputs)})


# ---------------------------------------------------------------------------
# branch
# ---------------------------------------------------------------------------

COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: _coerce(a) == _coerce(b),
    "!=": lambda a, b: _coerce(a) != _coerce(b),
    ">": lambda a, b: _numeric(a) > _numeric(b),
    ">=": lambda a, b: _numeric(a) >= _numeric(b),
    "<": lambda a, b: _numeric(a) < _numeric(b),
    "<=": lambda a, b: _numeric(a) <= _numeric(b),
    "contains": lambda a, b: str(b).lower() in str(a).lower(),
    "not_contains": lambda a, b: str(b).lower() not in str(a).lower(),
    "is_empty": lambda a, _b: not str(a).strip(),
    "is_not_empty": lambda a, _b: bool(str(a).strip()),
}


def evaluate_branch(request: ExecutionRequest) -> ExecutionOutcome:
    """Decide which outgoing edge a branch node selects.

    Pure: it only compares values already present in run context. That is what
    lets the planner resolve a branch inline rather than dispatching it as its
    own queued job.
    """
    params = request.node.params or {}
    operator = str(params.get("operator", "==")).lower()
    comparator = COMPARATORS.get(operator)
    if comparator is None:
        raise NodeExecutionError(
            f"branch node '{request.node.id}' uses unknown operator '{operator}'; "
            f"expected one of {sorted(COMPARATORS)}"
        )

    try:
        left = resolve(params.get("left", ""), request.context)
        right = resolve(params.get("right", ""), request.context)
    except BindingError as exc:
        raise NodeExecutionError(f"branch node '{request.node.id}': {exc}") from exc

    decision = bool(comparator(left, right))
    request.emit(
        "branch.evaluated",
        message=f"'{request.node.id}' took the {'true' if decision else 'false'} path",
        data={"operator": operator, "left": _preview(left), "right": _preview(right)},
    )
    return ExecutionOutcome(
        output={"decision": decision, "evaluated": _preview(left)},
        derived_context={"operator": operator},
        branch=decision,
    )


def _coerce(value: Any) -> Any:
    """Compare 5 and "5" as equal — canvas params arrive as strings from JSON."""
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped) if "." in stripped else int(stripped)
        except ValueError:
            return stripped.lower()
    if isinstance(value, bool):
        return value
    return value


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise NodeExecutionError(
            f"cannot compare '{value}' numerically; use == or contains for text"
        ) from exc


def _preview(value: Any, limit: int = 200) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


# ---------------------------------------------------------------------------
# http (the only executor with an external side effect)
# ---------------------------------------------------------------------------


def execute_http(request: ExecutionRequest) -> ExecutionOutcome:
    """Mandate §3.A — PRE-CHECK, then EXECUTE & COMMIT.

    The gate is keyed on (run_id, step_execution_id, "http_request"), so however
    many times the broker redelivers this job, the third party is called once.
    """
    params = request.node.params or {}
    method = str(params.get("method", "GET")).upper()
    url = str(params.get("url") or request.inputs.get("url") or "").strip()
    if not url:
        raise NodeExecutionError(f"http node '{request.node.id}' has no url")

    _assert_host_allowed(url, request.node.id)

    with unit_of_work() as session:
        gate = IdempotencyGate(
            session, request.run_id, request.step_execution_id, "http_request"
        )
        claim = gate.pre_check()

    if not claim.should_execute:
        if claim.in_flight_elsewhere:
            raise NodeExecutionError(
                f"node '{request.node.id}' is already in flight on another worker; "
                "this delivery is a duplicate"
            )
        request.emit(
            "http.replayed",
            message=f"'{request.node.id}' already completed for this run; replaying result",
            data={"url": url},
        )
        return ExecutionOutcome(
            output=claim.prior_response or {}, derived_context={"idempotent_replay": True}
        )

    settings = get_settings()
    body = params.get("body")
    if isinstance(body, str) and body:
        body = render_template(body, request.inputs)

    request.emit(
        "http.request",
        message=f"{method} {url}",
        data={"method": method, "url": url, "headers": params.get("headers") or {}},
    )

    try:
        import httpx  # noqa: PLC0415 - only needed by this node type

        with httpx.Client(timeout=settings.http_node_timeout_seconds) as client:
            response = client.request(
                method,
                url,
                headers=params.get("headers") or {},
                content=body if isinstance(body, str) else None,
                json=body if isinstance(body, dict) else None,
            )
        payload: dict[str, Any] = {
            "status": response.status_code,
            "body": _decode(response),
            "ok": response.is_success,
        }
    except Exception as exc:
        # Release the claim so a genuine retry is allowed to try again.
        with unit_of_work() as session:
            IdempotencyGate(
                session, request.run_id, request.step_execution_id, "http_request"
            ).abandon(str(exc))
        raise NodeExecutionError(f"node '{request.node.id}' request failed: {exc}") from exc

    # COMMIT: record success before the job is acknowledged.
    with unit_of_work() as session:
        IdempotencyGate(
            session, request.run_id, request.step_execution_id, "http_request"
        ).commit(payload, external_ref=str(payload["status"]))

    if not payload["ok"]:
        raise NodeExecutionError(
            f"node '{request.node.id}' got HTTP {payload['status']} from {url}"
        )

    return ExecutionOutcome(output=payload, derived_context={"url": url, "method": method})


def _assert_host_allowed(url: str, node_id: str) -> None:
    """Default-deny egress.

    A workflow is user-authored content that the platform executes server-side,
    so an unrestricted HTTP node is an SSRF primitive. Hosts must be explicitly
    allow-listed by an operator via `CWAP_HTTP_ALLOWLIST`.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        scheme = parsed.scheme or "none"
        raise NodeExecutionError(
            f"http node '{node_id}' must use http or https, got '{scheme}'"
        )

    host = (parsed.hostname or "").lower()
    allowed = get_settings().http_allowed_hosts
    if not allowed:
        raise NodeExecutionError(
            f"http node '{node_id}' is blocked: no outbound hosts are allow-listed. "
            "Set CWAP_HTTP_ALLOWLIST to enable external calls."
        )
    if host not in allowed and not any(
        host.endswith(f".{entry}") for entry in allowed
    ):
        raise NodeExecutionError(
            f"http node '{node_id}' is blocked: host '{host}' is not in the allow-list"
        )


def _decode(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text[:10000]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

EXECUTORS: dict[NodeType, Executor] = {
    NodeType.INPUT: execute_input,
    NodeType.AGENT: execute_agent,
    NodeType.LLM: execute_llm,
    NodeType.RAG_RETRIEVE: execute_rag,
    NodeType.TRANSFORM: execute_transform,
    NodeType.BRANCH: evaluate_branch,
    NodeType.HTTP_REQUEST: execute_http,
    NodeType.OUTPUT: execute_output,
}

#: Node types whose execution can be observed outside the platform, and which
#: therefore must pass through the idempotency gate before running. Agents are
#: included because a skill they invoke may reach outward.
SIDE_EFFECTING = frozenset({NodeType.HTTP_REQUEST, NodeType.AGENT})


def executor_for(node_type: NodeType) -> Executor:
    try:
        return EXECUTORS[node_type]
    except KeyError as exc:  # pragma: no cover - contract enum is closed
        raise NodeExecutionError(f"no executor registered for node type '{node_type}'") from exc


#: Re-exported so the worker can label log lines without importing LogLevel.
__all__ = [
    "COMPARATORS",
    "EXECUTORS",
    "SIDE_EFFECTING",
    "ExecutionOutcome",
    "ExecutionRequest",
    "Executor",
    "LogLevel",
    "NodeExecutionError",
    "executor_for",
]
