"""Parameter mapping between connected nodes.

Epic 2's "parameter mapping between connected components" is this file. An edge
carries bindings like `{"search_term": "$output.topic"}`; this resolves those
expressions against run state at dispatch time.

Three namespaces, deliberately few:

    $run.input.<key>          the values the run was started with
    $output.<key>             the immediately preceding step's output
    $steps.<node_id>.<key>    any completed step's output, by node id

Anything that is not a `$` expression is passed through as a literal, so a user
can type a constant into a binding without escaping it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}")


class BindingError(ValueError):
    """A binding referenced something that does not exist in run state.

    Raised rather than resolved to None: a silently empty prompt variable is the
    kind of failure that produces a plausible-looking wrong answer.
    """


@dataclass
class RunContext:
    """Everything a binding can see at the moment a step is dispatched."""

    run_id: str
    inputs: dict[str, Any] = field(default_factory=dict)
    #: node_id -> that node's output dict
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The node whose output `$output.*` refers to.
    previous_node_id: str | None = None

    def record(self, node_id: str, output: dict[str, Any]) -> None:
        self.steps[node_id] = output
        self.previous_node_id = node_id

    @property
    def previous_output(self) -> dict[str, Any]:
        if self.previous_node_id is None:
            return {}
        return self.steps.get(self.previous_node_id, {})


def resolve(expression: str, context: RunContext) -> Any:
    """Resolve one binding expression."""
    if not isinstance(expression, str) or not expression.startswith("$"):
        return expression

    parts = expression[1:].split(".")
    root = parts[0]

    if root == "run":
        if len(parts) < 3 or parts[1] != "input":
            raise BindingError(
                f"'{expression}' is not a valid run reference; expected $run.input.<key>"
            )
        return _dig(context.inputs, parts[2:], expression)

    if root == "output":
        if context.previous_node_id is None:
            raise BindingError(
                f"'{expression}' refers to the previous step's output, but this is the "
                "first step of the run"
            )
        return _dig(context.previous_output, parts[1:], expression)

    if root == "steps":
        if len(parts) < 3:
            raise BindingError(
                f"'{expression}' is not a valid step reference; expected $steps.<node>.<key>"
            )
        node_id = parts[1]
        if node_id not in context.steps:
            raise BindingError(
                f"'{expression}' refers to step '{node_id}', which has not completed "
                f"(completed: {sorted(context.steps)})"
            )
        return _dig(context.steps[node_id], parts[2:], expression)

    raise BindingError(
        f"unknown namespace '${root}' in '{expression}'; expected $run, $output or $steps"
    )


def _dig(source: Any, path: list[str], expression: str) -> Any:
    current = source
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
            continue
        if isinstance(current, list) and key.isdigit() and int(key) < len(current):
            current = current[int(key)]
            continue
        available = sorted(current) if isinstance(current, dict) else type(current).__name__
        raise BindingError(f"'{expression}' has no value at '{key}' (available: {available})")
    return current


def resolve_bindings(bindings: dict[str, str], context: RunContext) -> dict[str, Any]:
    """Resolve every binding on an edge into concrete node inputs."""
    return {name: resolve(expression, context) for name, expression in bindings.items()}


def render_template(template: str, values: dict[str, Any]) -> str:
    """Substitute `{{name}}` placeholders.

    Intentionally not a general template engine — no logic, no attribute access,
    no code execution. Node parameters are user-authored and reach this from the
    browser, so the only safe surface is plain named substitution.
    """
    missing: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in values:
            return _stringify(values[name])
        root = name.split(".")[0]
        if root in values:
            try:
                return _stringify(_dig(values[root], name.split(".")[1:], f"{{{{{name}}}}}"))
            except BindingError:
                missing.append(name)
                return match.group(0)
        missing.append(name)
        return match.group(0)

    rendered = _TEMPLATE_RE.sub(substitute, template)
    if missing:
        raise BindingError(
            f"template referenced undefined variable(s) {sorted(set(missing))}; "
            f"available: {sorted(values)}"
        )
    return rendered


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(item) for item in value)
    if isinstance(value, dict):
        import json  # noqa: PLC0415 - only needed on this branch

        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def template_variables(template: str) -> set[str]:
    """Names a template expects. Used to validate a node before a run starts."""
    return {match.group(1) for match in _TEMPLATE_RE.finditer(template)}
