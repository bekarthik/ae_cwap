"""Connected MCP servers, and turning their tools into skills.

The design decision worth stating: an MCP tool becomes a **skill**, not a new
node type. Everything the platform already does for skills then applies to it
without a second implementation — an agent is given it the same way, the model
sees it in the same tool list, failures come back to the agent the same way, and
its lessons accrue to the same per-skill memory. An agent that can open a pull
request is an agent holding a skill, exactly like one that can summarise text.

Syncing is one-directional and idempotent: the server's tool list is the truth,
and a re-sync adds what appeared, updates what changed, and disables what the
server no longer advertises. Disabling rather than deleting is deliberate — an
agent still referencing a removed tool should report "that capability is gone",
not silently lose a skill from its list.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import MCPServer as MCPServerRow
from cwap_common.models import Skill as SkillRow
from cwap_common.secrets import decrypt, encrypt
from cwap_contracts.v4 import (
    MCPConnectionResult,
    MCPServerConfig,
    MCPServerRecord,
    MCPTool,
    SkillDefinition,
    SkillKind,
    SkillOrigin,
    SkillParameter,
)

from mcp_connect.client import MCPError, get_client
from mcp_connect.policy import MCPPolicyError

#: Skills imported from a server are named `<server>_<tool>` so two servers can
#: both offer `search` without colliding, and so a user reading an agent's skill
#: list can see where a capability came from.
NAME_SEPARATOR = "_"


class MCPServerNotFound(LookupError):
    """No such server for this tenant."""


class MCPRegistryError(RuntimeError):
    """A server could not be stored."""


# ---------------------------------------------------------------------------
# servers
# ---------------------------------------------------------------------------


def connect(
    tenant_id: str,
    *,
    name: str,
    config: MCPServerConfig,
    description: str = "",
    credentials: dict | None = None,
) -> MCPConnectionResult:
    """Verify a server, store it, and import its tools as skills.

    Verification happens *before* storage on purpose: a saved server that has
    never connected is a row that looks like a working integration and is not.
    """
    if find_by_name(tenant_id, name) is not None:
        raise MCPRegistryError(f"a server named '{name}' is already connected")

    probe = test(config, credentials or {})
    if not probe.ok:
        return probe

    server_id = f"mcp_{uuid.uuid4().hex[:20]}"
    with unit_of_work() as session:
        session.add(
            MCPServerRow(
                id=server_id,
                tenant_id=tenant_id,
                name=name,
                description=description,
                transport=config.transport.value,
                config=config.model_dump(mode="json"),
                credentials=encrypt(json.dumps(credentials or {})),
                tools=[tool.model_dump(mode="json") for tool in probe.tools],
                last_connected_at=datetime.now(timezone.utc),
                enabled=True,
            )
        )

    imported = sync_skills(tenant_id, server_id)
    return MCPConnectionResult(
        ok=True,
        message=(
            f"Connected. {len(probe.tools)} tool(s) available, "
            f"{imported} added as skills your agents can use."
        ),
        tools=probe.tools,
    )


def test(config: MCPServerConfig, credentials: dict | None = None) -> MCPConnectionResult:
    """Try a server without storing anything.

    Returns a result rather than raising: a refused command and a crashed server
    are both things the user reads and acts on, and they need different advice.
    """
    try:
        tools = get_client().probe(config, credentials or {})
    except MCPPolicyError as exc:
        return MCPConnectionResult(ok=False, message=str(exc), blocked=True)
    except MCPError as exc:
        return MCPConnectionResult(ok=False, message=str(exc))
    except Exception as exc:  # noqa: BLE001 - a server can fail in its own ways
        return MCPConnectionResult(ok=False, message=f"{type(exc).__name__}: {exc}")

    if not tools:
        return MCPConnectionResult(
            ok=True,
            message="Connected, but the server advertises no tools.",
            tools=[],
        )
    return MCPConnectionResult(
        ok=True, message=f"Connected. {len(tools)} tool(s) available.", tools=tools
    )


def get(tenant_id: str, server_id: str) -> MCPServerRecord:
    with read_only_session() as session:
        row = (
            session.query(MCPServerRow).filter_by(tenant_id=tenant_id, id=server_id).one_or_none()
        )
        if row is None:
            raise MCPServerNotFound(f"no MCP server '{server_id}' for this workspace")
        return _to_record(row)


def find_by_name(tenant_id: str, name: str) -> MCPServerRecord | None:
    with read_only_session() as session:
        row = session.query(MCPServerRow).filter_by(tenant_id=tenant_id, name=name).one_or_none()
        return _to_record(row) if row else None


def list_all(tenant_id: str) -> list[MCPServerRecord]:
    with read_only_session() as session:
        rows = (
            session.query(MCPServerRow)
            .filter_by(tenant_id=tenant_id)
            .order_by(MCPServerRow.name)
            .all()
        )
        return [_to_record(row) for row in rows]


def set_enabled(tenant_id: str, server_id: str, enabled: bool) -> MCPServerRecord:
    """Turn a server off without losing it, or back on.

    Disabling drops the live session and disables its skills, so an agent that
    holds one reports a clear "this capability is switched off" rather than
    hanging on a connection to something the user has deliberately stopped.
    """
    with unit_of_work() as session:
        row = (
            session.query(MCPServerRow).filter_by(tenant_id=tenant_id, id=server_id).one_or_none()
        )
        if row is None:
            raise MCPServerNotFound(server_id)
        row.enabled = enabled
        record = _to_record(row)

    get_client().disconnect(server_id)
    if enabled:
        sync_skills(tenant_id, server_id)
    else:
        _disable_skills(tenant_id, server_id)
    return record


def disconnect(tenant_id: str, server_id: str) -> bool:
    """Remove a server and the skills it contributed."""
    get_client().disconnect(server_id)
    # Resolved before the delete, and in Python rather than as a JSON path: the
    # set is small, and a query that reads inside a JSON column is written
    # differently on every dialect for no benefit here.
    doomed = _skill_ids_for(tenant_id, server_id)

    with unit_of_work() as session:
        removed = (
            session.query(MCPServerRow).filter_by(tenant_id=tenant_id, id=server_id).delete()
        )
        if removed and doomed:
            # The skills are references to something that no longer exists.
            session.query(SkillRow).filter(
                SkillRow.tenant_id == tenant_id, SkillRow.id.in_(doomed)
            ).delete(synchronize_session=False)
        return bool(removed)


# ---------------------------------------------------------------------------
# tools → skills
# ---------------------------------------------------------------------------


def sync_skills(tenant_id: str, server_id: str) -> int:
    """Reconcile this server's tools with the tenant's skills.

    Returns how many skills now exist for the server. Re-running is safe and is
    the intended way to pick up a server that has gained or lost tools.
    """
    record = get(tenant_id, server_id)
    if not record.enabled:
        return 0

    tools = _refresh_tools(tenant_id, record)
    wanted = {_skill_name(record.name, tool.name): tool for tool in tools}

    existing = {
        skill.name: skill
        for skill in _skills_for_server(tenant_id, server_id)
    }

    with unit_of_work() as session:
        for name, tool in wanted.items():
            row = session.query(SkillRow).filter_by(tenant_id=tenant_id, name=name).one_or_none()
            payload = _skill_payload(record, tool)
            if row is None:
                session.add(
                    SkillRow(
                        id=f"skl_{uuid.uuid4().hex[:20]}",
                        tenant_id=tenant_id,
                        name=name,
                        origin=SkillOrigin.MCP.value,
                        **payload,
                    )
                )
            else:
                # The server is the truth; a changed schema must reach the model.
                row.description = payload["description"]
                row.parameters = payload["parameters"]
                row.definition = payload["definition"]
                row.kind = payload["kind"]
                row.version += 1

        # A tool the server no longer advertises is marked unavailable rather
        # than deleted, so an agent still holding it says so instead of silently
        # losing a capability its objective depends on.
        for name, skill in existing.items():
            if name not in wanted:
                row = session.query(SkillRow).filter_by(tenant_id=tenant_id, id=skill.id).one()
                row.definition = {**(row.definition or {}), "unavailable": True}

    return len(wanted)


def _refresh_tools(tenant_id: str, record: MCPServerRecord) -> list[MCPTool]:
    """Ask the server what it has now, falling back to the cached list.

    A sync that fails should not erase a working integration's skills — the tools
    were there a moment ago and the server is probably restarting.
    """
    try:
        tools = get_client().list_tools(
            record.id, record.config, _credentials(tenant_id, record.id)
        )
    except (MCPError, MCPPolicyError) as exc:
        _record_error(tenant_id, record.id, str(exc))
        return record.tools

    _record_success(tenant_id, record.id, tools)
    return tools


def _skill_payload(record: MCPServerRecord, tool: MCPTool) -> dict:
    schema = tool.input_schema or {"type": "object", "properties": {}}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    return {
        "description": _describe(record, tool),
        "kind": SkillKind.MCP.value,
        "parameters": [
            param.model_dump(mode="json")
            for param in _parameters(properties, required)
        ],
        "definition": {
            "server_id": record.id,
            "server_name": record.name,
            "tool_name": tool.name,
            # Passed to the model verbatim: a paraphrase would drift from what
            # the server actually validates against.
            "input_schema": schema,
            "read_only": tool.read_only,
        },
    }


def _describe(record: MCPServerRecord, tool: MCPTool) -> str:
    described = tool.description.strip() or f"The '{tool.name}' tool."
    return f"{described} (via {record.name})"[:1000]


def _parameters(properties: dict, required: set) -> list[SkillParameter]:
    """A readable summary of the tool's schema, for the UI.

    The model gets the server's schema verbatim; these exist so a person looking
    at a skill can see what it takes without reading JSON Schema.
    """
    out: list[SkillParameter] = []
    for name, spec in list(properties.items())[:24]:
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", str(name)):
            continue
        declared = (spec or {}).get("type", "string")
        if isinstance(declared, list):
            declared = next((item for item in declared if item != "null"), "string")
        if declared not in {"string", "number", "integer", "boolean", "object", "array"}:
            declared = "string"
        out.append(
            SkillParameter(
                name=str(name),
                type=declared,
                description=str((spec or {}).get("description", ""))[:400],
                required=name in required,
            )
        )
    return out


def _skill_name(server_name: str, tool_name: str) -> str:
    """`<server>_<tool>`, normalised to the skill name pattern."""
    raw = f"{server_name}{NAME_SEPARATOR}{tool_name}"
    slug = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"mcp_{slug}"
    return slug[:64]


def _skills_for_server(tenant_id: str, server_id: str) -> list[SkillDefinition]:
    with read_only_session() as session:
        rows = (
            session.query(SkillRow)
            .filter_by(tenant_id=tenant_id, origin=SkillOrigin.MCP.value)
            .all()
        )
        return [
            _to_skill(row)
            for row in rows
            if (row.definition or {}).get("server_id") == server_id
        ]


def _skill_ids_for(tenant_id: str, server_id: str) -> list[str]:
    return [skill.id for skill in _skills_for_server(tenant_id, server_id)]


def _disable_skills(tenant_id: str, server_id: str) -> None:
    with unit_of_work() as session:
        for skill in _skills_for_server(tenant_id, server_id):
            row = session.query(SkillRow).filter_by(tenant_id=tenant_id, id=skill.id).one()
            row.definition = {**(row.definition or {}), "unavailable": True}


# ---------------------------------------------------------------------------
# storage helpers
# ---------------------------------------------------------------------------


def credentials_for(tenant_id: str, server_id: str) -> dict:
    """Decrypted credentials for a server. Never leaves the service layer."""
    return _credentials(tenant_id, server_id)


def _credentials(tenant_id: str, server_id: str) -> dict:
    with read_only_session() as session:
        row = (
            session.query(MCPServerRow).filter_by(tenant_id=tenant_id, id=server_id).one_or_none()
        )
        if row is None or not row.credentials:
            return {}
    try:
        loaded = json.loads(decrypt(row.credentials))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:  # noqa: BLE001 - a bad credential must not break listing
        return {}


def _record_success(tenant_id: str, server_id: str, tools: list[MCPTool]) -> None:
    with unit_of_work() as session:
        row = session.query(MCPServerRow).filter_by(tenant_id=tenant_id, id=server_id).one_or_none()
        if row is None:
            return
        row.tools = [tool.model_dump(mode="json") for tool in tools]
        row.last_connected_at = datetime.now(timezone.utc)
        row.last_error = ""


def _record_error(tenant_id: str, server_id: str, message: str) -> None:
    with unit_of_work() as session:
        row = session.query(MCPServerRow).filter_by(tenant_id=tenant_id, id=server_id).one_or_none()
        if row is not None:
            row.last_error = message[:2000]


def _to_record(row: MCPServerRow) -> MCPServerRecord:
    return MCPServerRecord(
        id=row.id,
        tenant_id=row.tenant_id,
        name=row.name,
        description=row.description or "",
        config=MCPServerConfig.model_validate(row.config or {}),
        enabled=bool(row.enabled),
        has_credentials=bool(row.credentials),
        tools=[MCPTool.model_validate(tool) for tool in (row.tools or [])],
        last_connected_at=row.last_connected_at.isoformat() if row.last_connected_at else "",
        last_error=row.last_error or "",
    )


def _to_skill(row: SkillRow) -> SkillDefinition:
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
