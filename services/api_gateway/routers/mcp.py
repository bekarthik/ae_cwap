"""Connecting MCP servers.

Connect, test, list, re-sync, disable, disconnect. Connecting is the interesting
one: it verifies the server before storing it, because a saved server that has
never connected is a row that looks like a working integration and is not.

Credentials go in and never come out — the same rule as model provider keys.
"""

from __future__ import annotations

from cwap_contracts.v3 import MCPServerConfig, MCPServerRecord
from fastapi import APIRouter, Depends, HTTPException
from mcp_connect import directory, policy, registry
from pydantic import BaseModel, ConfigDict, Field

from api_gateway.security import Principal, current_principal

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class ConnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = Field(default="", max_length=1000)
    config: MCPServerConfig
    #: Secret values: auth headers for an HTTP server, environment values (an
    #: access token) for a stdio one. Stored encrypted, never returned.
    credentials: dict[str, str] = Field(default_factory=dict)


class TestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: MCPServerConfig
    credentials: dict[str, str] = Field(default_factory=dict)


@router.get("")
def list_servers(
    principal: Principal = Depends(current_principal),
) -> dict[str, object]:
    """Connected servers, and what this deployment permits connecting to."""
    return {
        "servers": [
            server.model_dump(mode="json") for server in registry.list_all(principal.tenant_id)
        ],
        # Surfaced so the UI can say "stdio servers are disabled here" before a
        # user fills in a form that will be refused.
        "policy": policy.describe(),
    }


@router.get("/directory")
def list_directory(
    principal: Principal = Depends(current_principal),
) -> dict[str, object]:
    """Servers that can be connected without an administrator being asked first.

    Ahead of the connected list on purpose: "which of these do you want" is a
    better first question than an empty URL field, and every entry already knows
    its endpoint, what it is for and which credential it will ask for.
    """
    return {"servers": directory.as_dicts(), "policy": policy.describe()}


@router.post("/test")
def test_server(
    request: TestRequest, principal: Principal = Depends(current_principal)
) -> dict[str, object]:
    """Try a server without storing anything."""
    return registry.test(request.config, request.credentials).model_dump(mode="json")


@router.post("", status_code=201)
def connect_server(
    request: ConnectRequest, principal: Principal = Depends(current_principal)
) -> dict[str, object]:
    """Verify a server, store it, and import its tools as skills."""
    try:
        result = registry.connect(
            principal.tenant_id,
            name=request.name,
            config=request.config,
            description=request.description,
            credentials=request.credentials,
        )
    except registry.MCPRegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if not result.ok:
        # 400 for a fault the user can fix, 403 when the deployment forbids it —
        # the distinction is what tells them whether to retry or ask an admin.
        raise HTTPException(status_code=403 if result.blocked else 400, detail=result.message)

    return result.model_dump(mode="json")


@router.get("/{server_id}", response_model=MCPServerRecord)
def get_server(
    server_id: str, principal: Principal = Depends(current_principal)
) -> MCPServerRecord:
    try:
        return registry.get(principal.tenant_id, server_id)
    except registry.MCPServerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{server_id}/sync")
def sync_server(
    server_id: str, principal: Principal = Depends(current_principal)
) -> dict[str, object]:
    """Re-read the server's tools. Picks up tools it has gained or lost."""
    try:
        count = registry.sync_skills(principal.tenant_id, server_id)
    except registry.MCPServerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    server = registry.get(principal.tenant_id, server_id)
    return {"skills": count, "server": server.model_dump(mode="json")}


@router.post("/{server_id}/enabled")
def set_enabled(
    server_id: str,
    enabled: bool = True,
    principal: Principal = Depends(current_principal),
) -> MCPServerRecord:
    """Switch a server off without losing it, or back on."""
    try:
        return registry.set_enabled(principal.tenant_id, server_id, enabled)
    except registry.MCPServerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{server_id}", status_code=204)
def disconnect_server(
    server_id: str, principal: Principal = Depends(current_principal)
) -> None:
    """Remove a server and the skills it contributed."""
    if not registry.disconnect(principal.tenant_id, server_id):
        raise HTTPException(status_code=404, detail="server not found")
