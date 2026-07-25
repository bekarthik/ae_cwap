"""MCP — connecting the platform to tools it did not build.

The Model Context Protocol is how an agent reaches a system nobody here wrote a
connector for: GitHub, a filesystem, a database, an internal service. A tenant
connects a server, the platform reads what tools it advertises, and each becomes
a skill an agent can be given. Nothing about a workflow changes — an agent that
can read a repository holds a skill, exactly like one that can summarise text.

Transport is the security boundary, and the two are not comparable:

* **stdio** launches a process on the worker. That is arbitrary code execution
  with the worker's privileges, so a command must be allow-listed by an operator
  before any tenant can name it. Without that, "connect an MCP server" is a
  remote shell.
* **http** talks to a URL. That is egress, and goes through the same host
  allow-list an HTTP node does.

`MCPServerConfig` is what a tenant supplies; secrets live beside it encrypted and
are never part of this model, so a server can be listed, exported and logged
without leaking a token.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from cwap_contracts.v1.base import ContractModel


class MCPTransport(str, Enum):
    #: The platform launches a process and speaks JSON-RPC over its pipes.
    STDIO = "stdio"
    #: Streamable HTTP to a URL the server already hosts.
    HTTP = "http"


class MCPTool(ContractModel):
    """One tool a server advertises, as reported by `tools/list`."""

    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=2000)
    #: The server's own JSON schema, passed to the model verbatim.
    input_schema: dict[str, Any] = Field(default_factory=dict)
    #: Servers may mark a tool read-only; when they do, the platform can offer it
    #: on a run that holds no write scope.
    read_only: bool = False


class MCPServerConfig(ContractModel):
    """How to reach one MCP server. Never carries credentials."""

    transport: MCPTransport
    #: stdio: the executable. http: unused.
    command: str = Field(default="", max_length=256)
    args: list[str] = Field(default_factory=list, max_length=32)
    #: stdio: extra environment. Values that are secret belong in credentials.
    env: dict[str, str] = Field(default_factory=dict)
    #: http: the endpoint.
    url: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _transport_has_what_it_needs(self) -> MCPServerConfig:
        if self.transport is MCPTransport.STDIO and not self.command:
            raise ValueError("a stdio MCP server needs a command")
        if self.transport is MCPTransport.HTTP and not self.url:
            raise ValueError("an http MCP server needs a url")
        if self.transport is MCPTransport.HTTP and not self.url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("an MCP server url must be http or https")
        return self


class MCPServerRecord(ContractModel):
    """A connected server, as stored and as shown to a user."""

    id: str = Field(..., min_length=1, max_length=64)
    tenant_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = Field(default="", max_length=1000)
    config: MCPServerConfig
    enabled: bool = True
    #: Whether a credential is stored for this server. Never the credential.
    has_credentials: bool = False
    tools: list[MCPTool] = Field(default_factory=list)
    last_connected_at: str = ""
    last_error: str = ""

    @property
    def is_connected(self) -> bool:
        return bool(self.last_connected_at) and not self.last_error


class MCPConnectionResult(ContractModel):
    """What happened when the platform tried to reach a server.

    Returned rather than raised: "your command is not on the allow-list" and
    "the server started but crashed" are both things a user has to read and act
    on, and a stack trace tells them less than the server's own message.
    """

    ok: bool
    message: str = Field(default="", max_length=4000)
    tools: list[MCPTool] = Field(default_factory=list)
    #: Set when the failure is a policy decision rather than a fault, so the UI
    #: can point at the operator instead of suggesting a retry.
    blocked: bool = False
