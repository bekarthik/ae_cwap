"""What a tenant is allowed to connect to.

Connecting an MCP server is not like adding a workflow step. The two transports
have very different blast radii and each needs its own gate:

**stdio launches a process on the worker.** With no gate, "connect an MCP server"
is a remote shell with the worker's privileges — a tenant could name `bash` and
be done. So a command must be allow-listed by an operator, by name, before any
tenant can reference it. The allow-list is empty by default, which means stdio
servers are off until somebody deliberately turns them on.

**http is egress**, and reuses the same host allow-list an HTTP node goes
through, so an operator who has already decided which hosts this deployment may
reach does not have to decide again in a second place.

Both gates are deliberately operator-side. A tenant self-service surface that can
grant itself either one is not a gate.
"""

from __future__ import annotations

import os
import shutil
from urllib.parse import urlparse

from cwap_common.settings import get_settings
from cwap_contracts.v3 import MCPServerConfig, MCPTransport


class MCPPolicyError(RuntimeError):
    """A connection was refused by policy rather than failing technically.

    Separate from a connection error because the fix is different: nothing the
    user retries will help, and the message has to point at the operator.
    """


def allowed_commands() -> frozenset[str]:
    """Executables an operator has permitted stdio servers to launch."""
    raw = os.environ.get("CWAP_MCP_ALLOWED_COMMANDS", "").strip()
    if not raw:
        return frozenset()
    return frozenset(entry.strip() for entry in raw.split(",") if entry.strip())


def assert_permitted(config: MCPServerConfig) -> None:
    """Raise `MCPPolicyError` unless this deployment permits the connection."""
    if config.transport is MCPTransport.STDIO:
        _assert_command_permitted(config.command)
    else:
        _assert_host_permitted(config.url)


def _assert_command_permitted(command: str) -> None:
    allowed = allowed_commands()
    if not allowed:
        raise MCPPolicyError(
            "stdio MCP servers are disabled on this deployment. They launch a "
            "process on the worker, so an administrator must allow specific "
            "commands with CWAP_MCP_ALLOWED_COMMANDS. An HTTP MCP server needs "
            "no such permission."
        )

    # An entry is either a bare name (`npx` — resolved through PATH, which the
    # operator controls) or an absolute path (`/opt/tools/mcp-server`). A bare
    # entry deliberately does *not* authorise a path: otherwise an allow-list of
    # `npx` would be satisfied by `/tmp/uploaded/npx`, and the gate would be
    # decoration.
    wanted = command.strip()
    if wanted in allowed:
        permitted = wanted
    elif os.sep not in wanted and os.path.basename(wanted) in allowed:
        permitted = os.path.basename(wanted)
    else:
        raise MCPPolicyError(
            f"'{wanted}' is not an allowed MCP command on this deployment. "
            f"Permitted: {sorted(allowed)}. An allow-list entry matches either a "
            "bare command name resolved through PATH, or the exact absolute path "
            "written in the entry."
        )

    if os.sep in permitted:
        if not os.path.isfile(permitted) or not os.access(permitted, os.X_OK):
            raise MCPPolicyError(f"'{permitted}' is allow-listed but not executable")
    elif shutil.which(permitted) is None:
        raise MCPPolicyError(f"'{permitted}' is allow-listed but not installed on the worker")


def _assert_host_permitted(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        raise MCPPolicyError(f"'{url}' has no host")

    # A host the built-in directory names is already an answer to "which hosts
    # may this deployment reach" — decided in a reviewed file rather than in a
    # text box. Everything else still goes through the operator's allow-list.
    from mcp_connect import directory  # noqa: PLC0415 - avoids an import cycle

    if directory.vouches_for(host):
        return

    allowed = get_settings().http_allowed_hosts
    if not allowed:
        raise MCPPolicyError(
            f"'{host}' is not one of the servers this platform ships with, and no "
            "outbound hosts are allow-listed on this deployment. Pick a server "
            "from the list, or ask an administrator to add the host to "
            "CWAP_HTTP_ALLOWLIST."
        )
    if host not in allowed and not any(host.endswith(f".{entry}") for entry in allowed):
        raise MCPPolicyError(f"host '{host}' is not in the outbound allow-list")


def describe() -> dict[str, object]:
    """What this deployment permits, so the UI can say so before a user tries."""
    from mcp_connect import directory  # noqa: PLC0415 - avoids an import cycle

    return {
        "stdio_enabled": bool(allowed_commands()),
        "allowed_commands": sorted(allowed_commands()),
        "allowed_hosts": sorted(get_settings().http_allowed_hosts),
        "directory_enabled": directory.enabled(),
        "directory_hosts": sorted(directory.hosts()),
    }
