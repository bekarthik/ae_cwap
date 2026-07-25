"""The same fixture server, over streamable HTTP instead of stdio.

Every MCP test the platform had ran over stdio, so `_open_transport`'s HTTP
branch — the one every hosted server in the directory actually uses — was
never executed by anything. A user reporting that GitHub's server hung had no
test that could confirm or deny the platform's side of it.

Launched as a subprocess on a port the caller picks, so a test can talk to a
real streamable-HTTP MCP server: real handshake, real session id, real
`tools/list` over the wire.
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP


def build(slow_tool_seconds: float = 0.0) -> FastMCP:
    server = FastMCP("fixture-http-host")

    @server.tool(annotations={"readOnlyHint": True})
    def read_file(path: str) -> str:
        """Read a file from the repository."""
        return f"contents of {path}"

    @server.tool()
    def create_pull_request(title: str, body: str) -> str:
        """Open a pull request."""
        return f"opened '{title}'"

    return server


if __name__ == "__main__":
    port = int(sys.argv[1])
    app = build()
    app.settings.host = "127.0.0.1"
    app.settings.port = port
    # `log_level` down so a failing test's output is the assertion, not uvicorn.
    app.settings.log_level = "warning"
    app.run(transport="streamable-http")
