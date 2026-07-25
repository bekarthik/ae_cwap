"""A real MCP server, for tests that should not settle for a mock.

Launched as a subprocess over stdio by `test_mcp.py`, so the tests exercise the
actual protocol — handshake, `tools/list`, `tools/call`, content blocks — rather
than a stand-in that agrees with whatever the client happens to do.

It imitates the shape of a code-hosting server, because that is the example the
platform has to work for: a read-only tool, a write tool, one that takes a
structured argument, and one that fails.
"""

from __future__ import annotations

import json
import sys

from mcp.server.fastmcp import FastMCP

server = FastMCP("fixture-code-host")

#: Written by `create_pull_request` so a test can prove a redelivered call did
#: not open two. Lives in the process, which is exactly the point — the platform
#: must not invoke the tool twice, not merely deduplicate afterwards.
_opened: list[dict] = []


@server.tool(annotations={"readOnlyHint": True})
def read_file(path: str) -> str:
    """Read a file from the repository."""
    return f"contents of {path}"


@server.tool(annotations={"readOnlyHint": True})
def search_code(query: str, limit: int = 5) -> str:
    """Search the repository for a string."""
    return json.dumps([f"match {index} for {query}" for index in range(min(limit, 3))])


@server.tool()
def create_pull_request(title: str, body: str, labels: list[str] | None = None) -> str:
    """Open a pull request. Takes a structured argument on purpose."""
    _opened.append({"title": title, "body": body, "labels": labels or []})
    return f"opened #{len(_opened)}: {title} with labels {labels or []}"


@server.tool()
def always_fails(reason: str) -> str:
    """A tool that raises, so failure handling is exercised."""
    raise RuntimeError(f"the server refused: {reason}")


if __name__ == "__main__":
    # stdout is the protocol channel; anything else printed there corrupts it.
    print("fixture server starting", file=sys.stderr)
    server.run()
