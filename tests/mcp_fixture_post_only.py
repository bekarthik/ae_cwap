"""A streamable-HTTP MCP server shaped like GitHub's: POST-only.

The MCP spec makes the server→client GET stream optional, and several hosted
servers — GitHub's `api.githubcopilot.com/mcp/` among them — decline it with
`405 Method Not Allowed`. That shape is not exotic and it is not broken; it is
the spec's own "server does not offer server-initiated messages" case.

It is also the shape that hangs the Python SDK
(modelcontextprotocol/python-sdk#1941), so the platform needs a test that talks
to a server behaving this way. FastMCP always offers the GET stream, so this is
written by hand: real HTTP, real SSE framing, real session id, and a flat 405
for GET.

Deliberately minimal. It answers `initialize`, `notifications/initialized` and
`tools/list` and nothing else, because the fault being reproduced happens during
the handshake and the listing that follows it.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SESSION_ID = "fixture-session-1"

#: What the server was asked for, so a test can assert the GET was attempted and
#: refused rather than inferring it from timing.
REQUEST_LOG: list[str] = []

#: A JSON-RPC method to accept and then never answer. Set to "initialize" or to
#: "tools/list" to stall exactly one phase of opening a connection, which is how
#: the two are told apart in the error a user sees.
STALL_ON: list[str] = []

#: Set to a content type to answer every POST with `200 OK` and that type
#: instead of MCP. What a captive portal, a proxy error page or a plain wrong
#: URL looks like from here: a perfectly successful HTTP response carrying
#: something that is not the protocol.
ANSWER_AS: list[str] = []


class PostOnlyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # noqa: D102 - silence the default logger
        pass

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        """No server-initiated stream here, exactly as GitHub's server answers."""
        REQUEST_LOG.append("GET")
        self.send_response(405)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_DELETE(self) -> None:  # noqa: N802
        REQUEST_LOG.append("DELETE")
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        message = json.loads(raw or b"{}")
        method = message.get("method", "")
        REQUEST_LOG.append(f"POST {method}")

        if method.startswith("notifications/"):
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if ANSWER_AS:
            body = b"<html><body>Sign in to continue</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", ANSWER_AS[0])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if method in STALL_ON:
            # Headers, the right content type, and then silence — the shape of
            # a stalled phase. Held rather than closed, because a closed socket
            # is an error and this is specifically the case that is not one.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            if method == "initialize":
                self.send_header("Mcp-Session-Id", SESSION_ID)
            self.end_headers()
            self.wfile.flush()
            time.sleep(600)
            return

        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "post-only-fixture", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file from the repository.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                        "annotations": {"readOnlyHint": True},
                    }
                ]
            }
        else:
            result = {}

        payload = json.dumps({"jsonrpc": "2.0", "id": message.get("id"), "result": result})
        frame = f"event: message\ndata: {payload}\n\n".encode()

        self.send_response(200)
        # SSE rather than JSON, because that is what the servers in question
        # send and it is the branch the deadlock lives in.
        self.send_header("Content-Type", "text/event-stream")
        if method == "initialize":
            self.send_header("Mcp-Session-Id", SESSION_ID)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.write(b"%X\r\n" % len(frame) + frame + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def serve(port: int) -> ThreadingHTTPServer:
    """Start the fixture on a background thread and return it for shutdown."""
    REQUEST_LOG.clear()
    STALL_ON.clear()
    ANSWER_AS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", port), PostOnlyHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    serve(int(sys.argv[1])).serve_forever()
