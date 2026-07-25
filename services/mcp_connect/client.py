"""Talking to MCP servers from a synchronous worker.

The MCP SDK is async and this platform is not, which is a real problem rather
than a cosmetic one. The naive fix — `asyncio.run()` per tool call — reconnects
every time, and for a stdio server that means spawning and killing a process for
each invocation. An agent that calls three tools would start three processes.

So there is one background event loop for the whole process, and sessions live on
it and are reused. A synchronous caller submits a coroutine and blocks for the
result. Sessions are keyed by server id, opened lazily, and closed when the
server is edited, deleted, or the process shuts down.

Everything here is transport-neutral above the connect step: `list_tools` and
`call_tool` do not know whether they are talking to a subprocess or a URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from concurrent.futures import Future
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from cwap_contracts.v3 import MCPServerConfig, MCPTool, MCPTransport

from mcp_connect.policy import MCPPolicyError, assert_permitted

logger = logging.getLogger("cwap.mcp")

#: A server that cannot complete a handshake in this long is not usable for an
#: agent loop, where a person is watching a run.
CONNECT_TIMEOUT = 30.0
CALL_TIMEOUT = 120.0


class MCPError(RuntimeError):
    """A server could not be reached, or a tool call failed."""


def explain(exc: BaseException) -> str:
    """A message a person can act on, out of whatever the SDK raised.

    The transport runs inside an anyio task group, so a refused connection
    arrives as an ExceptionGroup whose `str()` is "unhandled errors in a
    TaskGroup (1 sub-exception)" — accurate, and worthless to somebody who just
    typed a URL. The leaves of the group are the actual answer: a DNS failure, a
    401, a certificate. Nested groups are flattened, since the nesting is an
    implementation detail of how the transport is supervised.
    """
    leaves: list[str] = []

    def walk(error: BaseException) -> None:
        inner = getattr(error, "exceptions", None)
        if inner:
            for sub in inner:
                walk(sub)
            return
        text = str(error).strip() or type(error).__name__
        if text not in leaves:
            leaves.append(text)

    walk(exc)
    # Three is enough to see a pattern without pasting a stack into a dialog.
    return "; ".join(leaves[:3]) or f"{type(exc).__name__}"


class _LoopThread:
    """One asyncio loop, on one daemon thread, for the whole process.

    A per-call loop would be simpler and would also close every session it
    opened, which defeats the point of keeping them.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                threading.Thread(
                    target=self._loop.run_forever,
                    name="cwap-mcp-loop",
                    daemon=True,
                ).start()
            return self._loop

    def submit(self, coroutine, timeout: float) -> Any:
        future: Future = asyncio.run_coroutine_threadsafe(coroutine, self.loop())
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            future.cancel()
            raise MCPError(f"the MCP server did not respond within {timeout:.0f}s") from exc

    def shutdown(self) -> None:
        with self._lock:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None


_thread = _LoopThread()


@dataclass
class _Session:
    session: Any
    tools: list[MCPTool]
    #: Set to ask the owning task to tear the connection down.
    stop: asyncio.Event
    #: The task that owns the connection. Everything the transport opened is
    #: closed inside it, because anyio's task groups — which `stdio_client` and
    #: `streamablehttp_client` both use — may only be exited from the task that
    #: entered them. Closing from a second coroutine leaves the subprocess
    #: running and logs a cancel-scope error, which is exactly what happened
    #: before this was structured as a long-lived owner task.
    task: Any


class MCPClient:
    """Connections to MCP servers, kept alive and reused."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    # ---- public, synchronous API ---------------------------------------

    def list_tools(self, server_id: str, config: MCPServerConfig, credentials: dict) -> list[MCPTool]:
        """What this server can do. Connects if it is not already connected."""
        return self._session_for(server_id, config, credentials).tools

    def call_tool(
        self,
        server_id: str,
        config: MCPServerConfig,
        credentials: dict,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Invoke one tool and return its result as text for the agent."""
        entry = self._session_for(server_id, config, credentials)
        try:
            result = _thread.submit(
                entry.session.call_tool(tool_name, arguments or {}), CALL_TIMEOUT
            )
        except MCPError:
            raise
        except BaseException as exc:  # noqa: BLE001 - the SDK raises a family of errors
            # A dead session should not poison every later call; drop it so the
            # next invocation reconnects rather than failing the same way.
            self.disconnect(server_id)
            raise MCPError(f"calling '{tool_name}' failed: {explain(exc)}") from exc

        if getattr(result, "isError", False):
            raise MCPError(_render(result) or f"'{tool_name}' reported an error")
        return _render(result)

    def probe(self, config: MCPServerConfig, credentials: dict) -> list[MCPTool]:
        """Connect, list tools, disconnect. Used to verify a server before saving."""
        assert_permitted(config)
        try:
            return _thread.submit(_probe(config, credentials), CONNECT_TIMEOUT)
        except (MCPError, MCPPolicyError):
            raise
        except BaseException as exc:  # noqa: BLE001 - includes ExceptionGroup
            raise MCPError(f"could not connect: {explain(exc)}") from exc

    def disconnect(self, server_id: str) -> None:
        with self._lock:
            entry = self._sessions.pop(server_id, None)
        if entry is None:
            return
        try:
            _thread.submit(_shutdown(entry), 15.0)
        except Exception:  # noqa: BLE001 - teardown must not raise into a caller
            logger.warning("MCP session for %s did not close cleanly", server_id)

    def disconnect_all(self) -> None:
        for server_id in list(self._sessions):
            self.disconnect(server_id)
        _thread.shutdown()

    # ---- internals ------------------------------------------------------

    def _session_for(
        self, server_id: str, config: MCPServerConfig, credentials: dict
    ) -> _Session:
        with self._lock:
            existing = self._sessions.get(server_id)
        if existing is not None:
            return existing

        assert_permitted(config)
        try:
            entry = _thread.submit(_connect(config, credentials), CONNECT_TIMEOUT)
        except MCPError:
            raise
        except BaseException as exc:  # noqa: BLE001 - includes ExceptionGroup
            raise MCPError(f"could not connect: {explain(exc)}") from exc

        with self._lock:
            # Another thread may have connected while this one was waiting; keep
            # one session per server rather than leaking the loser.
            winner = self._sessions.setdefault(server_id, entry)
        if winner is not entry:
            with contextlib.suppress(Exception):
                _thread.submit(_shutdown(entry), 15.0)
        return winner


async def _own_connection(
    config: MCPServerConfig,
    credentials: dict,
    ready: asyncio.Future,
    stop: asyncio.Event,
) -> None:
    """Hold one connection open for its whole life, in one task.

    The transport context managers are entered and exited here and nowhere else.
    Everything else — `call_tool`, `list_tools` — is a plain method call on the
    session object and is safe to make from another task.
    """
    from mcp import ClientSession  # noqa: PLC0415

    try:
        async with AsyncExitStack() as stack:
            read, write = await _open_transport(stack, config, credentials)
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listing = await session.list_tools()
            if not ready.done():
                ready.set_result((session, _to_tools(listing)))
            await stop.wait()
    except BaseException as exc:  # noqa: BLE001 - reported to whoever is waiting
        if not ready.done():
            ready.set_exception(exc)
        elif not isinstance(exc, asyncio.CancelledError):
            logger.warning("MCP connection ended: %s", exc)
            raise


async def _connect(config: MCPServerConfig, credentials: dict) -> _Session:
    loop = asyncio.get_running_loop()
    ready: asyncio.Future = loop.create_future()
    stop = asyncio.Event()
    task = loop.create_task(_own_connection(config, credentials, ready, stop))

    try:
        session, tools = await ready
    except BaseException:
        stop.set()
        task.cancel()
        raise

    return _Session(session=session, tools=tools, stop=stop, task=task)


async def _shutdown(entry: _Session) -> None:
    """Ask the owning task to close, and wait for it to finish doing so."""
    entry.stop.set()
    try:
        await asyncio.wait_for(entry.task, timeout=10.0)
    except (TimeoutError, asyncio.CancelledError):
        entry.task.cancel()


async def _probe(config: MCPServerConfig, credentials: dict) -> list[MCPTool]:
    entry = await _connect(config, credentials)
    try:
        return entry.tools
    finally:
        await _shutdown(entry)


async def _open_transport(stack: AsyncExitStack, config: MCPServerConfig, credentials: dict):
    if config.transport is MCPTransport.STDIO:
        from mcp import StdioServerParameters  # noqa: PLC0415
        from mcp.client.stdio import (
            get_default_environment,  # noqa: PLC0415
            stdio_client,  # noqa: PLC0415
        )

        # Credentials for a stdio server are environment values — a token the
        # server itself reads. They are merged here rather than stored in
        # `config.env`, so listing a server never discloses them.
        #
        # On top of the SDK's default environment, not instead of it. Passing a
        # bare dict hands the child an environment with no PATH and no HOME,
        # which breaks the single most common way to launch an MCP server —
        # `npx -y <package>` — in a way that looks like a hung server rather
        # than a missing variable. The default is a deliberate short list (PATH,
        # HOME, SHELL, TERM), not the worker's whole environment, so nothing
        # this process holds leaks into a server a tenant named.
        streams = await stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=config.command,
                    args=list(config.args),
                    env={
                        **get_default_environment(),
                        **config.env,
                        **{key: str(value) for key, value in credentials.items()},
                    },
                )
            )
        )
        return streams[0], streams[1]

    from mcp.client.streamable_http import streamablehttp_client  # noqa: PLC0415

    headers = {str(k): str(v) for k, v in credentials.items()}
    streams = await stack.enter_async_context(
        streamablehttp_client(config.url, headers=headers or None)
    )
    return streams[0], streams[1]


def _to_tools(listing: Any) -> list[MCPTool]:
    tools: list[MCPTool] = []
    for tool in getattr(listing, "tools", []) or []:
        annotations = getattr(tool, "annotations", None)
        tools.append(
            MCPTool(
                name=tool.name,
                description=(getattr(tool, "description", "") or "")[:2000],
                input_schema=getattr(tool, "inputSchema", None) or {},
                read_only=bool(getattr(annotations, "readOnlyHint", False)),
            )
        )
    return tools


def _render(result: Any) -> str:
    """Flatten an MCP tool result into text an agent can read.

    A result is a list of content blocks — text, images, embedded resources. The
    agent loop is text-in/text-out, so non-text blocks are named rather than
    dropped silently: "there was an image here" is more useful to a model than a
    gap it cannot account for.
    """
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(getattr(block, "text", "") or "")
        elif kind == "resource":
            resource = getattr(block, "resource", None)
            text = getattr(resource, "text", None)
            parts.append(text if text else f"[resource: {getattr(resource, 'uri', 'unknown')}]")
        else:
            parts.append(f"[{kind or 'non-text'} content omitted]")

    structured = getattr(result, "structuredContent", None)
    if structured and not parts:
        import json  # noqa: PLC0415

        parts.append(json.dumps(structured, indent=2, default=str))

    return "\n".join(part for part in parts if part).strip()


_client: MCPClient | None = None


def get_client() -> MCPClient:
    global _client
    if _client is None:
        _client = MCPClient()
    return _client


def reset_client(client: MCPClient | None = None) -> None:
    """Swap the client. Used by tests and at shutdown."""
    global _client
    if _client is not None and client is not _client:
        _client.disconnect_all()
    _client = client
