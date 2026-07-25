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
import os
import threading
from concurrent.futures import Future
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from cwap_contracts.v4 import MCPServerConfig, MCPTool, MCPTransport

from mcp_connect.policy import MCPPolicyError, assert_permitted

logger = logging.getLogger("cwap.mcp")


def _seconds(name: str, default: float) -> float:
    """A timeout from the environment, ignoring anything unusable."""
    try:
        value = float(os.getenv(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


#: How long a handshake may take. Sixty seconds rather than thirty because the
#: thirty was chosen against a server on localhost: a hosted server behind a
#: TLS-inspecting corporate proxy, or one that cold-starts, routinely needs more
#: — and there was no way to ask for more, since this was a constant.
CONNECT_TIMEOUT = _seconds("CWAP_MCP_TIMEOUT", 60.0)

#: How long one tool call may take. A repository search over a large org is not
#: a hung server.
CALL_TIMEOUT = _seconds("CWAP_MCP_CALL_TIMEOUT", 120.0)

#: How long to wait to *reach* the host, as opposed to waiting for the server to
#: answer. Deliberately short and deliberately separate: those are two different
#: faults with two different remedies, and merging them is what made an
#: unreachable host and a wedged server produce the same sentence after the same
#: thirty seconds. A TCP handshake that has not completed in ten seconds is not
#: going to.
REACH_TIMEOUT = _seconds("CWAP_MCP_CONNECT_TIMEOUT", 10.0)

#: Added to an inner deadline to get the outer one. Every wall in this module
#: must fire *after* the thing it is backstopping, or it pre-empts a specific,
#: actionable error with a generic one — which is precisely what a blanket
#: timeout equal to the SDK's own default did.
GRACE = 10.0


def connect_wall() -> float:
    """The outer deadline for opening a connection.

    Above `CONNECT_TIMEOUT`, which is what the two phases inside share, so a
    phase that stalls gets to say which phase it was before this fires.
    """
    return CONNECT_TIMEOUT + GRACE


class MCPError(RuntimeError):
    """A server could not be reached, or a tool call failed."""


#: httpx raises its timeout errors with an *empty* `str()`, so the generic
#: fallback renders them as a bare class name — "ConnectTimeout" — which is a
#: fact about our stack rather than an answer to "what do I do now".
#:
#: Only the types whose own message is empty are listed. An exception that says
#: something for itself keeps saying it: "Name or service not known" is the most
#: useful sentence a mistyped hostname can produce, and a table entry that
#: replaced it would be a downgrade dressed as an improvement.
_SILENT_FAULTS = {
    "ConnectTimeout": (
        "could not reach the host — the connection timed out. Check the URL, and "
        "whether this deployment is allowed to make outbound connections"
    ),
    "ReadTimeout": "the host accepted the connection and then sent nothing back",
    "WriteTimeout": "the request could not be sent in time",
    "PoolTimeout": "no connection slot was free in time",
}


#: Faults the transport describes in its own vocabulary, and what they mean to
#: whoever typed the URL. Matched as substrings of the SDK's own message, and
#: the SDK's wording is kept inside the explanation rather than replaced — a
#: user pasting an error into a search engine should still find the library.
_INTERPRETATIONS: tuple[tuple[str, str], ...] = (
    (
        "unexpected content type",
        "that URL answered, but not with MCP ({detail}). An MCP endpoint replies "
        "as JSON or as an event stream; HTML usually means a login page, a proxy "
        "sitting in front of the host, or simply a URL that is not an MCP endpoint",
    ),
    (
        "session terminated",
        "the server rejected the session for that URL. Check the path — this is "
        "what a 404 looks like once the protocol has started",
    ),
)


def _interpret(error: BaseException) -> str:
    """Say what one fault means, keeping whatever the transport said about it.

    The single place a raw exception becomes a sentence, whether it arrived by
    being raised or by being posted into the read stream. Those are two routes
    for the same faults, and giving them two vocabularies would mean a 404
    reading one way during the handshake and another way afterwards.
    """
    name = type(error).__name__
    detail = str(error).strip() or _SILENT_FAULTS.get(name) or name
    lowered = detail.lower()
    for marker, template in _INTERPRETATIONS:
        if marker in lowered:
            return template.format(detail=detail)
    return detail


def already_explained(exc: BaseException) -> MCPError | None:
    """An `MCPError` this module raised, dug back out of whatever boxed it.

    A phase timeout is raised inside the transport's anyio task group, so what
    reaches the caller is an ExceptionGroup rather than the error. Wrapping that
    in "could not connect to <url>: ..." produced a message naming the URL
    twice and burying the sentence that mattered. If we already said something
    useful, that is the message.
    """
    if isinstance(exc, MCPError):
        return exc
    for inner in getattr(exc, "exceptions", None) or ():
        found = already_explained(inner)
        if found is not None:
            return found
    return None


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
        text = _interpret(error)
        if text not in leaves:
            leaves.append(text)

    walk(exc)
    # Three is enough to see a pattern without pasting a stack into a dialog.
    return "; ".join(leaves[:3]) or f"{type(exc).__name__}"


def _failure(exc: BaseException, config: MCPServerConfig) -> MCPError:
    """The error to hand back after a connection attempt has come apart."""
    return already_explained(exc) or MCPError(
        f"could not connect to {_target(config)}: {explain(exc)}"
    )


def _target(config: MCPServerConfig) -> str:
    """What was being reached, for an error a person has to act on.

    Somebody debugging a failed connection is usually looking at a form with
    several fields in it; naming the one that was actually used costs nothing
    and saves a round of guessing.
    """
    if config.transport is MCPTransport.STDIO:
        return config.command or "the configured command"
    return config.url or "the configured URL"


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
        # The deadline is enforced *inside* the loop, with `wait_for`, rather
        # than by abandoning the future out here. `Future.cancel()` on a
        # coroutine that has already started is a no-op, so every timed-out
        # attempt used to leave its task running for the life of the process —
        # holding a socket, or a subprocess, that nobody would ever close.
        future: Future = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(coroutine, timeout), self.loop()
        )
        try:
            # Above the inner deadline: this one only fires if the *cancellation*
            # wedges, which is a different fault and worth not hanging on.
            return future.result(timeout=timeout + GRACE)
        except TimeoutError as exc:
            future.cancel()
            raise MCPError(
                f"the MCP server did not respond within {timeout:.0f}s. "
                "It accepted the connection and then went quiet; raise "
                "CWAP_MCP_TIMEOUT if the server is simply slow."
            ) from exc

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
            return _thread.submit(_probe(config, credentials), connect_wall())
        except (MCPError, MCPPolicyError):
            raise
        except BaseException as exc:  # noqa: BLE001 - includes ExceptionGroup
            raise _failure(exc, config) from exc

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
            entry = _thread.submit(_connect(config, credentials), connect_wall())
        except MCPError:
            raise
        except BaseException as exc:  # noqa: BLE001 - includes ExceptionGroup
            raise _failure(exc, config) from exc

        with self._lock:
            # Another thread may have connected while this one was waiting; keep
            # one session per server rather than leaking the loser.
            winner = self._sessions.setdefault(server_id, entry)
        if winner is not entry:
            with contextlib.suppress(Exception):
                _thread.submit(_shutdown(entry), 15.0)
        return winner


#: The two phases of opening a connection, and what a stall in each one means.
#:
#: Splitting them is not tidiness. A server that never completes the handshake
#: and a server that completes it and then stalls listing its tools are
#: different problems with different causes, and reporting both as "the MCP
#: server did not respond" told a user neither which one they had nor that the
#: second is a known defect in the client library rather than in their setup.
HANDSHAKE = (
    "completing the MCP handshake",
    "The host answered, so the URL and the network are fine. A handshake that "
    "then stalls usually means the endpoint is not an MCP server, or a proxy is "
    "sitting between this deployment and it. If the server is merely slow, "
    "raise CWAP_MCP_TIMEOUT.",
)
LISTING = (
    "listing its tools",
    "The handshake succeeded, so the URL and the credential are both good. "
    "Servers that decline the optional server-to-client stream — GitHub's is "
    "one — can stall the MCP client library on the request after the handshake "
    "(modelcontextprotocol/python-sdk#1941). Raising CWAP_MCP_TIMEOUT gives it "
    "longer to get through.",
)


async def _phase(
    phase: tuple[str, str],
    awaitable: Any,
    budget: float,
    config: MCPServerConfig,
) -> Any:
    """Run one phase of the connection under its own share of the budget."""
    what, why = phase
    try:
        return await asyncio.wait_for(awaitable, max(budget, 1.0))
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise MCPError(
            f"{_target(config)} stopped responding while {what}. {why}"
        ) from exc


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

    async def transport_fault(message: Any) -> None:
        """Fail the connection when the transport reports a fault as a message.

        The SDK does not raise for everything that goes wrong. A response with
        an unusable content type — an HTML login page, a proxy's error page,
        a URL that is not an MCP endpoint at all — is reported by *sending a
        ValueError into the read stream*, which the default message handler
        drops on the floor. The `initialize()` call is still waiting for its
        response, and no response is ever coming, so the connection hangs until
        something outside kills it.

        That is what "the MCP server did not respond" was hiding: a server that
        answered instantly with the wrong thing, described as one that said
        nothing at all. The information was there and nobody was listening.
        """
        if not isinstance(message, Exception):
            return

        detail = _interpret(message)
        logger.warning("MCP transport fault from %s: %s", _target(config), detail)
        if not ready.done():
            ready.set_exception(MCPError(detail))
        # Mid-session rather than during the handshake: the connection is no
        # longer trustworthy, so drop it and let the next call reconnect.
        stop.set()

    try:
        async with AsyncExitStack() as stack:
            read, write = await _open_transport(stack, config, credentials)
            session = await stack.enter_async_context(
                ClientSession(read, write, message_handler=transport_fault)
            )

            # One budget for the whole connection, spent across two phases that
            # are timed separately. Which of the two stalled is the single most
            # useful thing an error can say here — see `_phase`.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + CONNECT_TIMEOUT

            await _phase(HANDSHAKE, session.initialize(), deadline - loop.time(), config)
            listing = await _phase(
                LISTING, session.list_tools(), deadline - loop.time(), config
            )

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
        streamablehttp_client(
            config.url,
            headers=headers or None,
            # `timeout` is the connect/write deadline and `sse_read_timeout` is
            # how long to wait for the server to say something. Splitting them is
            # the whole point: a host that never completes a TCP handshake fails
            # in REACH_TIMEOUT with an error that names the host, instead of
            # looking identical to a server that is merely thinking.
            timeout=REACH_TIMEOUT,
            # One number for the life of the session, so it has to cover the
            # slowest thing the session will do — a tool call, not the handshake.
            sse_read_timeout=max(CONNECT_TIMEOUT, CALL_TIMEOUT),
        )
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
