"""Telling "I cannot reach that host" apart from "that server is slow".

A user connecting GitHub's MCP server got `the MCP server did not respond within
30s` and had nothing to do next. The message was the same sentence a genuinely
slow server produces, it arrived after the same thirty seconds, and thirty
seconds was a constant in the source — so someone whose only problem was a slow
link could not wait longer even if they knew that was the problem.

Reproducing it showed the fault was structural rather than a bad number. Three
different faults were collapsing into one message:

| What is wrong | What the user should do |
| --- | --- |
| The host is not reachable at all | Fix the URL, or the deployment's egress |
| The server accepted the connection and went quiet | Wait longer, or the server is broken |
| Nothing is listening on that port | Fix the URL |

The transport can tell these apart — httpx raises `ConnectTimeout` for the
first and `ReadTimeout` for the second — and the platform was throwing that
information away by wrapping the whole handshake in a single blanket deadline
set to exactly the SDK's own. Whichever fired first, the user got the vaguer of
the two errors.

These tests hold the distinction in place. They use real sockets, because the
whole bug lived in the difference between kinds of network silence and a mock
would have to be told which kind to imitate.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest
from cwap_contracts.v4 import MCPServerConfig, MCPTransport
from mcp_connect import client as mcp_client
from mcp_connect.client import MCPError

#: RFC 5737 documentation space — an address nothing routes to. Whether a given
#: network drops the SYN or answers with an ICMP unreachable is the network's
#: business, and these tests deliberately assert only on what must hold either
#: way: it is reported as a failure to connect, never as a slow server. The
#: distinction between the two *messages* is asserted on the exception types
#: directly, in `TestTheExplanationIsUseful`, where no network is involved.
BLACK_HOLE = "http://192.0.2.1:9/mcp"


@pytest.fixture(autouse=True)
def _no_policy_gate(monkeypatch):
    """These tests are about the transport, not about who may reach what."""
    monkeypatch.setattr(mcp_client, "assert_permitted", lambda config: None)


@pytest.fixture(autouse=True)
def _drop_sessions():
    yield
    mcp_client.reset_client(None)


@pytest.fixture
def wedged() -> str:
    """A server that completes the TCP handshake and then says nothing.

    The failure mode a blanket timeout cannot distinguish from an unreachable
    host, and the one that actually deserves "raise the timeout".
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    held: list[socket.socket] = []

    def accept_forever() -> None:
        while True:
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            held.append(connection)  # never read from, never replied to

    threading.Thread(target=accept_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/mcp"
    finally:
        for connection in held:
            connection.close()
        listener.close()


def http(url: str) -> MCPServerConfig:
    return MCPServerConfig(transport=MCPTransport.HTTP, url=url)


class TestAnUnreachableHostSaysSo:
    def test_it_is_not_reported_as_a_slow_server(self, monkeypatch):
        """The bug, stated as a property: an address that cannot be reached must
        never produce the sentence a working-but-slow server produces."""
        monkeypatch.setattr(mcp_client, "REACH_TIMEOUT", 2.0)

        with pytest.raises(MCPError) as raised:
            mcp_client.get_client().probe(http(BLACK_HOLE), {})

        message = str(raised.value)
        assert "could not connect to" in message
        assert "did not respond within" not in message

    def test_it_names_the_address_that_failed(self, monkeypatch):
        monkeypatch.setattr(mcp_client, "REACH_TIMEOUT", 2.0)

        with pytest.raises(MCPError, match="192.0.2.1"):
            mcp_client.get_client().probe(http(BLACK_HOLE), {})

    def test_it_gives_up_on_reaching_the_host_long_before_the_whole_budget(
        self, monkeypatch
    ):
        """The point of a separate reach deadline: waiting the full handshake
        budget for a host that will never answer a SYN helps nobody."""
        monkeypatch.setattr(mcp_client, "REACH_TIMEOUT", 2.0)
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 30.0)

        started = time.monotonic()
        with pytest.raises(MCPError):
            mcp_client.get_client().probe(http(BLACK_HOLE), {})

        assert time.monotonic() - started < 10.0

    def test_nothing_listening_is_immediate(self):
        started = time.monotonic()
        with pytest.raises(MCPError, match="could not connect"):
            mcp_client.get_client().probe(http("http://127.0.0.1:1/mcp"), {})

        assert time.monotonic() - started < 5.0


class TestAQuietServerSaysSomethingElse:
    def test_it_reports_silence_rather_than_unreachability(self, monkeypatch, wedged):
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 2.0)

        with pytest.raises(MCPError) as raised:
            mcp_client.get_client().probe(http(wedged), {})

        message = str(raised.value)
        assert "stopped responding" in message
        assert "could not reach the host" not in message

    def test_it_names_the_setting_that_would_wait_longer(self, monkeypatch, wedged):
        """The complaint that started this: a timeout with no way to raise it."""
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 2.0)

        with pytest.raises(MCPError, match="CWAP_MCP_TIMEOUT"):
            mcp_client.get_client().probe(http(wedged), {})

    def test_the_url_is_named_once(self, monkeypatch, wedged):
        """A phase error is raised inside the transport's task group, so it
        comes back boxed. Re-wrapping it named the URL twice and buried the
        sentence that mattered underneath a second, vaguer one."""
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 2.0)

        with pytest.raises(MCPError) as raised:
            mcp_client.get_client().probe(http(wedged), {})

        assert str(raised.value).count(wedged) == 1

    def test_a_timed_out_attempt_does_not_stay_running(self, monkeypatch, wedged):
        """`Future.cancel()` is a no-op once a coroutine has started, so every
        failed attempt used to leave its task — and its socket — alive for the
        life of the process."""
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 2.0)

        with pytest.raises(MCPError):
            mcp_client.get_client().probe(http(wedged), {})

        loop = mcp_client._thread.loop()
        # Give the cancellation a moment to unwind on the loop thread.
        for _ in range(40):
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if not pending:
                break
            time.sleep(0.05)

        assert not [task for task in asyncio.all_tasks(loop) if not task.done()]


class TestTheTransportGetsBothDeadlines:
    """The fix itself: two numbers where there was one blanket wrapper.

    Asserted on the call rather than on a network, because "does httpx raise
    ConnectTimeout or ConnectError here" is a property of whoever is running the
    tests, and this is a property of the platform.
    """

    @pytest.fixture
    def recorded(self, monkeypatch) -> dict:
        seen: dict = {}
        import mcp.client.streamable_http as transport

        original = transport.streamablehttp_client

        def spy(url, **kwargs):
            seen.update(kwargs, url=url)
            return original(url, **kwargs)

        monkeypatch.setattr(transport, "streamablehttp_client", spy)
        with pytest.raises(MCPError):
            mcp_client.get_client().probe(http("http://127.0.0.1:1/mcp"), {})
        return seen

    def test_reaching_the_host_is_bounded_separately(self, recorded):
        assert recorded["timeout"] == mcp_client.REACH_TIMEOUT

    def test_waiting_for_the_server_covers_the_slowest_call(self, recorded):
        """Sizing the read deadline to the handshake would silently cut off a
        long-running tool call later in the same session."""
        assert recorded["sse_read_timeout"] >= mcp_client.CALL_TIMEOUT
        assert recorded["sse_read_timeout"] >= mcp_client.CONNECT_TIMEOUT


class TestTheDiagnosisArrivesBeforeTheTeardown:
    """A slow close must not swallow the reason.

    Reported from a Docker deployment: `did not respond within 70s` — the outer
    wall (`CONNECT_TIMEOUT + GRACE`), carrying the generic sentence, when the
    phase deadline at 60s should have fired first with a specific one.

    The cause was ordering. `ready.set_exception` sat in the handler *outside*
    the `async with`, so it ran only once the transport had finished closing —
    and closing waits on the SDK's task group, whose children may be parked in a
    read with the session's own read timeout. The phase error was raised on
    time, then queued behind a teardown that outlasted the caller's patience.

    This is asserted with a transport whose close blocks on an event the test
    controls, because "teardown is slow" is not something a real server can be
    asked to do on demand.
    """

    @pytest.fixture
    def slow_to_close(self, monkeypatch):
        """A transport that hands over dead streams and then refuses to close."""
        import anyio

        released = anyio.Event()

        async def _open(stack, config, credentials):
            send_to_reader, read_stream = anyio.create_memory_object_stream(10)
            write_stream, _ = anyio.create_memory_object_stream(10)

            class Blocks:
                async def __aenter__(self):
                    return None

                async def __aexit__(self, *exc):
                    await released.wait()
                    return False

            await stack.enter_async_context(Blocks())
            return read_stream, write_stream

        monkeypatch.setattr(mcp_client, "_open_transport", _open)
        yield released
        released.set()

    def test_the_phase_message_beats_a_wedged_close(self, monkeypatch, slow_to_close):
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 2.0)

        with pytest.raises(MCPError) as raised:
            mcp_client.get_client().probe(http("http://127.0.0.1:9/mcp"), {})

        assert "stopped responding while" in str(raised.value)
        assert "did not respond within" not in str(raised.value)

    def test_the_caller_is_answered_within_the_budget(self, monkeypatch, slow_to_close):
        """Not merely the right message — the right message *on time*."""
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 2.0)
        started = time.monotonic()

        with pytest.raises(MCPError):
            mcp_client.get_client().probe(http("http://127.0.0.1:9/mcp"), {})

        assert time.monotonic() - started < mcp_client.CONNECT_TIMEOUT + mcp_client.GRACE


class TestATimeoutFiresOnSomethingThatIgnoresCancellation:
    """The deadline cannot depend on the stuck thing agreeing to stop.

    `asyncio.wait_for` cancels the inner task and then *awaits the
    cancellation*, so it honours its own deadline only when the task can be
    cancelled promptly. A request wedged inside the SDK's anyio task group
    cannot be: the group is suspended at the generator yield we are still
    inside, so the cancellation never completes and `wait_for` never returns.

    That is why a 60s phase deadline produced no error at all and the 70s outer
    wall reported the generic one — twice, to a user who had already been told
    the specific message was coming.
    """

    @pytest.fixture
    def uncancellable(self, monkeypatch):
        """A handshake that hangs and refuses to be cancelled, as a wedged
        transport does."""
        import anyio

        async def _open(stack, config, credentials):
            send, read_stream = anyio.create_memory_object_stream(10)
            write_stream, _ = anyio.create_memory_object_stream(10)
            return read_stream, write_stream

        class Immovable:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def initialize(self):
                while True:
                    try:
                        await asyncio.sleep(3600)
                    except asyncio.CancelledError:
                        # Swallowing cancellation is the whole point: this is
                        # what a task parked inside a wedged cancel scope looks
                        # like from outside.
                        continue

            async def list_tools(self):  # pragma: no cover - never reached
                return None

        monkeypatch.setattr(mcp_client, "_open_transport", _open)
        monkeypatch.setattr(
            mcp_client, "_preflight", lambda config, creds, budget: _noop()
        )
        yield Immovable

    def test_the_phase_deadline_still_fires(self, monkeypatch, uncancellable):
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 2.0)
        monkeypatch.setattr("mcp.ClientSession", lambda *a, **k: uncancellable())

        started = time.monotonic()
        with pytest.raises(MCPError) as raised:
            mcp_client.get_client().probe(http("http://127.0.0.1:9/mcp"), {})

        assert "stopped responding while" in str(raised.value)
        assert "did not respond within" not in str(raised.value)
        assert time.monotonic() - started < mcp_client.CONNECT_TIMEOUT + mcp_client.GRACE


async def _noop():
    return None


class TestARefusalIsReportedAsARefusal:
    """A server that says no in milliseconds must not read as one that went quiet.

    GitHub's remote MCP server answers some clients with `400 Bad Request`. The
    SDK calls `raise_for_status()` inside a task started with `tg.start_soon`,
    so that error lands in a background task while `initialize()` waits on a
    memory stream nothing will ever write to. The request is dead and nothing
    says so, which is how an immediate refusal became "did not respond within
    70s".
    """

    @pytest.fixture
    def refusing(self):
        """A server that rejects the handshake the way a real one does."""
        import json

        from mcp_fixture_post_only import free_port

        port = free_port()
        listener = socket.socket()
        listener.bind(("127.0.0.1", port))
        listener.listen(8)

        def serve() -> None:
            while True:
                try:
                    connection, _ = listener.accept()
                except OSError:
                    return
                connection.recv(65535)
                body = json.dumps({"error": "protocol version not supported"}).encode()
                connection.sendall(
                    b"HTTP/1.1 400 Bad Request\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
                connection.close()

        threading.Thread(target=serve, daemon=True).start()
        try:
            yield f"http://127.0.0.1:{port}/mcp/"
        finally:
            listener.close()

    def test_it_fails_immediately(self, monkeypatch, refusing):
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 30.0)
        started = time.monotonic()

        with pytest.raises(MCPError):
            mcp_client.get_client().probe(http(refusing), {})

        assert time.monotonic() - started < 5.0

    def test_it_reports_the_status(self, refusing):
        with pytest.raises(MCPError, match="HTTP 400"):
            mcp_client.get_client().probe(http(refusing), {})

    def test_it_quotes_what_the_server_said(self, refusing):
        """The server's own words are the actionable part — ours are a guess."""
        with pytest.raises(MCPError, match="protocol version not supported"):
            mcp_client.get_client().probe(http(refusing), {})

    def test_it_is_not_described_as_silence(self, refusing):
        with pytest.raises(MCPError) as raised:
            mcp_client.get_client().probe(http(refusing), {})

        assert "did not respond" not in str(raised.value)
        assert "stopped responding" not in str(raised.value)


class TestAKeptAliveStreamCannotTrapThePreflight:
    """The pre-flight must judge on headers, never on a body that may not end.

    Its first version called `probe.post()`, which buffers the whole body
    before returning — and a streamable-HTTP server is entitled to hold the
    POST's event stream open with keepalive pings. Each ping resets httpx's
    read timeout, so the pre-flight sat inside a healthy, chatty connection
    forever, and the refusal check it exists to perform never ran. GitHub's
    remote server answers exactly like this, which is how the check built to
    end the 70-second mystery became its next cause.
    """

    @pytest.fixture
    def dripping(self):
        """200, correct content type, an event, then pings forever."""
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)

        def serve() -> None:
            while True:
                try:
                    connection, _ = listener.accept()
                except OSError:
                    return
                connection.recv(65535)
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    b"Cache-Control: no-cache\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                )
                try:
                    frame = b"event: message\ndata: {}\n\n"
                    connection.sendall(b"%X\r\n" % len(frame) + frame + b"\r\n")
                    while True:
                        ping = b": ping\n\n"
                        connection.sendall(b"%X\r\n" % len(ping) + ping + b"\r\n")
                        time.sleep(0.2)
                except OSError:
                    connection.close()

        threading.Thread(target=serve, daemon=True).start()
        try:
            yield f"http://127.0.0.1:{listener.getsockname()[1]}/mcp/"
        finally:
            listener.close()

    def test_the_preflight_returns_promptly(self, dripping):
        started = time.monotonic()

        asyncio.run(mcp_client._preflight(http(dripping), {}, 60.0))

        # Headers arrive immediately; nothing here may wait for a stream that
        # is designed never to end.
        assert time.monotonic() - started < 5.0

    def test_a_refusals_body_is_still_read_but_bounded(self):
        """Reading the quote on a 4xx must not reintroduce the same trap."""

        class NeverEnds:
            async def aread(self):
                await asyncio.Event().wait()

        said = asyncio.run(mcp_client._said(NeverEnds()))

        assert said == "(a body that never finished arriving)"


class TestTheWatchdogNamesTheStage:
    """Whatever blocks next fails with its stage named, not with the backstop.

    Five rounds of this bug followed one pattern: the step everyone knew could
    not block was the one that blocked, and the error came from a generic wall
    that said nothing. The connection now narrates where it is, and a single
    watchdog covers everything before `ready` resolves — including code that
    does not exist yet.
    """

    def test_an_unbounded_stage_is_named(self, monkeypatch):
        async def wedged_preflight(config, credentials, budget):
            await asyncio.Event().wait()

        monkeypatch.setattr(mcp_client, "_preflight", wedged_preflight)
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 2.0)

        with pytest.raises(MCPError) as raised:
            mcp_client.get_client().probe(http("http://127.0.0.1:9/mcp"), {})

        message = str(raised.value)
        assert "sending the pre-flight request" in message
        assert "gave up after" not in message

    def test_it_fires_before_the_backstop(self, monkeypatch):
        async def wedged_preflight(config, credentials, budget):
            await asyncio.Event().wait()

        monkeypatch.setattr(mcp_client, "_preflight", wedged_preflight)
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 2.0)
        started = time.monotonic()

        with pytest.raises(MCPError):
            mcp_client.get_client().probe(http("http://127.0.0.1:9/mcp"), {})

        assert time.monotonic() - started < mcp_client.CONNECT_TIMEOUT + mcp_client.GRACE


class TestTheBackstopIdentifiesItself:
    """The last-resort message must not read like any other build's message.

    Three rounds of a Docker deployment reporting the *same* generic timeout
    burned most of a day on one question: is the container even running the
    new code? It was not — but nothing on screen could show that, because the
    old build's only message and the new build's backstop were the same
    sentence. They are different sentences now, so the message text alone
    settles which build produced it.
    """

    def test_it_does_not_use_the_old_builds_words(self):
        with pytest.raises(MCPError) as raised:
            mcp_client._thread.submit(asyncio.sleep(30), 0.2)

        assert "did not respond within" not in str(raised.value)

    def test_it_asks_to_be_reported(self):
        """Firing at all means a case the specific checks do not cover."""
        with pytest.raises(MCPError, match="report this message"):
            mcp_client._thread.submit(asyncio.sleep(30), 0.2)


class TestThePlatformsLogsActuallyComeOut:
    """`cwap.*` loggers must have a handler in the shipped containers.

    The MCP connector wrote down what it tried, what answered, and how long
    each step took — into a logger with no handler, while the operator debugged
    against uvicorn's access lines. Everything below WARNING was dropped in
    exactly the topology the compose file ships.
    """

    def test_configuring_gives_the_namespace_a_handler(self):
        import logging

        from cwap_common.diagnostics import NAMESPACE, configure_logging

        logger = logging.getLogger(NAMESPACE)
        before = list(logger.handlers)
        try:
            logger.handlers.clear()
            configure_logging()

            assert logger.handlers, "the cwap namespace still has no handler"
        finally:
            logger.handlers[:] = before

    def test_configuring_twice_does_not_double_every_line(self):
        import logging

        from cwap_common.diagnostics import NAMESPACE, configure_logging

        logger = logging.getLogger(NAMESPACE)
        before = list(logger.handlers)
        try:
            logger.handlers.clear()
            configure_logging()
            configure_logging()

            assert len(logger.handlers) == 1
        finally:
            logger.handlers[:] = before

    def test_the_namespace_does_not_also_propagate(self):
        """Its handler is the whole story — propagating too would print every
        line twice in any process that also configures the root logger."""
        import logging

        from cwap_common.diagnostics import NAMESPACE, configure_logging

        logger = logging.getLogger(NAMESPACE)
        before = list(logger.handlers)
        try:
            logger.handlers.clear()
            configure_logging()

            assert logger.propagate is False
        finally:
            logger.handlers[:] = before


class TestTheDeadlinesAreOrdered:
    def test_reaching_is_bounded_more_tightly_than_answering(self):
        """So the transport's specific error arrives before the blanket one.

        This ordering *is* the fix. When the outer deadline was 30s and the
        SDK's own was also 30s, the two fired together and the vaguer message
        won.
        """
        assert mcp_client.REACH_TIMEOUT < mcp_client.CONNECT_TIMEOUT

    def test_the_outer_wall_is_not_flush_with_the_inner_one(self):
        """A zero grace period recreates the original bug: the wall would pre-empt
        the cancellation it is supposed to be a backstop for."""
        assert mcp_client.GRACE > 0


class TestTheTimeoutsAreConfigurable:
    def test_a_deployment_can_wait_longer(self, monkeypatch):
        monkeypatch.setenv("CWAP_MCP_TIMEOUT", "300")

        assert mcp_client._seconds("CWAP_MCP_TIMEOUT", 60.0) == 300.0

    def test_an_unset_value_keeps_the_default(self, monkeypatch):
        monkeypatch.delenv("CWAP_MCP_TIMEOUT", raising=False)

        assert mcp_client._seconds("CWAP_MCP_TIMEOUT", 60.0) == 60.0

    @pytest.mark.parametrize("bad", ["", "soon", "-5", "0"])
    def test_nonsense_falls_back_rather_than_failing_to_boot(self, monkeypatch, bad):
        """A typo in an environment variable must not take the platform down."""
        monkeypatch.setenv("CWAP_MCP_TIMEOUT", bad)

        assert mcp_client._seconds("CWAP_MCP_TIMEOUT", 60.0) == 60.0


class TestTheExplanationIsUseful:
    def test_an_error_with_no_message_is_still_actionable(self):
        import httpx

        assert "could not reach the host" in mcp_client.explain(httpx.ConnectTimeout(""))

    def test_an_error_that_speaks_for_itself_is_left_alone(self):
        """"Name or service not known" is the most useful sentence a mistyped
        hostname can produce; replacing it would be a downgrade."""
        import httpx

        explained = mcp_client.explain(httpx.ConnectError("Name or service not known"))

        assert explained == "Name or service not known"

    def test_a_task_group_is_flattened_to_its_causes(self):
        """What the user must never be shown: "unhandled errors in a TaskGroup".

        Built by hand rather than with `BaseExceptionGroup`, which is a 3.11
        builtin and this project supports 3.10. `explain` walks `.exceptions`
        duck-typed, so a stand-in exercises the same path — and tests the
        flattening rather than CPython's group implementation.
        """

        class Group(Exception):
            def __init__(self, causes):
                super().__init__("unhandled errors in a TaskGroup (1 sub-exception)")
                self.exceptions = causes

        assert mcp_client.explain(Group([ValueError("a 401")])) == "a 401"

    def test_nesting_is_flattened_too(self):
        """The transport supervises tasks in nested groups; the nesting is an
        implementation detail and not something to show anyone."""

        class Group(Exception):
            def __init__(self, causes):
                super().__init__("unhandled errors in a TaskGroup")
                self.exceptions = causes

        nested = Group([Group([ConnectionRefusedError("[Errno 111] refused")])])

        assert mcp_client.explain(nested) == "[Errno 111] refused"
