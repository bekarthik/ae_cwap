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
        assert "did not respond within" in message
        assert "could not reach the host" not in message

    def test_it_names_the_setting_that_would_wait_longer(self, monkeypatch, wedged):
        """The complaint that started this: a timeout with no way to raise it."""
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 2.0)

        with pytest.raises(MCPError, match="CWAP_MCP_TIMEOUT"):
            mcp_client.get_client().probe(http(wedged), {})

    def test_a_timed_out_attempt_does_not_stay_running(self, monkeypatch, wedged):
        """`Future.cancel()` is a no-op once a coroutine has started, so every
        failed attempt used to leave its task — and its socket — alive for the
        life of the process."""
        import asyncio

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
