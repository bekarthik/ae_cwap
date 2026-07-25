"""Connecting over HTTP, which is how every hosted MCP server is reached.

Every MCP test the platform had ran over **stdio**. `_open_transport`'s HTTP
branch — the one GitHub, DeepWiki, Context7, Sentry and Stripe all go through —
was never executed by any test, which is why a user reporting that GitHub's
server hung could not be answered from the suite.

Two servers are exercised here:

* a normal one (FastMCP over streamable HTTP), which offers the optional
  server→client GET stream;
* a **POST-only** one, which refuses that GET with 405 — the shape GitHub's
  server has, and the shape that hangs the Python SDK
  (modelcontextprotocol/python-sdk#1941).

The second is the interesting one. A server declining the GET stream is not
broken and is not unusual: the MCP specification makes that stream optional, so
"works against FastMCP" was never evidence that the platform could connect to
anything a user would actually name.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from cwap_contracts.v4 import MCPServerConfig, MCPTransport
from mcp_connect import client as mcp_client
from mcp_fixture_post_only import ANSWER_AS, REQUEST_LOG, STALL_ON, free_port, serve

HTTP_FIXTURE = str(Path(__file__).parent / "mcp_fixture_http_server.py")


@pytest.fixture(autouse=True)
def _no_policy_gate(monkeypatch):
    """These tests are about the transport, not about who may reach what."""
    monkeypatch.setattr(mcp_client, "assert_permitted", lambda config: None)


@pytest.fixture(autouse=True)
def _drop_sessions():
    yield
    mcp_client.reset_client(None)


def wait_for(port: int, seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return
        except OSError:
            time.sleep(0.15)
    raise AssertionError(f"nothing came up on port {port}")


@pytest.fixture
def ordinary_server():
    """FastMCP over streamable HTTP — a server that offers the GET stream."""
    port = free_port()
    process = subprocess.Popen([sys.executable, HTTP_FIXTURE, str(port)])
    try:
        wait_for(port)
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        process.terminate()
        process.wait(timeout=15)


@pytest.fixture
def post_only_server():
    """A server that refuses the GET stream with 405, as GitHub's does."""
    port = free_port()
    server = serve(port)
    try:
        wait_for(port)
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        server.shutdown()
        server.server_close()


def http(url: str) -> MCPServerConfig:
    return MCPServerConfig(transport=MCPTransport.HTTP, url=url)


class TestAnOrdinaryHttpServer:
    def test_connecting_lists_its_tools(self, ordinary_server):
        tools = mcp_client.get_client().probe(http(ordinary_server), {})

        assert {tool.name for tool in tools} == {"read_file", "create_pull_request"}

    def test_a_read_only_tool_is_marked_read_only(self, ordinary_server):
        tools = {tool.name: tool for tool in mcp_client.get_client().probe(http(ordinary_server), {})}

        assert tools["read_file"].read_only
        assert not tools["create_pull_request"].read_only

    def test_a_missing_trailing_slash_still_works(self, ordinary_server):
        """Users paste URLs. The directory stores one form and people type the
        other, and a redirect must not become a failed connection."""
        tools = mcp_client.get_client().probe(http(ordinary_server.rstrip("/")), {})

        assert tools


class TestAPostOnlyServer:
    """The GitHub shape. This is the bug the user hit."""

    def test_it_connects_rather_than_hanging(self, post_only_server):
        started = time.monotonic()

        tools = mcp_client.get_client().probe(http(post_only_server), {})

        assert {tool.name for tool in tools} == {"read_file"}
        # Generous, but far below the handshake budget: the failure mode being
        # guarded against is a wait that only ends when something kills it.
        assert time.monotonic() - started < 15.0

    def test_the_server_did_refuse_the_get_stream(self, post_only_server):
        """Proves the fixture reproduced the right shape, rather than passing
        because the client never attempted the optional stream."""
        mcp_client.get_client().probe(http(post_only_server), {})

        assert "POST initialize" in REQUEST_LOG
        assert "POST tools/list" in REQUEST_LOG

    def test_tools_survive_a_second_call_on_the_same_session(self, post_only_server):
        """`list_tools` is the call the upstream issue reports hanging, and a
        kept-alive session is how this platform uses a connected server."""
        client = mcp_client.get_client()
        config = http(post_only_server)

        first = client.list_tools("srv-post-only", config, {})
        second = client.list_tools("srv-post-only", config, {})

        assert [tool.name for tool in first] == [tool.name for tool in second]


class TestAnEndpointThatIsNotMcp:
    """`200 OK` carrying something that is not the protocol.

    This is the fault that hung, and the reason it hung is worth stating: the
    MCP client library reports an unusable content type by *posting a
    ValueError into the read stream*, and the default message handler drops it.
    The pending request is never answered and never fails — it simply waits.
    From outside, a server that replied instantly with a login page was
    indistinguishable from one that said nothing at all for a minute.
    """

    @pytest.fixture(autouse=True)
    def _short_budget(self, monkeypatch):
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 5.0)

    def test_an_html_page_fails_immediately_instead_of_hanging(self, post_only_server):
        ANSWER_AS.append("text/html")
        started = time.monotonic()

        with pytest.raises(mcp_client.MCPError):
            mcp_client.get_client().probe(http(post_only_server), {})

        # Well under the budget: the point is that it fails on the *answer*,
        # not on a deadline.
        assert time.monotonic() - started < 3.0

    def test_it_says_the_url_is_not_an_mcp_endpoint(self, post_only_server):
        ANSWER_AS.append("text/html")

        with pytest.raises(mcp_client.MCPError, match="not with MCP"):
            mcp_client.get_client().probe(http(post_only_server), {})

    def test_it_suggests_what_answers_like_that(self, post_only_server):
        """A login page and a proxy are the two things that do this, and naming
        them is the difference between a fix and an evening of guessing."""
        ANSWER_AS.append("text/html")

        with pytest.raises(mcp_client.MCPError, match="login page|proxy"):
            mcp_client.get_client().probe(http(post_only_server), {})

    def test_it_keeps_what_the_library_said(self, post_only_server):
        """So the error is still searchable against the library's own issues."""
        ANSWER_AS.append("text/html")

        with pytest.raises(mcp_client.MCPError, match="text/html"):
            mcp_client.get_client().probe(http(post_only_server), {})

    def test_it_is_not_reported_as_silence(self, post_only_server):
        ANSWER_AS.append("text/plain")

        with pytest.raises(mcp_client.MCPError) as raised:
            mcp_client.get_client().probe(http(post_only_server), {})

        assert "stopped responding" not in str(raised.value)


class TestAStallSaysWhichPhaseStalled:
    """The user's report, made answerable.

    "the MCP server did not respond within 60s" was true of two completely
    different situations. Told apart, the first says *your setup is wrong* and
    the second says *your setup is right and the client library has a known
    defect* — and a user can act on either.
    """

    @pytest.fixture(autouse=True)
    def _short_budget(self, monkeypatch):
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 3.0)
        monkeypatch.setattr(mcp_client, "REACH_TIMEOUT", 2.0)

    def test_a_stalled_handshake_names_the_handshake(self, post_only_server):
        STALL_ON.append("initialize")

        with pytest.raises(mcp_client.MCPError) as raised:
            mcp_client.get_client().probe(http(post_only_server), {})

        message = str(raised.value)
        assert "completing the MCP handshake" in message
        assert "listing its tools" not in message

    def test_a_stalled_listing_names_the_listing(self, post_only_server):
        """The shape the upstream issue describes: handshake fine, next request
        never returns."""
        STALL_ON.append("tools/list")

        with pytest.raises(mcp_client.MCPError) as raised:
            mcp_client.get_client().probe(http(post_only_server), {})

        message = str(raised.value)
        assert "listing its tools" in message
        assert "completing the MCP handshake" not in message

    def test_a_stalled_listing_points_at_the_known_defect(self, post_only_server):
        """Because the remedy is "wait longer", not "check your URL" — and a
        user who has just been told their setup is fine will otherwise spend the
        evening re-checking it."""
        STALL_ON.append("tools/list")

        with pytest.raises(mcp_client.MCPError, match="python-sdk#1941"):
            mcp_client.get_client().probe(http(post_only_server), {})

    def test_a_stalled_listing_says_the_credential_was_accepted(self, post_only_server):
        STALL_ON.append("tools/list")

        with pytest.raises(mcp_client.MCPError, match="handshake succeeded"):
            mcp_client.get_client().probe(http(post_only_server), {})

    def test_the_phase_message_beats_the_outer_wall(self, post_only_server):
        """The specific error has to arrive before the blanket one, or the
        blanket one is all anybody ever sees. That is the original bug."""
        STALL_ON.append("tools/list")

        with pytest.raises(mcp_client.MCPError) as raised:
            mcp_client.get_client().probe(http(post_only_server), {})

        assert "did not respond within" not in str(raised.value)

    def test_the_whole_thing_still_gives_up_within_the_budget(self, post_only_server):
        """Two phases share one budget rather than getting one each, so
        splitting them did not double how long a user waits."""
        STALL_ON.append("tools/list")
        started = time.monotonic()

        with pytest.raises(mcp_client.MCPError):
            mcp_client.get_client().probe(http(post_only_server), {})

        assert time.monotonic() - started < mcp_client.CONNECT_TIMEOUT + mcp_client.GRACE
