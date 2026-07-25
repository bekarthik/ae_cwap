"""Connecting a system without an administrator being asked first.

The host allow-list is empty on a fresh deployment, so the first thing anyone
saw when they tried to connect GitHub was *set CWAP_HTTP_ALLOWLIST* — an
environment variable, on a machine they may not administer, to reach a service
they already trust. A gate that stops the intended use as reliably as the
unintended one has stopped protecting anything.

So a small reviewed list of published endpoints is reachable without the
allow-list, and everything else is exactly as gated as before. These tests hold
both halves of that: the listed server connects, and the unlisted one does not.
"""

from __future__ import annotations

import contextlib

import pytest
from cwap_common.settings import reset_settings_cache
from cwap_contracts.v4 import MCPServerConfig, MCPTransport
from mcp_connect import directory, policy, registry

# ExceptionGroup is a builtin from 3.11. On 3.10 anyio raises the backport's
# class instead, and this suite has to construct the real one to be testing
# anything.
with contextlib.suppress(ImportError):  # pragma: no cover - 3.11 and later
    from exceptiongroup import ExceptionGroup  # type: ignore[import-not-found]


@pytest.fixture
def settings_reset():
    """The directory reads settings, which are cached for the process."""
    reset_settings_cache()
    yield
    reset_settings_cache()


class TestTheListedServersAreReachable:
    def test_a_listed_host_needs_no_allowlist(self, settings_reset):
        """The whole point: GitHub connects without an environment variable."""
        github = directory.find("github")
        assert github is not None
        policy.assert_permitted(
            MCPServerConfig(transport=MCPTransport.HTTP, url=github.url)
        )

    def test_every_listed_http_entry_passes_its_own_policy(self, settings_reset):
        """A directory entry that the platform then refuses is worse than absent."""
        for entry in directory.DIRECTORY:
            if entry.transport is not MCPTransport.HTTP:
                continue
            policy.assert_permitted(
                MCPServerConfig(transport=MCPTransport.HTTP, url=entry.url)
            )

    def test_an_unlisted_host_is_still_refused(self, settings_reset):
        """The exemption is a fixed list, not a hole."""
        result = registry.test(
            MCPServerConfig(transport=MCPTransport.HTTP, url="https://evil.example.com/mcp")
        )

        assert result.ok is False
        assert result.blocked is True

    def test_the_refusal_says_what_to_do_instead(self, settings_reset):
        """Naming the two ways forward beats naming an environment variable."""
        result = registry.test(
            MCPServerConfig(transport=MCPTransport.HTTP, url="https://other.example.com/mcp")
        )

        assert "list" in result.message
        assert "CWAP_HTTP_ALLOWLIST" in result.message

    def test_a_subdomain_of_a_listed_host_is_not_covered(self, settings_reset):
        """`mcp.deepwiki.com` being listed must not admit `evil.deepwiki.com`."""
        assert directory.vouches_for("mcp.deepwiki.com")
        assert not directory.vouches_for("evil.deepwiki.com")

    def test_an_operator_can_switch_the_whole_thing_off(self, monkeypatch):
        monkeypatch.setenv("CWAP_MCP_DIRECTORY", "false")
        reset_settings_cache()
        try:
            assert directory.hosts() == frozenset()
            result = registry.test(
                MCPServerConfig(
                    transport=MCPTransport.HTTP, url=directory.find("github").url
                )
            )
            assert result.blocked is True
        finally:
            monkeypatch.delenv("CWAP_MCP_DIRECTORY")
            reset_settings_cache()


class TestTheListIsHonestAboutWhatItCanDo:
    def test_stdio_entries_stay_behind_the_command_gate(self, settings_reset):
        """Running a process on the worker is not the same as calling an API,
        and no amount of convenience should merge the two."""
        listed = {entry["key"]: entry for entry in directory.as_dicts()}

        assert listed["filesystem"]["available"] is False
        assert "administrator" in listed["filesystem"]["blocked_reason"]

    def test_stdio_entries_become_available_once_a_command_is_allowed(
        self, monkeypatch
    ):
        monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", "npx,uvx")
        reset_settings_cache()
        try:
            listed = {entry["key"]: entry for entry in directory.as_dicts()}
            assert listed["filesystem"]["available"] is True
        finally:
            monkeypatch.delenv("CWAP_MCP_ALLOWED_COMMANDS")
            reset_settings_cache()

    def test_every_entry_carries_the_vendors_own_documentation(self):
        """Endpoints move. A link is what makes a stale entry fixable rather
        than a lie the platform keeps telling."""
        for entry in directory.DIRECTORY:
            assert entry.docs_url.startswith("https://"), entry.key
            assert entry.summary and entry.description, entry.key

    def test_a_credential_says_where_to_get_it(self):
        for entry in directory.DIRECTORY:
            for credential in entry.credentials:
                assert credential.how, f"{entry.key}.{credential.name}"

    def test_entries_are_shaped_for_their_transport(self):
        for entry in directory.DIRECTORY:
            if entry.transport is MCPTransport.HTTP:
                assert entry.url.startswith("https://"), entry.key
                assert not entry.command, entry.key
            else:
                assert entry.command, entry.key
                assert not entry.url, entry.key


class TestTheApi:
    def test_the_directory_is_served_to_the_browser(self, client, auth):
        response = client.get("/api/mcp/directory", headers=auth)

        assert response.status_code == 200
        body = response.json()
        keys = {entry["key"] for entry in body["servers"]}
        assert "github" in keys
        assert body["policy"]["directory_enabled"] is True

    def test_it_needs_a_session(self, client):
        assert client.get("/api/mcp/directory").status_code == 401

    def test_no_credential_value_is_ever_in_the_listing(self, client, auth):
        """The directory describes *which* secret is needed, never one."""
        body = client.get("/api/mcp/directory", headers=auth).json()

        for entry in body["servers"]:
            for credential in entry["credentials"]:
                assert set(credential) == {"name", "label", "how", "required", "prefix"}


class TestConnectingIsSurvivable:
    def test_a_stdio_server_is_given_an_environment_it_can_run_in(self):
        """`env={}` hands the child no PATH and no HOME.

        The single most common way to launch an MCP server is `npx -y <package>`,
        and npx needs both. Passing a bare dict broke every one of them in a way
        that looks like a hung server rather than a missing variable — so the
        SDK's default environment is the floor, and configuration goes on top.
        """
        import inspect

        from mcp.client.stdio import get_default_environment
        from mcp_connect import client

        source = inspect.getsource(client._open_transport)
        assert "get_default_environment()" in source
        assert set(get_default_environment()) >= {"PATH", "HOME"}

    def test_a_failure_inside_the_transport_is_reported_in_words(self):
        """anyio supervises the transport in a task group, so a refused
        connection arrives as "unhandled errors in a TaskGroup (1 sub-exception)"
        — true, and useless to somebody who has just typed a URL."""
        from mcp_connect.client import explain

        group = ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [ExceptionGroup("inner", [ConnectionRefusedError("403 Forbidden")])],
        )

        assert explain(group) == "403 Forbidden"

    def test_several_causes_are_summarised_rather_than_dumped(self):
        from mcp_connect.client import explain

        group = ExceptionGroup(
            "boom", [ValueError("first"), ValueError("second"), ValueError("first")]
        )

        assert explain(group) == "first; second"

    def test_an_ordinary_error_is_passed_through(self):
        from mcp_connect.client import explain

        assert explain(RuntimeError("plain")) == "plain"
