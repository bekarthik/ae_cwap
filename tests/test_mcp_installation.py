"""A missing library must say so, not impersonate a slow server.

`mcp` was never declared as a dependency. It was present in every development
environment — pulled in long ago and never noticed — and absent from the
container image, which installs exactly what `pyproject.toml` lists. The whole
platform booted, the catalogue of servers rendered, and only *connecting*
failed.

How it failed is the part worth keeping tests for. `from mcp import
ClientSession` sat above the `try` in the task that owns a connection, so it
raised `ModuleNotFoundError` into a bare task: `ready` was never resolved,
nobody was listening for the exception, and the caller waited out the entire
budget. From the browser, a one-line packaging mistake read as *"GitHub's
server stopped responding"* — for days, across five rounds of increasingly
detailed timeout diagnostics that were all, necessarily, wrong.

So: the dependency is declared and shipped, the import is translated into an
answer, and no path through the connection may leave the caller unanswered.
"""

from __future__ import annotations

import builtins
import time
from pathlib import Path

import pytest
import tomllib
from cwap_contracts.v4 import MCPServerConfig, MCPTransport
from mcp_connect import client as mcp_client
from mcp_connect.client import MCPError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_policy_gate(monkeypatch):
    monkeypatch.setattr(mcp_client, "assert_permitted", lambda config: None)


@pytest.fixture(autouse=True)
def _drop_sessions():
    yield
    mcp_client.reset_client(None)


@pytest.fixture
def library_missing(monkeypatch):
    """`import mcp` raises, as it did in the container."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


class TestTheDependencyIsDeclaredAndShipped:
    """The packaging half. Everything else here is damage limitation for it."""

    def test_mcp_is_declared_somewhere(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        project = data["project"]
        declared = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            declared.extend(extra)

        assert any(entry.startswith("mcp") for entry in declared), (
            "the connector imports `mcp` but nothing in pyproject.toml asks for it, "
            "so any environment built from the manifest cannot connect a server"
        )

    def test_the_container_image_installs_it(self):
        """A declared-but-unshipped extra is the same outage with more steps."""
        dockerfile = (ROOT / "deploy" / "Dockerfile.api").read_text()
        install = next(
            line for line in dockerfile.splitlines() if "pip install" in line
        )

        assert "mcp" in install, f"the image does not install it: {install.strip()}"

    def test_the_dev_extra_installs_it_too(self):
        """Otherwise the suite passes only where it happens to be lying around —
        which is precisely how this went unnoticed."""
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        dev = data["project"]["optional-dependencies"]["dev"]

        assert any(entry.startswith("mcp") for entry in dev)


class TestAMissingLibrarySaysSo:
    def test_it_is_reported_immediately(self, library_missing, monkeypatch):
        """Not after the connection budget. The budget is for servers."""
        monkeypatch.setattr(mcp_client, "CONNECT_TIMEOUT", 30.0)
        started = time.monotonic()

        with pytest.raises(MCPError):
            mcp_client.get_client().probe(
                MCPServerConfig(transport=MCPTransport.HTTP, url="http://127.0.0.1:9/mcp"),
                {},
            )

        assert time.monotonic() - started < 5.0

    def test_it_names_the_package(self, library_missing):
        with pytest.raises(MCPError, match="'mcp' package is not installed"):
            mcp_client.get_client().probe(
                MCPServerConfig(transport=MCPTransport.HTTP, url="http://127.0.0.1:9/mcp"),
                {},
            )

    def test_it_absolves_the_server(self, library_missing):
        """The user had a correct URL and a valid token, and was told for days
        that the server was at fault."""
        with pytest.raises(MCPError, match="Nothing about the server"):
            mcp_client.get_client().probe(
                MCPServerConfig(transport=MCPTransport.HTTP, url="http://127.0.0.1:9/mcp"),
                {},
            )

    def test_it_is_not_described_as_a_timeout(self, library_missing):
        with pytest.raises(MCPError) as raised:
            mcp_client.get_client().probe(
                MCPServerConfig(transport=MCPTransport.HTTP, url="http://127.0.0.1:9/mcp"),
                {},
            )

        message = str(raised.value)
        assert "stopped responding" not in message
        assert "gave up after" not in message

    def test_availability_is_reportable(self, library_missing):
        assert mcp_client.mcp_available() is False

    def test_availability_is_true_when_it_is_there(self):
        assert mcp_client.mcp_available() is True


class TestTheDeploymentSaysSoBeforeAnyoneTries:
    def test_the_policy_reports_it(self, library_missing):
        from mcp_connect import policy

        assert policy.describe()["library_installed"] is False

    def test_the_policy_reports_it_when_present(self):
        from mcp_connect import policy

        assert policy.describe()["library_installed"] is True

    def test_the_api_exposes_it(self, client, auth):
        """So the connect dialog can warn instead of listing servers that would
        all fail for the same invisible reason."""
        body = client.get("/api/mcp/directory", headers=auth).json()

        assert "library_installed" in body["policy"]
