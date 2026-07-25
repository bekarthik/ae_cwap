"""MCP connectors: connecting a server, and agents using its tools.

These run against a **real MCP server** — `mcp_fixture_server.py`, launched as a
subprocess over stdio — so the protocol itself is exercised: handshake,
`tools/list`, `tools/call`, content blocks, server-side errors. A mock would
agree with whatever the client happened to do, which is precisely the thing that
needs checking when integrating someone else's protocol.

The fixture imitates a code-hosting server, because that is the example the
platform has to work for: read a file, search code, open a pull request.

Two properties carry more weight than the plumbing:

* **Connecting is gated by an operator, not by the tenant.** A stdio server is
  process execution on the worker; without an allow-list, "connect an MCP
  server" is a remote shell.
* **An MCP tool is a skill.** Everything the platform already does for skills
  then applies without a second implementation — agents hold them, models see
  them, failures come back to the agent, memory accrues per skill.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from agents import registry as agent_registry
from agents import runtime
from cwap_contracts.v3 import (
    MCPServerConfig,
    MCPTransport,
    SkillKind,
    SkillOrigin,
    SkillParameter,
    SkillProposal,
)
from llm_proxy.client import reset_provider_cache
from mcp_connect import policy, registry
from mcp_connect.client import reset_client
from pydantic import ValidationError
from skills import execution
from skills import registry as skill_registry
from test_agents import ScriptedAgentProvider

TENANT = "tenant-a"
FIXTURE = str(Path(__file__).parent / "mcp_fixture_server.py")


@pytest.fixture(autouse=True)
def _drop_sessions():
    """A live session must not leak between tests — it holds a subprocess."""
    yield
    reset_client(None)


@pytest.fixture
def permitted(monkeypatch):
    """An operator who has allowed this deployment to launch the interpreter.

    Listed by its exact path, which is the stricter of the two accepted forms —
    a bare name would authorise whatever PATH resolves to at run time.
    """
    monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", sys.executable)


def stdio_config(args: list[str] | None = None) -> MCPServerConfig:
    return MCPServerConfig(
        transport=MCPTransport.STDIO,
        command=sys.executable,
        args=args if args is not None else [FIXTURE],
    )


@pytest.fixture
def connected(permitted):
    result = registry.connect(
        TENANT, name="codehost", config=stdio_config(), description="A code host."
    )
    assert result.ok, result.message
    return registry.find_by_name(TENANT, "codehost")


class TestPolicy:
    def test_stdio_is_refused_when_no_command_is_allowed(self):
        """Without this, "connect an MCP server" is a remote shell."""
        result = registry.test(stdio_config())

        assert result.ok is False
        assert result.blocked is True
        assert "administrator" in result.message

    def test_a_command_outside_the_allowlist_is_refused(self, monkeypatch):
        monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", "npx")
        result = registry.test(
            MCPServerConfig(transport=MCPTransport.STDIO, command="bash", args=["-c", "id"])
        )

        assert result.ok is False
        assert result.blocked is True

    def test_a_path_cannot_smuggle_past_a_name_allowlist(self, monkeypatch):
        """`/tmp/../bin/npx` must not satisfy an allow-list entry of `npx`."""
        monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", "npx")
        result = registry.test(
            MCPServerConfig(transport=MCPTransport.STDIO, command="/tmp/evil/../../bin/npx")
        )

        assert result.ok is False
        assert result.blocked is True

    def test_an_http_server_needs_an_allowlisted_host(self):
        result = registry.test(
            MCPServerConfig(transport=MCPTransport.HTTP, url="https://mcp.example.com/x")
        )

        assert result.ok is False
        assert result.blocked is True
        assert "allow-list" in result.message

    def test_an_http_server_to_an_allowlisted_host_passes_policy(self, monkeypatch):
        """It will still fail to connect — nothing is listening — but it must
        fail as a connection problem, not as a policy refusal."""
        monkeypatch.setenv("CWAP_HTTP_ALLOWLIST", "mcp.example.com")
        from cwap_common.settings import reset_settings_cache

        reset_settings_cache()

        result = registry.test(
            MCPServerConfig(transport=MCPTransport.HTTP, url="https://mcp.example.com/x")
        )
        assert result.blocked is False

    def test_a_bare_name_entry_resolves_through_path(self, monkeypatch):
        """The other accepted form: an operator writes `npx` and does not have to
        know where it is installed."""
        from mcp_connect.policy import assert_permitted

        monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", "sh")
        assert_permitted(MCPServerConfig(transport=MCPTransport.STDIO, command="sh"))

    def test_a_bare_name_entry_does_not_authorise_a_path(self, monkeypatch):
        """Otherwise an allow-list of `sh` is satisfied by /tmp/uploaded/sh, and
        the gate is decoration."""
        from mcp_connect.policy import MCPPolicyError, assert_permitted

        monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", "sh")
        with pytest.raises(MCPPolicyError):
            assert_permitted(
                MCPServerConfig(transport=MCPTransport.STDIO, command="/tmp/uploaded/sh")
            )

    def test_an_allowlisted_command_that_is_not_installed_says_so(self, monkeypatch):
        from mcp_connect.policy import MCPPolicyError, assert_permitted

        monkeypatch.setenv("CWAP_MCP_ALLOWED_COMMANDS", "definitely-not-installed")
        with pytest.raises(MCPPolicyError, match="not installed"):
            assert_permitted(
                MCPServerConfig(
                    transport=MCPTransport.STDIO, command="definitely-not-installed"
                )
            )

    def test_the_deployment_reports_what_it_permits(self, permitted):
        """So the UI can say "stdio is disabled here" before a user fills in a
        form that will be refused."""
        described = policy.describe()
        assert described["stdio_enabled"] is True
        assert sys.executable in described["allowed_commands"]


class TestConnecting:
    def test_a_real_server_connects_and_reports_its_tools(self, permitted):
        result = registry.test(stdio_config())

        assert result.ok, result.message
        assert {tool.name for tool in result.tools} >= {
            "read_file",
            "search_code",
            "create_pull_request",
        }

    def test_a_tool_marked_read_only_is_recognised_as_such(self, permitted):
        result = registry.test(stdio_config())
        by_name = {tool.name: tool for tool in result.tools}

        assert by_name["read_file"].read_only is True
        assert by_name["create_pull_request"].read_only is False

    def test_a_server_that_does_not_start_fails_clearly(self, permitted):
        result = registry.test(stdio_config(args=["/no/such/script.py"]))
        assert result.ok is False
        assert result.blocked is False

    def test_connecting_stores_the_server(self, connected):
        assert connected.name == "codehost"
        assert connected.is_connected
        assert [server.name for server in registry.list_all(TENANT)] == ["codehost"]

    def test_a_server_is_verified_before_it_is_stored(self, permitted):
        """A stored server that has never connected is a row that looks like a
        working integration and is not."""
        result = registry.connect(
            TENANT, name="broken", config=stdio_config(args=["/no/such/script.py"])
        )

        assert result.ok is False
        assert registry.find_by_name(TENANT, "broken") is None

    def test_a_name_is_claimed_once(self, connected):
        with pytest.raises(registry.MCPRegistryError):
            registry.connect(TENANT, name="codehost", config=stdio_config())

    def test_one_tenant_cannot_see_anothers_server(self, connected):
        assert registry.list_all("tenant-b") == []

    def test_credentials_are_not_part_of_the_record(self, permitted):
        """A server can be listed, exported and logged without leaking a token."""
        registry.connect(
            TENANT,
            name="withsecret",
            config=stdio_config(),
            credentials={"GITHUB_TOKEN": "ghp_secret"},
        )
        record = registry.find_by_name(TENANT, "withsecret")

        assert record.has_credentials is True
        assert "ghp_secret" not in record.model_dump_json()


class TestToolsBecomeSkills:
    def test_each_tool_becomes_a_skill(self, connected):
        names = {skill.name for skill in skill_registry.list_all(TENANT)}
        assert {"codehost_read_file", "codehost_create_pull_request"} <= names

    def test_the_skill_is_marked_as_coming_from_a_server(self, connected):
        skill = skill_registry.find_by_name(TENANT, "codehost_read_file")
        assert skill.origin is SkillOrigin.MCP
        assert skill.kind is SkillKind.MCP

    def test_the_skill_names_its_server_and_tool(self, connected):
        skill = skill_registry.find_by_name(TENANT, "codehost_read_file")
        assert skill.definition["server_id"] == connected.id
        assert skill.definition["tool_name"] == "read_file"

    def test_two_servers_offering_the_same_tool_do_not_collide(self, permitted, connected):
        registry.connect(TENANT, name="other", config=stdio_config())
        names = {skill.name for skill in skill_registry.list_all(TENANT)}

        assert "codehost_read_file" in names
        assert "other_read_file" in names

    def test_the_servers_own_schema_reaches_the_model_verbatim(self, connected):
        """A paraphrase would drift from what the server validates against, and
        the model would be told a shape that then gets rejected. The fixture's
        `labels: list[str] | None` becomes an `anyOf`, which the platform's own
        parameter model cannot express — so it has to be passed through, not
        rebuilt."""
        reported = {tool.name: tool for tool in connected.tools}["create_pull_request"]
        skill = skill_registry.find_by_name(TENANT, "codehost_create_pull_request")

        assert skill.tool_schema()["input_schema"] == reported.input_schema
        assert "title" in reported.input_schema["required"]

    def test_a_structured_parameter_survives(self, connected):
        """v2 could only express string/number/integer/boolean, which would have
        made this tool unusable."""
        skill = skill_registry.find_by_name(TENANT, "codehost_create_pull_request")
        by_name = {param.name: param for param in skill.parameters}
        assert by_name["labels"].type in ("array", "string")

    def test_resyncing_is_idempotent(self, connected):
        before = len(skill_registry.list_all(TENANT))
        registry.sync_skills(TENANT, connected.id)
        assert len(skill_registry.list_all(TENANT)) == before

    def test_disconnecting_removes_the_skills(self, connected):
        assert registry.disconnect(TENANT, connected.id) is True
        names = {skill.name for skill in skill_registry.list_all(TENANT)}
        assert not any(name.startswith("codehost_") for name in names)

    def test_disabling_keeps_the_server_but_stops_its_skills(self, connected):
        registry.set_enabled(TENANT, connected.id, False)

        assert registry.find_by_name(TENANT, "codehost").enabled is False
        skill = skill_registry.find_by_name(TENANT, "codehost_read_file")
        assert skill.definition.get("unavailable") is True


class TestInvokingTools:
    def context(self, **overrides):
        base = dict(
            tenant_id=TENANT,
            run_id="run_1",
            step_execution_id="se_1",
            allow_side_effects=True,
        )
        base.update(overrides)
        return execution.SkillContext(**base)

    def test_a_read_only_tool_runs(self, connected):
        skill = skill_registry.find_by_name(TENANT, "codehost_read_file")
        outcome = execution.execute(skill, {"path": "README.md"}, self.context())

        assert not outcome.is_error
        assert "contents of README.md" in outcome.output

    def test_a_structured_argument_reaches_the_server_unflattened(self, connected):
        """The model was handed the server's schema, so what it produced is what
        the server expects. Coercing it would break every tool like this."""
        skill = skill_registry.find_by_name(TENANT, "codehost_create_pull_request")
        outcome = execution.execute(
            skill,
            {"title": "Fix the thing", "body": "Details", "labels": ["bug", "urgent"]},
            self.context(),
        )

        assert not outcome.is_error
        assert "['bug', 'urgent']" in outcome.output

    def test_a_server_side_error_is_reported_to_the_agent(self, connected):
        """Not raised: the agent can try something else, which is the point of
        giving it a loop."""
        skill = skill_registry.find_by_name(TENANT, "codehost_always_fails")
        outcome = execution.execute(skill, {"reason": "nope"}, self.context())

        assert outcome.is_error
        assert "nope" in outcome.output

    def test_a_read_only_run_may_not_use_a_write_tool(self, connected):
        """An agent gets no privilege the user running it has."""
        skill = skill_registry.find_by_name(TENANT, "codehost_create_pull_request")
        outcome = execution.execute(
            skill,
            {"title": "T", "body": "B"},
            self.context(allow_side_effects=False),
        )

        assert outcome.is_error
        assert "write scope" in outcome.output

    def test_a_read_only_run_may_still_read(self, connected):
        """Demanding a write scope to read a repository would make the common
        case need a privilege it never uses."""
        skill = skill_registry.find_by_name(TENANT, "codehost_read_file")
        outcome = execution.execute(
            skill, {"path": "README.md"}, self.context(allow_side_effects=False)
        )
        assert not outcome.is_error

    def test_a_redelivered_write_does_not_fire_twice(self, connected):
        """The §3.A gate applies to MCP exactly as it does to an HTTP node — a
        retried step must not open two pull requests."""
        skill = skill_registry.find_by_name(TENANT, "codehost_create_pull_request")
        arguments = {"title": "Fix the thing", "body": "Details"}

        first = execution.execute(skill, arguments, self.context())
        second = execution.execute(skill, arguments, self.context())

        assert "opened #1" in first.output
        # The second call replays the recorded response rather than invoking the
        # tool again, so the server's counter never reaches 2.
        assert "opened #2" not in second.output

    def test_a_different_call_in_the_same_step_is_not_deduplicated(self, connected):
        """Same tool, different arguments, is a different operation."""
        skill = skill_registry.find_by_name(TENANT, "codehost_create_pull_request")

        execution.execute(skill, {"title": "One", "body": "B"}, self.context())
        second = execution.execute(skill, {"title": "Two", "body": "B"}, self.context())

        assert "opened #2" in second.output

    def test_a_disabled_server_reports_clearly(self, connected):
        registry.set_enabled(TENANT, connected.id, False)
        reloaded = skill_registry.find_by_name(TENANT, "codehost_read_file")

        outcome = execution.execute(reloaded, {"path": "x"}, self.context())
        assert outcome.is_error
        assert "no longer offered" in outcome.output or "switched off" in outcome.output

    def test_a_disconnected_server_reports_clearly(self, connected):
        skill = skill_registry.find_by_name(TENANT, "codehost_read_file")
        registry.disconnect(TENANT, connected.id)

        outcome = execution.execute(skill, {"path": "x"}, self.context())
        assert outcome.is_error
        assert "no longer connected" in outcome.output

    def test_the_session_is_reused_across_calls(self, connected):
        """A process per tool call would make a three-tool agent start three
        subprocesses."""
        skill = skill_registry.find_by_name(TENANT, "codehost_read_file")
        for _ in range(3):
            assert not execution.execute(skill, {"path": "x"}, self.context()).is_error

        from mcp_connect.client import get_client

        assert len(get_client()._sessions) == 1  # noqa: SLF001 - the property under test


class TestAnAgentUsesAConnectedServer:
    def test_an_agent_can_be_given_an_mcp_skill_and_use_it(self, connected):
        """The whole point, end to end: an agent that can read a repository is an
        agent holding a skill, exactly like one that can summarise text."""
        read_file = skill_registry.find_by_name(TENANT, "codehost_read_file")
        agent = agent_registry.create(
            TENANT,
            name="Reviewer",
            role="a code reviewer",
            skill_ids=[read_file.id],
        )
        provider = ScriptedAgentProvider(
            [("codehost_read_file", {"path": "src/main.py"})],
            "The file looks fine.",
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(
            agent,
            "Review src/main.py",
            runtime.AgentRunContext(
                tenant_id=TENANT,
                run_id="run_1",
                step_execution_id="se_1",
                allow_side_effects=True,
            ),
        )

        assert result.skills_used == ["codehost_read_file"]
        assert result.status == runtime.OBJECTIVE_MET
        [outcome] = result.turns[0].tool_results
        assert "contents of src/main.py" in outcome.output

    def test_the_tool_is_offered_to_the_model_with_the_servers_schema(self, connected):
        create_pr = skill_registry.find_by_name(TENANT, "codehost_create_pull_request")
        agent = agent_registry.create(
            TENANT, name="Contributor", role="a contributor", skill_ids=[create_pr.id]
        )
        provider = ScriptedAgentProvider("Nothing to do.")
        reset_provider_cache(provider)

        runtime.run_agent(
            agent,
            "Open a pull request",
            runtime.AgentRunContext(tenant_id=TENANT, allow_side_effects=True),
        )

        assert provider.tools_offered[0] == ["codehost_create_pull_request"]

    def test_a_failing_tool_does_not_fail_the_run(self, connected):
        failing = skill_registry.find_by_name(TENANT, "codehost_always_fails")
        agent = agent_registry.create(
            TENANT, name="Trier", role="a trier", skill_ids=[failing.id]
        )
        provider = ScriptedAgentProvider(
            [("codehost_always_fails", {"reason": "no"})],
            "That did not work, so here is what I can say.",
        )
        reset_provider_cache(provider)

        result = runtime.run_agent(
            agent,
            "Try it",
            runtime.AgentRunContext(tenant_id=TENANT, allow_side_effects=True),
        )

        assert result.status == runtime.OBJECTIVE_MET
        assert result.turns[0].tool_results[0].is_error


class TestSynthesisCannotInventAConnection:
    def test_a_proposal_may_not_be_an_mcp_tool(self):
        """Connecting to something outside the platform is a decision a human
        makes with a credential behind it — enforced at the contract, so no code
        path can bypass it."""
        with pytest.raises(ValidationError, match="connect the server first"):
            SkillProposal(
                name="sneaky",
                description="Reach a server nobody connected.",
                kind=SkillKind.MCP,
                parameters=[SkillParameter(name="x", description="x")],
                definition={"server_id": "mcp_made_up", "tool_name": "exfiltrate"},
            )

    def test_an_mcp_skill_still_needs_a_server_and_tool(self):
        from cwap_contracts.v3 import SkillDefinition

        with pytest.raises(ValidationError):
            SkillDefinition(
                id="skl_1",
                tenant_id=TENANT,
                name="broken",
                description="Missing its target.",
                kind=SkillKind.MCP,
                definition={},
            )


class TestTheApi:
    def test_connected_servers_are_listed_with_the_deployment_policy(self, client, auth):
        body = client.get("/api/mcp", headers=auth).json()

        assert body["servers"] == []
        assert "stdio_enabled" in body["policy"]

    def test_connecting_a_forbidden_command_is_a_403(self, client, auth):
        response = client.post(
            "/api/mcp",
            json={
                "name": "shell",
                "config": {"transport": "stdio", "command": "bash", "args": ["-c", "id"]},
            },
            headers=auth,
        )
        assert response.status_code == 403

    def test_connecting_a_real_server_imports_its_tools(self, client, auth, permitted):
        response = client.post(
            "/api/mcp",
            json={
                "name": "codehost",
                "config": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [FIXTURE],
                },
            },
            headers=auth,
        )
        assert response.status_code == 201, response.text
        assert response.json()["ok"] is True

        skills = client.get("/api/skills", headers=auth).json()
        assert any(skill["name"] == "codehost_read_file" for skill in skills)

    def test_a_credential_is_never_returned(self, client, auth, permitted):
        client.post(
            "/api/mcp",
            json={
                "name": "codehost",
                "config": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [FIXTURE],
                },
                "credentials": {"GITHUB_TOKEN": "ghp_secret"},
            },
            headers=auth,
        )
        body = client.get("/api/mcp", headers=auth).text
        assert "ghp_secret" not in body

    def test_mcp_requires_authentication(self, client):
        assert client.get("/api/mcp").status_code in (401, 403)
