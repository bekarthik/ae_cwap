"""Permission to act outward has to be grantable by somebody who exists.

A user installed the platform, registered the only account on it, connected a
GitHub server, built a workflow and pressed Run — and was told to *"ask an
administrator to grant it"*. There was no administrator: it is a self-hosted
single-tenant install and they had made the only account. There was also no
grant. `WRITE_EXTERNAL` appeared in no registration path, no endpoint, no
command and no setting anywhere in the codebase. The scope could be required
and could never be given.

A permission that nothing can grant is not a permission. So an operator now
chooses with `CWAP_WRITE_EXTERNAL`, and the default is that a workspace's first
account — whoever installed this — has it.

What that does *not* do is open egress, and the tests at the bottom hold that
line. An HTTP node still needs `CWAP_HTTP_ALLOWLIST`; an MCP host still needs
the directory or the allow-list; every external call still passes the two-phase
gate. This decides who may ask, not what may be reached.
"""

from __future__ import annotations

import pytest
from api_gateway.security import WRITE_EXTERNAL, create_user, grant_owner_scopes
from cwap_common.db import read_only_session, unit_of_work
from cwap_common.models import User


def account(email: str, tenant: str = "acme"):
    return create_user(email, "hunter2hunter2", tenant_id=tenant)


def scopes_of(email: str) -> set[str]:
    with read_only_session() as session:
        user = session.query(User).filter_by(email=email).one()
        return set(user.scopes or [])


class TestTheFirstAccountCanUseItsOwnWorkspace:
    def test_the_first_account_may_act_outward(self, isolated_platform):
        principal = account("founder@example.com")

        assert WRITE_EXTERNAL in principal.scopes

    def test_a_later_account_may_not(self, isolated_platform):
        """Which is what keeps the scope meaningful once a workspace is shared."""
        account("founder@example.com")
        second = account("colleague@example.com")

        assert WRITE_EXTERNAL not in second.scopes

    def test_each_workspace_has_its_own_first_account(self, isolated_platform):
        account("founder@example.com", tenant="acme")
        other = account("founder@other.example.com", tenant="globex")

        assert WRITE_EXTERNAL in other.scopes

    def test_everyone_still_gets_the_base_scope(self, isolated_platform):
        account("founder@example.com")
        second = account("colleague@example.com")

        assert "READ_WORKFLOWS" in second.scopes


class TestTheOperatorDecides:
    def test_it_can_be_opened_to_everyone(self, isolated_platform, monkeypatch):
        monkeypatch.setenv("CWAP_WRITE_EXTERNAL", "everyone")
        from cwap_common.settings import get_settings

        get_settings.cache_clear()
        account("founder@example.com")

        assert WRITE_EXTERNAL in account("colleague@example.com").scopes

    def test_it_can_be_closed_entirely(self, isolated_platform, monkeypatch):
        monkeypatch.setenv("CWAP_WRITE_EXTERNAL", "nobody")
        from cwap_common.settings import get_settings

        get_settings.cache_clear()

        assert WRITE_EXTERNAL not in account("founder@example.com").scopes

    def test_an_unrecognised_setting_does_not_crash_registration(
        self, isolated_platform, monkeypatch
    ):
        """A typo in an environment variable must not stop people signing up."""
        monkeypatch.setenv("CWAP_WRITE_EXTERNAL", "sure-why-not")
        from cwap_common.settings import get_settings

        get_settings.cache_clear()

        assert "READ_WORKFLOWS" in account("founder@example.com").scopes


class TestAccountsThatAlreadyExistAreFixed:
    """Registration alone would only help people who have not signed up yet.

    And the consumer-side authorisation re-check reads scopes from the
    *database*, so an existing owner would have passed the gateway and then had
    the job parked in the dead letter queue — a worse failure than the one being
    fixed.
    """

    def test_an_existing_owner_is_granted_on_boot(self, isolated_platform):
        account("founder@example.com")
        with unit_of_work() as session:
            session.query(User).filter_by(email="founder@example.com").one().scopes = [
                "READ_WORKFLOWS"
            ]

        grant_owner_scopes()

        assert WRITE_EXTERNAL in scopes_of("founder@example.com")

    def test_a_later_account_is_left_alone(self, isolated_platform):
        account("founder@example.com")
        account("colleague@example.com")

        grant_owner_scopes()

        assert WRITE_EXTERNAL not in scopes_of("colleague@example.com")

    def test_running_it_twice_does_not_duplicate_the_scope(self, isolated_platform):
        account("founder@example.com")

        grant_owner_scopes()
        grant_owner_scopes()

        with read_only_session() as session:
            stored = session.query(User).filter_by(email="founder@example.com").one()

        assert stored.scopes.count(WRITE_EXTERNAL) == 1

    def test_it_reports_who_it_granted_to(self, isolated_platform):
        """So the boot log says what changed rather than changing it silently."""
        account("founder@example.com")
        with unit_of_work() as session:
            session.query(User).filter_by(email="founder@example.com").one().scopes = [
                "READ_WORKFLOWS"
            ]

        assert grant_owner_scopes() == ["founder@example.com"]

    def test_nobody_mode_grants_nothing_retroactively(self, isolated_platform, monkeypatch):
        account("founder@example.com")
        with unit_of_work() as session:
            session.query(User).filter_by(email="founder@example.com").one().scopes = [
                "READ_WORKFLOWS"
            ]
        monkeypatch.setenv("CWAP_WRITE_EXTERNAL", "nobody")
        from cwap_common.settings import get_settings

        get_settings.cache_clear()

        assert grant_owner_scopes() == []


class TestTheRefusalNamesSomethingReal:
    @pytest.fixture
    def refusal(self, client, auth) -> str:
        """The message a non-owner gets when the workflow acts outward."""
        from api_gateway.routers import runs

        # The graph's shape is beside the point here; what is being read is the
        # sentence, so the outward check is forced rather than constructed.
        runs_module = runs
        original = runs_module._needs_write_scope
        original_named = runs_module._outward_steps
        runs_module._needs_write_scope = lambda graph, tenant_id: True
        runs_module._outward_steps = lambda graph, tenant_id: "the step 'Publish'"
        try:
            with unit_of_work() as session:
                for user in session.query(User).all():
                    user.scopes = ["READ_WORKFLOWS"]

            graph = {
                "id": "wf_outward",
                "name": "Outward",
                "nodes": [
                    {"id": "input", "type": "input", "params": {"fields": ["goal"]}},
                    {"id": "output", "type": "output", "params": {}},
                ],
                "edges": [{"id": "e1", "source": "input", "target": "output"}],
            }
            client.put("/api/workflows/wf_outward", json={"graph": graph}, headers=auth)
            response = client.post(
                "/api/workflows/wf_outward/runs", json={"inputs": {}}, headers=auth
            )
            assert response.status_code == 403, response.text
            return response.json()["detail"]
        finally:
            runs_module._needs_write_scope = original
            runs_module._outward_steps = original_named

    def test_it_no_longer_invents_an_administrator(self, refusal):
        """There is nobody to ask on a self-hosted install, and there was no
        action they could have taken."""
        assert "administrator" not in refusal.lower()

    def test_it_names_the_setting_that_changes_it(self, refusal):
        assert "CWAP_WRITE_EXTERNAL" in refusal

    def test_it_names_the_step_that_needs_it(self, refusal):
        """"A step" is true and useless on a canvas with nine of them."""
        assert "Publish" in refusal


class TestAGrantReachesASessionAlreadyOpen:
    """Scopes come from the database, not from the token that claims them.

    They used to be read from the token at the gateway and from the database by
    the authorisation service, so the two could disagree. A scope granted after
    a token was issued did nothing until the user signed out and in — which,
    for somebody being told to "ask an administrator", is an invisible second
    step after an already-invisible first one. And a scope *revoked* still
    passed the gateway's own check, then failed deep inside the producer with a
    message about job context.
    """

    def test_a_grant_applies_without_signing_in_again(self, client, auth):
        with unit_of_work() as session:
            for user in session.query(User).all():
                user.scopes = ["READ_WORKFLOWS"]

        grant_owner_scopes()  # as the gateway does at boot

        # Same bearer token as before the grant.
        assert WRITE_EXTERNAL in client.get("/api/auth/me", headers=auth).json()["scopes"]

    def test_a_revocation_applies_just_as_fast(self, client, auth):
        with unit_of_work() as session:
            for user in session.query(User).all():
                user.scopes = ["READ_WORKFLOWS"]

        assert WRITE_EXTERNAL not in client.get("/api/auth/me", headers=auth).json()["scopes"]


class TestGrantingItDoesNotOpenEgress:
    """The line this change must not cross.

    Granting the scope decides *who may ask*. What may actually be reached is a
    separate gate, and it is still shut.
    """

    def test_an_http_node_still_needs_the_allowlist(self, isolated_platform):
        from cwap_common.settings import get_settings

        assert get_settings().http_allowed_hosts == frozenset()

    def test_the_owner_scope_does_not_add_hosts(self, isolated_platform):
        from cwap_common.settings import get_settings

        account("founder@example.com")

        assert get_settings().http_allowed_hosts == frozenset()

    def test_stdio_mcp_servers_stay_disabled(self, isolated_platform):
        from mcp_connect.policy import allowed_commands

        account("founder@example.com")

        assert allowed_commands() == frozenset()
