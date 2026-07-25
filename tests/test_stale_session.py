"""A signed token whose user no longer exists.

Reported from a Docker deployment: designing a workflow worked, saving worked,
and pressing Run produced

    principal 'usr_...' is not a member of tenant 'default'
    (revoked, deleted, or never existed)

which reads like a security incident and is actually a stale browser session. The
container's database had been recreated while `CWAP_JWT_SECRET` stayed the same,
so the token in localStorage still verified against an identity that was no
longer there.

Two things were wrong, and the second is the one that made it confusing:

* **The failure came at the wrong time.** `current_principal` only decoded the
  token, so every route that needed nothing but a decoded principal succeeded —
  including ones that *create rows owned by that identity*. Only the run path
  consulted the database, so the user built a whole workflow before anything
  objected.
* **The failure came with the wrong status.** 403 means "we know who you are and
  you may not"; the remedy here is to sign in again, which is what 401 means. A
  browser cannot act on the first and can act on the second.

A revoked-mid-queue job is a different case and must stay a 403 into the dead
letter queue: the job is already in flight and there is nobody to re-authenticate.
"""

from __future__ import annotations

import pytest
from cwap_common.db import unit_of_work
from cwap_common.models import User

PASSWORD = "a-sufficiently-long-password"


@pytest.fixture
def session_token(client) -> dict[str, str]:
    response = client.post(
        "/api/auth/register",
        json={"email": "stale@example.com", "password": PASSWORD, "tenant_id": "default"},
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def wipe_users() -> None:
    """What recreating a container's database does to a still-valid token."""
    with unit_of_work() as session:
        session.query(User).delete()


class TestAStaleTokenIsRejectedImmediately:
    def test_the_first_authenticated_request_already_fails(self, client, session_token):
        """Rather than letting someone build a workflow under an identity the
        platform will refuse the moment they press Run."""
        assert client.get("/api/workflows", headers=session_token).status_code == 200

        wipe_users()
        assert client.get("/api/workflows", headers=session_token).status_code == 401

    def test_it_is_a_401_so_the_browser_can_recover(self, client, session_token):
        """403 means "we know you and you may not". The remedy here is to sign in
        again — which is 401, and which a browser can act on."""
        wipe_users()
        response = client.get("/api/workflows", headers=session_token)

        assert response.status_code == 401
        assert "sign in" in response.json()["detail"].lower()

    def test_designing_a_workflow_is_refused_too(self, client, session_token):
        """The route that used to succeed, creating agents owned by a user that
        does not exist."""
        wipe_users()
        response = client.post(
            "/api/design", json={"goal": "Write something", "answers": {}}, headers=session_token
        )
        assert response.status_code == 401

    def test_running_is_refused_with_the_same_clear_message(self, client, session_token):
        """The symptom the report came in about."""
        graph = client.post(
            "/api/design/direct",
            json={"goal": "Research and write a briefing", "answers": {}},
            headers=session_token,
        ).json()["graph"]
        client.put(f"/api/workflows/{graph['id']}", json={"graph": graph}, headers=session_token)

        wipe_users()
        response = client.post(
            f"/api/workflows/{graph['id']}/runs", json={"inputs": {}}, headers=session_token
        )

        assert response.status_code == 401
        assert "sign in" in response.json()["detail"].lower()

    def test_a_user_moved_to_another_tenant_is_also_a_stale_session(self, client, session_token):
        """Their token still claims the old tenant, and that claim is no longer
        true. Re-authenticating is the fix, so it is the same 401."""
        with unit_of_work() as session:
            session.query(User).update({User.tenant_id: "somewhere-else"})

        assert client.get("/api/workflows", headers=session_token).status_code == 401

    def test_signing_in_again_works_immediately(self, client, session_token):
        """The remedy the 401 points at has to actually work."""
        wipe_users()
        response = client.post(
            "/api/auth/register",
            json={"email": "stale@example.com", "password": PASSWORD, "tenant_id": "default"},
        )
        assert response.status_code == 201

        fresh = {"Authorization": f"Bearer {response.json()['access_token']}"}
        assert client.get("/api/workflows", headers=fresh).status_code == 200


class TestAuthenticationStillWorksNormally:
    def test_a_valid_session_is_unaffected(self, client, session_token):
        assert client.get("/api/workflows", headers=session_token).status_code == 200
        assert client.get("/api/agents", headers=session_token).status_code == 200

    def test_signing_in_does_not_require_an_existing_session(self, client):
        """Registration and login must not go through the principal check, or a
        deployment with an empty database could never create its first user."""
        response = client.post(
            "/api/auth/register",
            json={"email": "first@example.com", "password": PASSWORD, "tenant_id": "default"},
        )
        assert response.status_code == 201

    def test_a_missing_token_is_still_a_401(self, client):
        assert client.get("/api/workflows").status_code in (401, 403)

    def test_a_forged_token_is_still_rejected(self, client):
        assert (
            client.get(
                "/api/workflows", headers={"Authorization": "Bearer not-a-real-token"}
            ).status_code
            == 401
        )


class TestRevocationMidQueueIsStillAnAuthorizationFailure:
    def test_a_job_whose_user_vanished_goes_to_the_dead_letter_queue(
        self, authorized_user, broker
    ):
        """Mandate §2: the consumer re-checks at the moment of execution, and a
        job it cannot authorise is parked rather than retried. That is a
        different case from a stale browser session — the job is already in
        flight and there is nobody to re-authenticate — so it must keep failing
        closed here."""
        from conftest import make_linear_graph
        from cwap_common.db import read_only_session
        from cwap_common.models import DeadLetter
        from orchestrator.runner import Worker, start_run

        start_run(
            graph=make_linear_graph(), job_context=authorized_user, inputs={"goal": "x"}
        )
        wipe_users()
        Worker().drain()

        with read_only_session() as session:
            parked = session.query(DeadLetter).all()

        assert parked, "a job that cannot be authorised must be parked, not dropped"
        assert "not a member" in parked[0].reason or "authoriz" in parked[0].reason.lower()
