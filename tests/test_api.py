"""API Gateway tests — the surface the browser actually talks to."""

from __future__ import annotations

import pytest
from api_gateway.app import create_app
from fastapi.testclient import TestClient
from orchestrator.runner import Worker

from tests.conftest import make_linear_graph

PASSWORD = "a-sufficiently-long-password"


@pytest.fixture
def client() -> TestClient:
    # `create_app` runs the lifespan, which would start the inline worker; the
    # conftest disables it so tests drive the worker explicitly and stay
    # deterministic.
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def auth(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/auth/register",
        json={"email": "builder@example.com", "password": PASSWORD, "tenant_id": "tenant-a"},
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


class TestMeta:
    def test_health_reports_configuration(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["broker"] == "memory"

    def test_contract_fingerprints_are_exposed(self, client):
        body = client.get("/api/contracts").json()
        assert "WorkflowJobPayload@v1" in body


class TestAuth:
    def test_register_then_login(self, client):
        client.post(
            "/api/auth/register",
            json={"email": "a@example.com", "password": PASSWORD, "tenant_id": "t"},
        )
        response = client.post(
            "/api/auth/login", json={"email": "a@example.com", "password": PASSWORD}
        )
        assert response.status_code == 200
        assert "READ_WORKFLOWS" in response.json()["scopes"]

    def test_duplicate_registration_is_rejected(self, client, auth):
        response = client.post(
            "/api/auth/register",
            json={"email": "builder@example.com", "password": PASSWORD, "tenant_id": "t"},
        )
        assert response.status_code == 409

    def test_wrong_password_is_indistinguishable_from_unknown_email(self, client, auth):
        wrong_password = client.post(
            "/api/auth/login", json={"email": "builder@example.com", "password": "nope-nope-nope"}
        )
        unknown_email = client.post(
            "/api/auth/login", json={"email": "ghost@example.com", "password": PASSWORD}
        )
        assert wrong_password.status_code == unknown_email.status_code == 401
        assert wrong_password.json()["detail"] == unknown_email.json()["detail"]

    def test_short_passwords_are_rejected(self, client):
        response = client.post(
            "/api/auth/register",
            json={"email": "b@example.com", "password": "short", "tenant_id": "t"},
        )
        assert response.status_code == 422

    def test_protected_routes_require_a_token(self, client):
        assert client.get("/api/workflows").status_code == 401

    def test_a_forged_token_is_rejected(self, client):
        response = client.get(
            "/api/workflows", headers={"Authorization": "Bearer not.a.real.token"}
        )
        assert response.status_code == 401


class TestWorkflows:
    def test_save_and_read_back(self, client, auth):
        graph = make_linear_graph(workflow_id="wf_demo")
        response = client.put(
            "/api/workflows/wf_demo",
            json={"graph": graph.model_dump(mode="json")},
            headers=auth,
        )
        assert response.status_code == 200
        assert response.json()["version"] == 1

        fetched = client.get("/api/workflows/wf_demo", headers=auth)
        assert fetched.json()["graph"]["nodes"][0]["id"] == "input"

    def test_saving_again_bumps_the_version(self, client, auth):
        graph = make_linear_graph(workflow_id="wf_demo")
        body = {"graph": graph.model_dump(mode="json")}
        client.put("/api/workflows/wf_demo", json=body, headers=auth)
        second = client.put("/api/workflows/wf_demo", json=body, headers=auth)
        assert second.json()["version"] == 2

    def test_a_cyclic_graph_cannot_be_saved(self, client, auth):
        """Structural validation happens at the boundary, so an unexecutable
        graph never reaches storage."""
        graph = make_linear_graph(workflow_id="wf_cycle").model_dump(mode="json")
        graph["edges"].append(
            {"id": "e_cycle", "source": "output", "target": "input", "bindings": {}}
        )
        response = client.put(
            "/api/workflows/wf_cycle", json={"graph": graph}, headers=auth
        )
        assert response.status_code == 422

    def test_path_and_body_ids_must_agree(self, client, auth):
        graph = make_linear_graph(workflow_id="wf_a")
        response = client.put(
            "/api/workflows/wf_b", json={"graph": graph.model_dump(mode="json")}, headers=auth
        )
        assert response.status_code == 400

    def test_another_tenant_gets_a_404_not_a_403(self, client, auth):
        """A 403 would confirm the workflow exists, leaking the id space."""
        graph = make_linear_graph(workflow_id="wf_private")
        client.put(
            "/api/workflows/wf_private",
            json={"graph": graph.model_dump(mode="json")},
            headers=auth,
        )
        other = client.post(
            "/api/auth/register",
            json={"email": "other@example.com", "password": PASSWORD, "tenant_id": "tenant-b"},
        ).json()
        response = client.get(
            "/api/workflows/wf_private",
            headers={"Authorization": f"Bearer {other['access_token']}"},
        )
        assert response.status_code == 404


class TestDesign:
    def test_the_first_turn_asks_before_designing(self, client, auth):
        """Item 2: the system gathers details rather than forcing the user to
        specify a workflow."""
        response = client.post(
            "/api/design", json={"goal": "Sort out our onboarding"}, headers=auth
        )
        assert response.status_code == 200
        body = response.json()
        assert body["stage"] == "clarifying"
        assert body["questions"]
        assert body["understanding"]
        # Every question explains itself; an unexplained one gets answered badly.
        assert all(question["why"] for question in body["questions"])

    def test_answering_produces_a_design_of_agents(self, client, auth):
        answers = {
            "deliverable": "A short report",
            "audience": "My team",
            "constraints": "keep it under a page",
        }
        response = client.post(
            "/api/design",
            json={"goal": "Research our competitors and write it up", "answers": answers},
            headers=auth,
        )
        body = response.json()
        assert body["stage"] == "designed"
        assert body["agents"], "a design with no agents is not a design"
        assert body["graph"]["nodes"]
        assert any(node["type"] == "agent" for node in body["graph"]["nodes"])

    def test_skipping_questions_designs_immediately(self, client, auth):
        response = client.post(
            "/api/design/direct",
            json={"goal": "Plan my weekend trip to Denver"},
            headers=auth,
        )
        body = response.json()
        assert body["stage"] == "designed"
        assert body["graph"] is not None

    def test_agents_are_created_and_listable(self, client, auth):
        client.post(
            "/api/design/direct", json={"goal": "Plan my weekend trip"}, headers=auth
        )
        agents = client.get("/api/agents", headers=auth).json()
        assert agents, "designing a workflow should create its agents"
        assert all(agent["role"] for agent in agents)


class TestKnowledge:
    def test_ingest_list_preview_delete(self, client, auth):
        created = client.post(
            "/api/knowledge/text",
            json={
                "title": "Handbook",
                "content": "Employees may claim meals up to 45 GBP per day when travelling.",
            },
            headers=auth,
        )
        assert created.status_code == 201
        handle = created.json()["handle"]

        listed = client.get("/api/knowledge", headers=auth).json()
        assert [item["handle"] for item in listed] == [handle]

        preview = client.post(
            f"/api/knowledge/{handle}/preview",
            json={"query": "meal allowance", "top_k": 2},
            headers=auth,
        )
        assert "45 GBP" in preview.json()["chunks"][0]["text"]

        assert client.delete(f"/api/knowledge/{handle}", headers=auth).status_code == 204

    def test_binary_uploads_are_refused_with_an_explanation(self, client, auth):
        response = client.post(
            "/api/knowledge/upload",
            files={"file": ("report.pdf", b"%PDF-1.7 binary", "application/pdf")},
            headers=auth,
        )
        assert response.status_code == 415
        assert "supported" in response.json()["detail"].lower()

    def test_a_text_upload_is_indexed(self, client, auth):
        response = client.post(
            "/api/knowledge/upload",
            files={"file": ("notes.md", b"# Notes\nRail travel is preferred in the UK.", "text/markdown")},
            headers=auth,
        )
        assert response.status_code == 201
        assert response.json()["chunk_count"] >= 1

    def test_unknown_handle_previews_as_404(self, client, auth):
        response = client.post(
            "/api/knowledge/kb_missing/preview", json={"query": "x"}, headers=auth
        )
        assert response.status_code == 404


class TestRuns:
    def _save_workflow(self, client, auth) -> str:
        graph = make_linear_graph(workflow_id="wf_run")
        client.put(
            "/api/workflows/wf_run",
            json={"graph": graph.model_dump(mode="json")},
            headers=auth,
        )
        return "wf_run"

    def test_starting_a_run_returns_immediately(self, client, auth):
        workflow_id = self._save_workflow(client, auth)
        response = client.post(
            f"/api/workflows/{workflow_id}/runs",
            json={"inputs": {"goal": "Plan a trip to Denver"}},
            headers=auth,
        )
        assert response.status_code == 202
        assert response.json()["status"] == "PENDING"

    def test_the_run_report_explains_every_step(self, client, auth):
        workflow_id = self._save_workflow(client, auth)
        run_id = client.post(
            f"/api/workflows/{workflow_id}/runs",
            json={"inputs": {"goal": "Plan a trip to Denver"}},
            headers=auth,
        ).json()["run_id"]

        Worker().drain()

        report = client.get(f"/api/runs/{run_id}", headers=auth).json()
        assert report["run"]["status"] == "SUCCEEDED"
        assert [step["node_id"] for step in report["steps"]] == ["input", "think", "output"]
        assert report["logs"][0]["event"] == "run.accepted"
        assert "Denver" in report["result"]["result"]

    def test_an_unsaved_graph_can_be_run_from_the_canvas(self, client, auth):
        graph = make_linear_graph(workflow_id="wf_scratch")
        response = client.post(
            "/api/workflows/wf_scratch/runs",
            json={"inputs": {"goal": "test"}, "graph": graph.model_dump(mode="json")},
            headers=auth,
        )
        assert response.status_code == 202

    def test_running_an_unknown_workflow_is_a_404(self, client, auth):
        response = client.post(
            "/api/workflows/wf_ghost/runs", json={"inputs": {}}, headers=auth
        )
        assert response.status_code == 404

    @staticmethod
    def _outward_graph():
        graph = make_linear_graph(workflow_id="wf_http").model_dump(mode="json")
        graph["nodes"][1] = {
            "id": "think",
            "type": "http_request",
            "label": "Call CRM",
            "params": {"method": "GET", "url": "https://example.com/api"},
            "position": {"x": 250, "y": 0},
            "knowledge_handle": None,
        }
        return graph

    def test_an_account_without_the_write_scope_is_refused(self, client, auth):
        """Least privilege: an account that lacks the grant cannot reach outward.

        The scope is stripped explicitly rather than relied upon to be absent —
        a workspace's first account now has it, and a test that passes only
        because nobody was ever granted anything was how a permission nothing
        could grant went unnoticed.
        """
        from cwap_common.db import unit_of_work
        from cwap_common.models import User

        with unit_of_work() as session:
            for user in session.query(User).all():
                user.scopes = ["READ_WORKFLOWS"]

        response = client.post(
            "/api/workflows/wf_http/runs",
            json={"inputs": {"goal": "x"}, "graph": self._outward_graph()},
            headers=auth,
        )
        assert response.status_code == 403
        assert "WRITE_EXTERNAL" in response.json()["detail"]

    def test_the_workspace_owner_may_run_it(self, client, auth):
        """The case the platform used to make impossible: the person who
        installed it, running their own outward workflow."""
        response = client.post(
            "/api/workflows/wf_http/runs",
            json={"inputs": {"goal": "x"}, "graph": self._outward_graph()},
            headers=auth,
        )

        assert response.status_code == 202, response.text

    def test_runs_are_listed_for_the_tenant(self, client, auth):
        workflow_id = self._save_workflow(client, auth)
        client.post(
            f"/api/workflows/{workflow_id}/runs",
            json={"inputs": {"goal": "x"}},
            headers=auth,
        )
        runs = client.get("/api/runs", headers=auth).json()
        assert len(runs) == 1


class TestLogStream:
    def test_the_websocket_replays_the_run_log(self, client, auth):
        graph = make_linear_graph(workflow_id="wf_ws")
        client.put(
            "/api/workflows/wf_ws", json={"graph": graph.model_dump(mode="json")}, headers=auth
        )
        token = auth["Authorization"].split(" ", 1)[1]
        run_id = client.post(
            "/api/workflows/wf_ws/runs", json={"inputs": {"goal": "x"}}, headers=auth
        ).json()["run_id"]

        Worker().drain()

        with client.websocket_connect(f"/api/runs/{run_id}/logs?token={token}") as socket:
            first = socket.receive_json()
        assert first["event"] == "run.accepted"
        assert first["run_id"] == run_id

    def test_an_unauthenticated_socket_is_closed(self, client, auth):
        graph = make_linear_graph(workflow_id="wf_ws2")
        client.put(
            "/api/workflows/wf_ws2", json={"graph": graph.model_dump(mode="json")}, headers=auth
        )
        run_id = client.post(
            "/api/workflows/wf_ws2/runs", json={"inputs": {"goal": "x"}}, headers=auth
        ).json()["run_id"]

        with (
            pytest.raises(Exception),
            client.websocket_connect(f"/api/runs/{run_id}/logs") as socket,
        ):
            socket.receive_json()


class TestAdmin:
    def test_dead_letters_are_visible_and_resubmittable(self, client, auth, broker):
        from cwap_common.contract_gateway import ConsumerWrapper
        from cwap_common.settings import get_settings

        broker.publish(get_settings().work_queue, "{ not a payload }")
        ConsumerWrapper().next()

        records = client.get("/api/admin/dead-letters", headers=auth).json()
        assert len(records) == 1
        assert records[0]["reason_code"] == "CONTRACT_VIOLATION"

        response = client.post(
            f"/api/admin/dead-letters/{records[0]['id']}/resubmit", headers=auth
        )
        assert response.status_code == 202
        assert broker.depth(get_settings().work_queue) == 1

    def test_queue_depths_are_reported(self, client, auth):
        body = client.get("/api/admin/queues", headers=auth).json()
        assert all(depth == 0 for depth in body.values())
