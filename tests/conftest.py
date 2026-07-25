"""Shared fixtures.

Every test gets its own database, its own broker, and a fresh settings cache, so
tests can run in any order and none of them can see another's state.

SQLite is the default because it needs nothing installed and gives each test a
private file. But SQLite and PostgreSQL are not interchangeable in the places
that matter most here — `ON CONFLICT` reporting, JSONB, and locking all differ —
and a bug that only appears on the production database is the worst kind. Set
`CWAP_TEST_DATABASE_URL` to a PostgreSQL DSN and the whole suite runs against it,
each test in its own schema:

    CWAP_TEST_DATABASE_URL=postgresql+psycopg://cwap@localhost/cwap pytest
"""

from __future__ import annotations

import os
import uuid

import pytest
from cwap_common import db
from cwap_common.broker import InMemoryBroker, set_broker
from cwap_common.logbus import log_bus
from cwap_common.settings import reset_settings_cache
from cwap_contracts.v4 import (
    JobContext,
    NodeType,
    PermissionRequirement,
    Position,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
)
from knowledge.embeddings import HashingEmbedder, set_embedder
from llm_proxy.client import StubProvider, reset_provider_cache

#: A PostgreSQL DSN here runs the whole suite against Postgres instead of SQLite.
POSTGRES_URL = os.environ.get("CWAP_TEST_DATABASE_URL", "")


@pytest.fixture
def database_url(tmp_path):
    """A private database for one test, on whichever backend is configured.

    On PostgreSQL that means a throwaway schema rather than a throwaway file —
    same isolation, and it is dropped whether or not the test passes.
    """
    if not POSTGRES_URL:
        yield f"sqlite:///{tmp_path / 'cwap-test.sqlite3'}"
        return

    from sqlalchemy import create_engine, text

    schema = f"t{uuid.uuid4().hex[:16]}"
    admin = create_engine(POSTGRES_URL)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    separator = "&" if "?" in POSTGRES_URL else "?"
    try:
        # `search_path` scopes every unqualified name to this test's schema, so
        # the platform code needs no awareness of the arrangement at all.
        yield f"{POSTGRES_URL}{separator}options=-csearch_path%3D{schema}"
    finally:
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture(autouse=True)
def isolated_platform(database_url, monkeypatch):
    """Point the whole platform at throwaway infrastructure for one test."""
    monkeypatch.setenv("CWAP_DATABASE_URL", database_url)
    monkeypatch.setenv("CWAP_BROKER", "memory")
    monkeypatch.setenv("CWAP_INLINE_WORKER", "false")
    monkeypatch.setenv("CWAP_JWT_SECRET", "test-secret-not-used-anywhere-real")
    monkeypatch.setenv("CWAP_LLM_PROVIDER", "stub")
    monkeypatch.setenv("CWAP_EMBEDDING_PROVIDER", "hashing")
    monkeypatch.delenv("CWAP_HTTP_ALLOWLIST", raising=False)
    monkeypatch.delenv("CWAP_DEMO_PASSWORD", raising=False)
    reset_settings_cache()

    db.configure(database_url)
    db.init_db()

    set_broker(InMemoryBroker())
    reset_provider_cache(StubProvider("stub-model"))
    set_embedder(HashingEmbedder())

    # The log bus caches per-run sequence counters in process memory.
    log_bus._sequences.clear()  # noqa: SLF001 - test isolation
    log_bus._subscribers.clear()  # noqa: SLF001

    yield

    db.dispose()
    set_broker(None)
    reset_provider_cache(None)
    set_embedder(None)
    reset_settings_cache()


@pytest.fixture
def broker() -> InMemoryBroker:
    from cwap_common.broker import get_broker

    return get_broker()  # type: ignore[return-value]


@pytest.fixture
def job_context() -> JobContext:
    return JobContext(
        tenant_id="tenant-a",
        initiating_user_id="usr_test",
        permissions=PermissionRequirement(required_scope="READ_WORKFLOWS"),
        trace_id="tr_test",
    )


@pytest.fixture
def authorized_user(job_context: JobContext):
    """A real user row, so the authorisation service can actually resolve it."""
    from cwap_common.db import unit_of_work
    from cwap_common.models import User

    with unit_of_work() as session:
        session.add(
            User(
                id=job_context.initiating_user_id,
                email="test@example.com",
                tenant_id=job_context.tenant_id,
                password_hash="unused",
                scopes=["READ_WORKFLOWS"],
            )
        )
    return job_context


def make_linear_graph(*, workflow_id: str | None = None) -> WorkflowGraph:
    """input -> llm -> output, the simplest runnable workflow."""
    return WorkflowGraph(
        id=workflow_id or f"wf_{uuid.uuid4().hex[:12]}",
        name="Linear test workflow",
        nodes=[
            WorkflowNode(
                id="input",
                type=NodeType.INPUT,
                label="Goal",
                params={"fields": ["goal"]},
                position=Position(x=0, y=0),
            ),
            WorkflowNode(
                id="think",
                type=NodeType.LLM,
                label="Think",
                params={"prompt_template": "Answer this: {{goal}}"},
                position=Position(x=250, y=0),
            ),
            WorkflowNode(
                id="output",
                type=NodeType.OUTPUT,
                label="Result",
                params={"result_template": "{{previous}}"},
                position=Position(x=500, y=0),
            ),
        ],
        edges=[
            WorkflowEdge(
                id="e1", source="input", target="think", bindings={"goal": "$run.input.goal"}
            ),
            WorkflowEdge(
                id="e2", source="think", target="output", bindings={"previous": "$output.text"}
            ),
        ],
    )


def make_branching_graph() -> WorkflowGraph:
    """input -> llm -> decision -> one of two outputs."""
    return WorkflowGraph(
        id=f"wf_{uuid.uuid4().hex[:12]}",
        name="Branching test workflow",
        nodes=[
            WorkflowNode(id="input", type=NodeType.INPUT, params={"fields": ["goal"]}),
            WorkflowNode(
                id="think",
                type=NodeType.LLM,
                params={"prompt_template": "{{goal}}"},
            ),
            WorkflowNode(
                id="decide",
                type=NodeType.BRANCH,
                params={"left": "$output.text", "operator": "contains", "right": "denver"},
            ),
            WorkflowNode(
                id="output", type=NodeType.OUTPUT, params={"result_template": "matched"}
            ),
            WorkflowNode(
                id="output_alt",
                type=NodeType.OUTPUT,
                params={"result_template": "not matched"},
            ),
        ],
        edges=[
            WorkflowEdge(
                id="e1", source="input", target="think", bindings={"goal": "$run.input.goal"}
            ),
            WorkflowEdge(id="e2", source="think", target="decide"),
            WorkflowEdge(id="e3", source="decide", target="output", condition=True),
            WorkflowEdge(id="e4", source="decide", target="output_alt", condition=False),
        ],
    )


@pytest.fixture
def client():
    """The gateway, wired to this test's isolated platform.

    `create_app` runs the lifespan, which would start the inline worker; the
    fixture above disables it so tests drive the worker explicitly and stay
    deterministic.
    """
    from api_gateway.app import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def auth(client) -> dict[str, str]:
    """Authorization header for a freshly registered account."""
    response = client.post(
        "/api/auth/register",
        json={
            "email": "fixture@example.com",
            "password": "a-sufficiently-long-password",
            "tenant_id": "tenant-a",
        },
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
