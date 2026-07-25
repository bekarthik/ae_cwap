"""Concurrent schema creation.

Regression guard for a failure reported from `docker compose up`: the gateway
and two worker replicas all call `init_db()` at boot, `create_all` is
check-then-create rather than atomic, and the losers of the race crash. On
PostgreSQL the crash is an opaque

    duplicate key value violates unique constraint "pg_type_typname_nsp_index"

rather than a plain "already exists", which makes it look like data corruption
instead of a startup race.
"""

from __future__ import annotations

import threading

import pytest
from cwap_common import db
from cwap_common.db import SCHEMA_LOCK_KEY, _is_already_exists, init_db
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

#: Verbatim from the reported container log.
POSTGRES_RACE_MESSAGE = (
    '(psycopg.errors.UniqueViolation) duplicate key value violates unique constraint '
    '"pg_type_typname_nsp_index"\nDETAIL:  Key (typname, typnamespace)=(users, 2200) '
    "already exists."
)


class TestRaceDetection:
    def test_the_reported_postgres_error_is_recognised_as_a_race(self):
        assert _is_already_exists(IntegrityError(POSTGRES_RACE_MESSAGE, None, Exception()))

    def test_a_plain_already_exists_is_recognised(self):
        assert _is_already_exists(OperationalError("table users already exists", None, Exception()))

    def test_an_unrelated_failure_is_not_treated_as_a_race(self):
        assert not _is_already_exists(OperationalError("connection refused", None, Exception()))


class TestIdempotence:
    def test_calling_init_twice_is_fine(self):
        init_db()
        init_db()

    def test_an_unrelated_error_still_propagates(self, monkeypatch):
        """Swallowing "already exists" must not turn into swallowing everything."""

        def explode(*_args, **_kwargs):
            raise OperationalError("could not connect to server", None, Exception())

        monkeypatch.setattr(db.Base.metadata, "create_all", explode)
        with pytest.raises(OperationalError, match="could not connect"):
            init_db()


class TestConcurrentStart:
    def test_several_processes_booting_together_do_not_crash(self, tmp_path):
        """The actual reported scenario: N containers calling init_db at once."""
        db.configure(f"sqlite:///{tmp_path / 'race.sqlite3'}")

        failures: list[BaseException] = []
        start = threading.Barrier(8)

        def boot() -> None:
            try:
                start.wait(timeout=5)
                init_db()
            except BaseException as exc:  # noqa: BLE001 - the test is what it catches
                failures.append(exc)

        threads = [threading.Thread(target=boot) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert not failures, f"concurrent init_db raised: {failures[0]!r}"

        # And the schema is actually usable afterwards.
        with db.read_only_session() as session:
            session.execute(text("SELECT 1 FROM users")).all()


class TestPostgresLocking:
    def test_postgres_serialises_creation_with_an_advisory_lock(self, monkeypatch):
        """The lock is what makes the race impossible rather than merely
        survivable — without it, every replica still issues a doomed CREATE."""
        statements: list[str] = []

        class FakeConnection:
            def execute(self, statement, params=None):
                statements.append(str(statement))
                assert params == {"key": SCHEMA_LOCK_KEY}

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        class FakeEngine:
            dialect = type("dialect", (), {"name": "postgresql"})()

            def begin(self):
                return FakeConnection()

        created_with: list[object] = []
        monkeypatch.setattr(db, "get_engine", lambda: FakeEngine())
        monkeypatch.setattr(
            db.Base.metadata, "create_all", lambda bind, **_kw: created_with.append(bind)
        )

        init_db()

        assert any("pg_advisory_xact_lock" in sql for sql in statements)
        # create_all must run on the *locked* connection, not a fresh one.
        assert created_with and isinstance(created_with[0], FakeConnection)
