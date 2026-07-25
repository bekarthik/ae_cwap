"""Engine, session factory and the explicit transactional boundary.

Mandate §3.C: every sequence of writes belonging to one node execution runs
inside a single `BEGIN ... COMMIT`. `unit_of_work()` is the only sanctioned way
to open one, so "did this code path remember to be atomic?" is answerable by
grep rather than by reading every worker.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from cwap_common.models import Base
from cwap_common.settings import get_settings

logger = logging.getLogger("cwap.db")

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _build_engine(database_url: str, echo: bool) -> Engine:
    kwargs: dict[str, Any] = {"echo": echo, "future": True}
    if database_url.startswith("sqlite"):
        # SQLite defaults break under a threaded ASGI server, and its legacy
        # transaction handling silently opens/commits behind our backs. Both are
        # incompatible with an explicit unit-of-work discipline.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True

    engine = create_engine(database_url, **kwargs)

    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = _build_engine(settings.database_url, settings.sql_echo)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _session_factory


#: Fixed application id for the schema-creation advisory lock. Arbitrary, but
#: every process running `init_db` must contend on the same key.
SCHEMA_LOCK_KEY = 863_105_201

#: Fragments Postgres and SQLite use when an object already exists. `create_all`
#: does check-then-create, which is not atomic, so a concurrent creator can win
#: between the check and the CREATE.
_ALREADY_EXISTS_MARKERS = (
    "already exists",
    "duplicate key value violates unique constraint",
    "pg_type_typname_nsp_index",
)


def _is_already_exists(error: BaseException) -> bool:
    return any(marker in str(error).lower() for marker in _ALREADY_EXISTS_MARKERS)


def init_db(*, attempts: int = 3) -> None:
    """Create tables, safely when several processes start at once.

    `Base.metadata.create_all` is check-then-create, so N containers booting
    together (a gateway and two worker replicas, say) will race: each sees the
    table missing, each issues `CREATE TABLE`, and all but one fail. On
    PostgreSQL the failure is a confusing `pg_type_typname_nsp_index` unique
    violation rather than a plain "already exists".

    Two defences, because the first is not available everywhere:

    * On PostgreSQL, take a transaction-scoped advisory lock so exactly one
      process creates the schema while the others wait and then find it present.
    * Everywhere, treat an "already exists" failure as success — another process
      did the work. Anything else propagates.

    Production should run migrations instead; this keeps dev and test one call.
    """
    engine = get_engine()

    for attempt in range(1, attempts + 1):
        try:
            if engine.dialect.name == "postgresql":
                with engine.begin() as connection:
                    connection.execute(
                        text("SELECT pg_advisory_xact_lock(:key)"), {"key": SCHEMA_LOCK_KEY}
                    )
                    Base.metadata.create_all(connection)
            else:
                Base.metadata.create_all(engine)
            return
        except (IntegrityError, OperationalError, ProgrammingError) as exc:
            if not _is_already_exists(exc):
                raise
            if attempt == attempts:
                # Someone else created it; that is the outcome we wanted.
                logger.info("schema already created by another process")
                return
            logger.debug("schema creation raced with another process; retrying")


def configure(database_url: str, *, echo: bool = False) -> None:
    """Point the process at a different database (used by the test fixtures)."""
    global _engine, _session_factory
    dispose()
    _engine = _build_engine(database_url, echo)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def dispose() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


@contextmanager
def unit_of_work() -> Iterator[Session]:
    """An explicit ACID boundary.

    Commits on clean exit; rolls back *everything* written inside the block on
    any exception, so a node that fails halfway leaves no partial state and no
    orphaned log rows behind.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def read_only_session() -> Iterator[Session]:
    """For queries. Never commits, so it cannot accidentally persist anything.

    Closes without an explicit rollback on purpose: `close()` already releases
    the transaction, but unlike `rollback()` it leaves loaded attributes on the
    now-detached instances. Callers can therefore read a row's columns after the
    block without tripping `DetachedInstanceError`.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
