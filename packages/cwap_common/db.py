"""Engine, session factory and the explicit transactional boundary.

Mandate §3.C: every sequence of writes belonging to one node execution runs
inside a single `BEGIN ... COMMIT`. `unit_of_work()` is the only sanctioned way
to open one, so "did this code path remember to be atomic?" is answerable by
grep rather than by reading every worker.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from cwap_common.models import Base
from cwap_common.settings import get_settings

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


def init_db() -> None:
    """Create tables. Production uses migrations; this keeps dev/test one call."""
    Base.metadata.create_all(get_engine())


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
