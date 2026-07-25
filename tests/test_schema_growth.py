"""A database that already holds work must survive a release that adds a column.

`create_all` only ever creates whole tables. The moment a release adds a setting
to an existing one, every database created before it starts failing on a query
naming a column the table does not have — and the fix on offer was "delete your
database", which is not a fix when it holds someone's workflows.

So one migration runs unattended at startup: adding a column that has a default
or is nullable. Nothing is renamed, retyped or dropped — those need a human.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from cwap_common import db
from sqlalchemy import Column, Integer, MetaData, String, Table, inspect, text

#: The column this release adds. The "older" table below is today's table
#: without it, which is exactly what an earlier release left behind.
NEW_COLUMN = "timeout_seconds"


@pytest.fixture
def older_database(isolated_platform):
    """A `model_settings` table as an earlier release created it, with a row.

    Built from the current mapping minus the new column rather than from
    hand-written DDL, so it is the real table on both SQLite and PostgreSQL —
    autoincrement and all — rather than an approximation that only works on one.
    """
    from cwap_common.models import ModelSetting

    engine = db.get_engine()
    older = Table(
        "model_settings",
        MetaData(),
        *[
            Column(
                column.name,
                column.type,
                primary_key=column.primary_key,
                nullable=column.nullable,
                autoincrement=column.autoincrement,
            )
            for column in ModelSetting.__table__.columns
            if column.name != NEW_COLUMN
        ],
    )

    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS model_settings"))
    older.create(engine)
    with engine.begin() as connection:
        connection.execute(
            older.insert().values(
                tenant_id="tenant-a",
                kind="llm",
                provider="ollama",
                model="llama3.1",
                base_url="",
                api_key="",
                updated_by="",
                # The copied column has no Python-side default of its own.
                updated_at=datetime.now(timezone.utc),
            )
        )
    return engine


class TestAnExistingDatabaseGrowsForward:
    def test_the_new_column_is_added(self, older_database):
        db.add_missing_columns()

        columns = {
            column["name"] for column in inspect(older_database).get_columns("model_settings")
        }
        assert NEW_COLUMN in columns

    def test_the_rows_already_there_are_kept(self, older_database):
        """The point of migrating rather than recreating."""
        db.add_missing_columns()

        with older_database.begin() as connection:
            row = connection.execute(
                text("SELECT provider, model, timeout_seconds FROM model_settings")
            ).one()

        assert row.provider == "ollama"
        assert row.model == "llama3.1"
        # The column's default, not NULL — the application reads it as an int.
        assert row.timeout_seconds == 0

    def test_the_orm_can_read_it_afterwards(self, older_database):
        from llm_proxy import store

        db.add_missing_columns()
        loaded = store.load("tenant-a")

        assert loaded is not None
        assert loaded.provider == "ollama"
        assert loaded.timeout_seconds == 0

    def test_running_it_again_changes_nothing(self, older_database):
        """Every worker boots into this; it has to be safe N times over."""
        db.add_missing_columns()
        db.add_missing_columns()

        columns = [
            column["name"] for column in inspect(older_database).get_columns("model_settings")
        ]
        assert columns.count(NEW_COLUMN) == 1

    def test_a_current_database_is_left_alone(self, isolated_platform):
        engine = db.get_engine()
        before = {
            table: [column["name"] for column in inspect(engine).get_columns(table)]
            for table in inspect(engine).get_table_names()
        }

        db.add_missing_columns()

        after = {
            table: [column["name"] for column in inspect(engine).get_columns(table)]
            for table in inspect(engine).get_table_names()
        }
        assert before == after


class TestWhatItRefusesToDo:
    def test_a_required_column_with_no_default_is_reported_not_forced(
        self, isolated_platform, caplog
    ):
        """Adding one to a populated table cannot work. Saying so beats failing
        every boot with a database error that does not name the cause."""
        table = Table(
            "growth_probe",
            db.Base.metadata,
            Column("id", Integer, primary_key=True),
            Column("existing", String(32)),
        )
        engine = db.get_engine()
        try:
            table.create(engine)
            table.append_column(Column("mandatory", String(32), nullable=False))

            with caplog.at_level("WARNING"):
                db.add_missing_columns()

            columns = {
                column["name"] for column in inspect(engine).get_columns("growth_probe")
            }
            assert "mandatory" not in columns
            assert "needs a migration" in caplog.text
        finally:
            table.drop(engine, checkfirst=True)
            db.Base.metadata.remove(table)
