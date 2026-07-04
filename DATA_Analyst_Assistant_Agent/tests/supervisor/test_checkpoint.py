from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.checkpoint import open_sqlite_checkpointer


def test_open_sqlite_checkpointer_creates_parent_directory(tmp_path) -> None:
    db_path = tmp_path / "checkpoints" / "supervisor.sqlite"

    with open_sqlite_checkpointer(db_path) as saver:
        assert saver is not None

    assert db_path.exists()
