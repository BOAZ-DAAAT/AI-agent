from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.checkpoint import DEFAULT_CHECKPOINT_PATH, open_sqlite_checkpointer


def test_open_sqlite_checkpointer_creates_parent_directory(tmp_path) -> None:
    db_path = tmp_path / "checkpoints" / "supervisor.sqlite"

    with open_sqlite_checkpointer(db_path) as saver:
        assert saver is not None

    assert db_path.exists()


def test_default_checkpoint_path_is_absolute_and_cwd_independent(monkeypatch, tmp_path) -> None:
    original_path = DEFAULT_CHECKPOINT_PATH

    monkeypatch.chdir(tmp_path)

    assert DEFAULT_CHECKPOINT_PATH == original_path
    assert DEFAULT_CHECKPOINT_PATH.is_absolute()
    assert DEFAULT_CHECKPOINT_PATH.name == "supervisor.sqlite"
    assert DEFAULT_CHECKPOINT_PATH.parent.name == "checkpoints"
