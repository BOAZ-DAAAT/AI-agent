from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / ".data_agent" / "checkpoints" / "supervisor.sqlite"


@contextmanager
def open_sqlite_checkpointer(path: str | Path | None = None) -> Iterator[SqliteSaver]:
    checkpoint_path = Path(path) if path else DEFAULT_CHECKPOINT_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        yield saver
