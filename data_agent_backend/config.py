from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    base_data_dir: Path = Field(default=Path(".data_agent"))
    sqlite_path: Path | None = None
    default_sql_row_limit: int = 1000
    max_sql_row_limit_without_approval: int = 10_000
    default_execution_timeout_ms: int = 30_000
    sandbox_enabled: bool = False
    sandbox_image: str = "python:3.11-slim"
    sandbox_memory: str = "512m"
    sandbox_cpus: float = 1.0
    sandbox_pids_limit: int = 128
    sandbox_tmpfs_size: str = "64m"
    sandbox_keep_run_dirs: bool = True
    network_enabled: bool = False
    package_install_policy: str = "blocked"
    artifact_id_strategy: str = "uuid4"

    @property
    def workspace_dir(self) -> Path:
        return self.base_data_dir / "workspace"

    @property
    def artifact_dir(self) -> Path:
        return self.base_data_dir / "artifacts"

    @property
    def catalog_dir(self) -> Path:
        return self.base_data_dir / "catalog"

    @property
    def skills_dir(self) -> Path:
        return self.base_data_dir / "skills"

    @property
    def exports_dir(self) -> Path:
        return self.base_data_dir / "exports"

    @property
    def sandbox_dir(self) -> Path:
        return self.base_data_dir / "sandbox"

    @property
    def db_path(self) -> Path:
        return self.sqlite_path or self.base_data_dir / "backend.sqlite"

    def ensure_dirs(self) -> None:
        for path in (
            self.base_data_dir,
            self.workspace_dir,
            self.artifact_dir,
            self.catalog_dir,
            self.skills_dir,
            self.exports_dir,
            self.sandbox_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
