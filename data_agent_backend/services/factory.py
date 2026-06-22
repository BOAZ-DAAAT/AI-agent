from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv

from data_agent_backend.config import BackendConfig
from data_agent_backend.services.approval_store import ApprovalStore
from data_agent_backend.services.artifact_registry import ArtifactRegistry
from data_agent_backend.services.artifact_store import ArtifactStore
from data_agent_backend.services.catalog_store import CatalogStore
from data_agent_backend.services.checkpoint_manager import CheckpointManager
from data_agent_backend.services.export_service import ExportService
from data_agent_backend.services.memory_store import MemoryStore
from data_agent_backend.services.policy_engine import PolicyEngine
from data_agent_backend.services.run_service import RunService
from data_agent_backend.services.sandbox_executor import DisabledSandboxExecutor, DockerSandboxExecutor, SandboxExecutor
from data_agent_backend.services.datasource_service import DatasourceService
from data_agent_backend.services.sql_executor import SQLExecutor
from data_agent_backend.services.workspace_backend import WorkspaceBackend
from data_agent_backend.services.workspace_router import ArtifactMount, LocalDirectoryMount, MemoryMount, ReadOnlyPolicyMount, WorkspaceRouter
from data_agent_backend.services.workspace_storage_service import WorkspaceStorageService
from data_agent_backend.storage.sqlite import SQLiteStore


@dataclass
class BackendServices:
    config: BackendConfig
    sqlite: SQLiteStore
    policy_engine: PolicyEngine
    approval_store: ApprovalStore
    artifact_store: ArtifactStore
    artifact_registry: ArtifactRegistry
    memory_store: MemoryStore
    catalog_store: CatalogStore
    workspace_backend: WorkspaceBackend
    workspace_storage: WorkspaceStorageService
    sql_executor: SQLExecutor
    sandbox_executor: SandboxExecutor
    export_service: ExportService
    checkpoint_manager: CheckpointManager
    run_service: RunService
    datasource_service: DatasourceService = field(default=None)  # type: ignore[assignment]


def create_backend_services(config: BackendConfig | None = None) -> BackendServices:
    # Load a local .env if present, while preserving explicit process env vars.
    load_dotenv(find_dotenv(usecwd=True), override=False)
    config = config or BackendConfig()
    config.ensure_dirs()
    sqlite = SQLiteStore(config.db_path)
    policy_engine = PolicyEngine(sqlite)
    approval_store = ApprovalStore(sqlite)
    artifact_store = ArtifactStore(config.artifact_dir)
    artifact_registry = ArtifactRegistry(sqlite, artifact_store, policy_engine)
    memory_store = MemoryStore(sqlite, policy_engine, approval_store)
    catalog_store = CatalogStore(config.catalog_dir, policy_engine)
    mounts = {
        "/workspace": LocalDirectoryMount("/workspace", config.workspace_dir, policy_engine, writable=True),
        "/artifacts": ArtifactMount(artifact_registry, artifact_store, policy_engine),
        "/catalog": ReadOnlyPolicyMount("/catalog", config.catalog_dir, policy_engine, writable=False),
        "/memory": MemoryMount(memory_store, policy_engine),
        "/skills": ReadOnlyPolicyMount("/skills", config.skills_dir, policy_engine, writable=False),
        "/exports": ReadOnlyPolicyMount("/exports", config.exports_dir, policy_engine, writable=False),
    }
    workspace_backend = WorkspaceBackend(WorkspaceRouter(mounts))
    workspace_storage = WorkspaceStorageService(artifact_registry, artifact_store, workspace_backend)
    sql_executor = SQLExecutor(config, artifact_registry, policy_engine)
    if config.sandbox_enabled:
        sandbox_executor = DockerSandboxExecutor(config, policy_engine, artifact_registry, artifact_store)
    else:
        sandbox_executor = DisabledSandboxExecutor(policy_engine)
    export_service = ExportService(config.exports_dir, sqlite, artifact_registry, artifact_store, policy_engine, approval_store)
    checkpoint_manager = CheckpointManager(sqlite)
    run_service = RunService(sqlite, policy_engine, artifact_registry)
    datasource_service = DatasourceService(sqlite)
    _auto_register_default_datasource(datasource_service)

    return BackendServices(
        config=config,
        sqlite=sqlite,
        policy_engine=policy_engine,
        approval_store=approval_store,
        artifact_store=artifact_store,
        artifact_registry=artifact_registry,
        memory_store=memory_store,
        catalog_store=catalog_store,
        workspace_backend=workspace_backend,
        workspace_storage=workspace_storage,
        sql_executor=sql_executor,
        sandbox_executor=sandbox_executor,
        export_service=export_service,
        checkpoint_manager=checkpoint_manager,
        run_service=run_service,
        datasource_service=datasource_service,
    )


def _auto_register_default_datasource(ds_service: DatasourceService) -> None:
    """If MYSQL_* env vars are all set, register/reuse a default MySQL datasource."""
    host = os.environ.get("MYSQL_HOST")
    database = os.environ.get("MYSQL_DATABASE")
    username = os.environ.get("MYSQL_USERNAME")
    if not (host and database and username):
        return

    from data_agent_backend.models.datasource import DatasourceCreateRequest, DatasourceType

    req = DatasourceCreateRequest(
        name=os.environ.get("MYSQL_DATASOURCE_NAME", "default_mysql"),
        type=DatasourceType.mysql,
        host=host,
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        database=database,
        username=username,
        password=os.environ.get("MYSQL_PASSWORD", ""),
    )
    ds_service.register(req)
