from __future__ import annotations

from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

from data_agent_backend.config import BackendConfig
from data_agent_backend.services.artifact_registry import ArtifactRegistry
from data_agent_backend.services.artifact_store import ArtifactStore
from data_agent_backend.services.policy_engine import PolicyEngine
from data_agent_backend.services.run_service import RunService
from data_agent_backend.services.sandbox_executor import DisabledSandboxExecutor, DockerSandboxExecutor, SandboxExecutor
from data_agent_backend.storage.sqlite import SQLiteStore


@dataclass
class CoreBackendServices:
    config: BackendConfig
    sqlite: SQLiteStore
    policy_engine: PolicyEngine
    artifact_store: ArtifactStore
    artifact_registry: ArtifactRegistry
    sandbox_executor: SandboxExecutor
    run_service: RunService


BackendServices = CoreBackendServices


def create_core_services(config: BackendConfig | None = None) -> CoreBackendServices:
    # Load a local .env if present, while preserving explicit process env vars.
    load_dotenv(find_dotenv(usecwd=True), override=False)
    config = config or BackendConfig()
    config.ensure_dirs()
    sqlite = SQLiteStore(config.db_path)
    policy_engine = PolicyEngine(sqlite)
    artifact_store = ArtifactStore(config.artifact_dir)
    artifact_registry = ArtifactRegistry(sqlite, artifact_store, policy_engine)
    if config.sandbox_enabled:
        sandbox_executor = DockerSandboxExecutor(config, policy_engine, artifact_registry, artifact_store)
    else:
        sandbox_executor = DisabledSandboxExecutor(policy_engine)
    run_service = RunService(sqlite, policy_engine, artifact_registry)

    return CoreBackendServices(
        config=config,
        sqlite=sqlite,
        policy_engine=policy_engine,
        artifact_store=artifact_store,
        artifact_registry=artifact_registry,
        sandbox_executor=sandbox_executor,
        run_service=run_service,
    )


def create_backend_services(config: BackendConfig | None = None) -> CoreBackendServices:
    """Backward-compatible factory for the default Core backend."""
    return create_core_services(config)
