from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from data_agent_backend.api.routes_artifacts import router as artifacts_router
from data_agent_backend.api.routes_datasources import router as datasources_router
from data_agent_backend.api.routes_execution import router as execution_router
from data_agent_backend.api.routes_policy import router as policy_router
from data_agent_backend.api.routes_runs import router as runs_router
from data_agent_backend.models.tool_results import ToolResult
from data_agent_backend.services.factory import BackendServices, create_core_services


def _json_result(result: ToolResult) -> JSONResponse:
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))


def create_app(services: BackendServices | None = None) -> FastAPI:
    app = FastAPI(title="Data Agent Backend API")
    app.state.services = services or create_core_services()

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request, exc: RequestValidationError) -> JSONResponse:
        errors = []
        for error in exc.errors():
            sanitized = dict(error)
            sanitized.pop("input", None)
            errors.append(sanitized)
        return _json_result(ToolResult.failure("VALIDATION_ERROR", "Request validation failed.", {"errors": errors}))

    @app.exception_handler(Exception)
    async def internal_exception_handler(_request, exc: Exception) -> JSONResponse:
        return _json_result(ToolResult.from_exception(exc))

    @app.get("/health")
    def health() -> dict:
        return ToolResult.success({"status": "ok"}).model_dump(mode="json")

    app.include_router(runs_router)
    app.include_router(artifacts_router)
    app.include_router(policy_router)
    app.include_router(datasources_router)
    app.include_router(execution_router)
    return app


def main() -> None:
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("The uvicorn package is required to run the HTTP API.") from exc

    uvicorn.run("data_agent_backend.api.app:create_app", factory=True, host="127.0.0.1", port=8000)
