from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.mysql.routes import router as mysql_router
from backend.storage.routes import router as storage_router
from data_agent_backend.api.routes_integrity import router as integrity_router
from data_agent_backend.services.factory import create_backend_services


@asynccontextmanager
async def _lifespan(app: FastAPI):
    async def _run() -> None:
        while True:
            try:
                app.state.services.integrity_service.process_next_pending()
            except Exception:
                pass
            await asyncio.sleep(0.25)

    app.state.integrity_worker_task = asyncio.create_task(_run())
    try:
        yield
    finally:
        task = getattr(app.state, "integrity_worker_task", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def create_app() -> FastAPI:
    app = FastAPI(title="DAAAT Backend API", lifespan=_lifespan)
    app.state.services = create_backend_services()
    app.state.integrity_worker_task = None

    origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    allow_origins = ["*"] if origins == "*" else [o.strip() for o in origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    app.include_router(mysql_router)
    app.include_router(storage_router)
    app.include_router(integrity_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
