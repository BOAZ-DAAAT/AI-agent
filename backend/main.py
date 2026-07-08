from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.mysql.routes import router as mysql_router
from backend.storage.routes import router as storage_router

def create_app() -> FastAPI:
    app = FastAPI(title="DAAAT Backend API")

    #지금은 일단 개발을 위해 CORS 열어둠 추후에 프론트엔드 주소로 지정
    origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    allow_origins = ["*"] if origins == "*" else [o.strip() for o in origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 서버 생존 확인용
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    app.include_router(mysql_router)
    app.include_router(storage_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

