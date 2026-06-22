from __future__ import annotations

from typing import Any

from .common import BackendError, BackendModel, JsonDict, to_jsonable


class ToolError(BackendModel):
    code: str
    message: str
    details: JsonDict = {}


class ToolResult(BackendModel):
    ok: bool
    data: Any | None = None
    error: ToolError | None = None

    @classmethod
    def success(cls, data: Any | None = None) -> "ToolResult":
        return cls(ok=True, data=to_jsonable(data), error=None)

    @classmethod
    def failure(cls, code: str, message: str, details: JsonDict | None = None) -> "ToolResult":
        return cls(ok=False, data=None, error=ToolError(code=code, message=message, details=details or {}))

    @classmethod
    def from_exception(cls, exc: Exception) -> "ToolResult":
        if isinstance(exc, BackendError):
            return cls.failure(exc.code, exc.message, exc.details)
        return cls.failure("INTERNAL_ERROR", "An internal backend error occurred.", {"type": type(exc).__name__})

