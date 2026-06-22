from __future__ import annotations

import enum
import sys
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict

# Python 3.9/3.10 compat: StrEnum was added in 3.11
if sys.version_info >= (3, 11):
    from enum import StrEnum  # noqa: F401
else:
    class StrEnum(str, enum.Enum):  # noqa: D101
        pass


JsonDict = dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackendModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class BackendError(Exception):
    def __init__(self, code: str, message: str, details: JsonDict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value

