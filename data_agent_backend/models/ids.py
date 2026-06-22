from __future__ import annotations

from typing import Protocol
from uuid import uuid4


class IdGenerator(Protocol):
    def new_id(self, prefix: str | None = None) -> str: ...


class UUID4IdGenerator:
    def new_id(self, prefix: str | None = None) -> str:
        value = uuid4().hex
        return f"{prefix}_{value}" if prefix else value

