"""Shared Pinecone connection and company-context search helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

from pinecone import Pinecone

# Load the repository .env before reading Pinecone settings.
import DATA_Analyst_Assistant_Agent.shared.config  # noqa: F401


DEFAULT_RETURN_FIELDS = (
    "document_id",
    "title",
    "doc_type",
    "department",
    "journey_stage",
    "business_entities",
    "source_tables",
    "grounding_level",
    "intended_use",
    "prohibited_use",
    "language",
    "version",
    "source_path",
    "chunk_index",
    "chunk_count",
)


class PineconeConfigurationError(ValueError):
    """Raised when required Pinecone environment settings are invalid."""


class PineconeSearchError(RuntimeError):
    """Raised when Pinecone rejects or cannot complete a context search."""


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str = field(repr=False)
    index_name: str = "olist"
    namespace: str = "olist-rag-v1"
    text_field: str = "text"
    top_k: int = 5
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise PineconeConfigurationError("PINECONE_API_KEY가 설정되지 않았습니다.")
        for env_name, value in (
            ("PINECONE_INDEX_NAME", self.index_name),
            ("PINECONE_NAMESPACE", self.namespace),
            ("PINECONE_TEXT_FIELD", self.text_field),
        ):
            if not value.strip():
                raise PineconeConfigurationError(f"{env_name}은 비어 있을 수 없습니다.")
        if self.top_k < 1:
            raise PineconeConfigurationError("PINECONE_TOP_K는 1 이상이어야 합니다.")
        if self.timeout_seconds <= 0:
            raise PineconeConfigurationError("PINECONE_TIMEOUT_SECONDS는 0보다 커야 합니다.")

    @classmethod
    def from_env(cls) -> "PineconeSettings":
        try:
            top_k = int(os.getenv("PINECONE_TOP_K", "5"))
            timeout_seconds = float(os.getenv("PINECONE_TIMEOUT_SECONDS", "10"))
        except ValueError as exc:
            raise PineconeConfigurationError(
                "PINECONE_TOP_K와 PINECONE_TIMEOUT_SECONDS는 숫자여야 합니다."
            ) from exc
        return cls(
            api_key=os.getenv("PINECONE_API_KEY", ""),
            index_name=os.getenv("PINECONE_INDEX_NAME", "olist"),
            namespace=os.getenv("PINECONE_NAMESPACE", "olist-rag-v1"),
            text_field=os.getenv("PINECONE_TEXT_FIELD", "text"),
            top_k=top_k,
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class CompanyContextHit:
    record_id: str
    score: float
    text: str
    document_id: str
    title: str
    metadata: dict[str, Any]


@lru_cache(maxsize=4)
def _get_cached_index(api_key: str, index_name: str) -> Any:
    return Pinecone(api_key=api_key).Index(index_name)


def get_pinecone_index(settings: PineconeSettings | None = None) -> Any:
    """Return a cached Pinecone index for the configured account and index."""

    resolved = settings or PineconeSettings.from_env()
    return _get_cached_index(resolved.api_key, resolved.index_name)


def search_company_context(
    query: str,
    *,
    top_k: int | None = None,
    metadata_filter: Mapping[str, Any] | None = None,
    settings: PineconeSettings | None = None,
    index: Any | None = None,
) -> list[CompanyContextHit]:
    """Search company documents using Pinecone's server-side embedding."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Pinecone 검색어는 비어 있을 수 없습니다.")

    resolved = settings or PineconeSettings.from_env()
    result_count = resolved.top_k if top_k is None else top_k
    if result_count < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    return_fields = list(dict.fromkeys((resolved.text_field, *DEFAULT_RETURN_FIELDS)))
    search_index = index or get_pinecone_index(resolved)
    try:
        response = search_index.search(
            namespace=resolved.namespace,
            inputs={"text": normalized_query},
            top_k=result_count,
            filter=metadata_filter,
            fields=return_fields,
            timeout=resolved.timeout_seconds,
        )
    except Exception as exc:
        raise PineconeSearchError(f"Pinecone 문맥 검색에 실패했습니다: {exc}") from exc

    return [_normalize_hit(hit, resolved.text_field) for hit in _response_hits(response)]


def _response_hits(response: Any) -> list[Any]:
    result = _read_value(response, "result", {})
    hits = _read_value(result, "hits", [])
    return list(hits or [])


def _normalize_hit(hit: Any, text_field: str) -> CompanyContextHit:
    fields = _as_dict(_read_value(hit, "fields", {}))
    record_id = str(_read_value(hit, "id", ""))
    return CompanyContextHit(
        record_id=record_id,
        score=float(_read_value(hit, "score", 0.0) or 0.0),
        text=str(fields.get(text_field) or ""),
        document_id=str(fields.get("document_id") or record_id),
        title=str(fields.get("title") or ""),
        metadata=fields,
    )


def _read_value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    return {}
