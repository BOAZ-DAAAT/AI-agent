"""Shared Pinecone connection and company-context search helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Iterable, Mapping

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
    "query_type",
    "record_type",
    "section",
    "rule_strength",
    "schema_warnings",
    "integrity_cautions",
    "referenced_columns",
    "rule_id",
    "rule",
    "rule_name",
    "rule_metadata",
    "source_text",
)
MAX_UPSERT_RECORDS_PER_BATCH = 96


class PineconeConfigurationError(ValueError):
    """Raised when required Pinecone environment settings are invalid."""


class PineconeSearchError(RuntimeError):
    """Raised when Pinecone rejects or cannot complete a context search."""


class PineconeUpsertError(RuntimeError):
    """Raised when Pinecone rejects or cannot complete a context upsert."""


class PineconeRerankError(RuntimeError):
    """Pinecone reranker 호출이 실패했을 때 발생합니다."""


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str = field(repr=False)
    index_name: str = "olist"
    namespace: str = "olist-rag-v1"
    text_field: str = "text"
    top_k: int = 5
    timeout_seconds: float = 10.0
    rerank_enabled: bool = True
    rerank_model: str = "bge-reranker-v2-m3"

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
        if self.rerank_enabled and not self.rerank_model.strip():
            raise PineconeConfigurationError("PINECONE_RERANK_MODEL은 비어 있을 수 없습니다.")

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
            rerank_enabled=_environment_boolean("PINECONE_RERANK_ENABLED", True),
            rerank_model=os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3"),
        )


@dataclass(frozen=True)
class CompanyContextHit:
    record_id: str
    score: float
    text: str
    document_id: str
    title: str
    metadata: dict[str, Any]
    rerank_score: float | None = None


CompanyContextRecord = dict[str, Any]


@lru_cache(maxsize=4)
def _get_cached_index(api_key: str, index_name: str) -> Any:
    return Pinecone(api_key=api_key).Index(index_name)


def get_pinecone_index(settings: PineconeSettings | None = None) -> Any:
    """Return a cached Pinecone index for the configured account and index."""

    resolved = settings or PineconeSettings.from_env()
    return _get_cached_index(resolved.api_key, resolved.index_name)


def rerank_company_context(
    query: str,
    hits: Iterable[CompanyContextHit],
    *,
    settings: PineconeSettings | None = None,
    client: Any | None = None,
) -> list[CompanyContextHit]:
    """dense 검색 후보를 Pinecone 다국어 reranker로 한 번에 재정렬합니다."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Pinecone rerank 검색어는 비어 있을 수 없습니다.")

    candidates = _unique_hits_by_record_id(hits)
    if not candidates:
        return []

    resolved = settings or PineconeSettings.from_env()
    if not resolved.rerank_enabled:
        return candidates

    inference_client = client or Pinecone(api_key=resolved.api_key)
    documents = [
        {"id": hit.record_id, "text": hit.text}
        for hit in candidates
    ]
    try:
        response = inference_client.inference.rerank(
            model=resolved.rerank_model,
            query=normalized_query,
            documents=documents,
            rank_fields=["text"],
            return_documents=False,
            top_n=len(documents),
        )
        reranked = _map_rerank_results(response, candidates)
    except Exception as exc:
        raise PineconeRerankError(f"Pinecone 문맥 rerank에 실패했습니다: {exc}") from exc
    return reranked


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


def upsert_company_context(
    records: Iterable[Mapping[str, Any]],
    *,
    settings: PineconeSettings | None = None,
    index: Any | None = None,
) -> Any:
    """Upsert JSON-serializable company-context records using integrated embedding.

    Each record is normalized to Pinecone's integrated-embedding record shape:
    ``{"_id": stable_id, settings.text_field: text, **metadata}``.
    """

    resolved = settings or PineconeSettings.from_env()
    normalized_records = [
        _normalize_upsert_record(record, resolved.text_field) for record in records
    ]
    if not normalized_records:
        raise ValueError("Pinecone 업서트 레코드는 1개 이상이어야 합니다.")

    upsert_index = index or get_pinecone_index(resolved)
    try:
        responses = [
            upsert_index.upsert_records(namespace=resolved.namespace, records=batch)
            for batch in _batches(normalized_records, MAX_UPSERT_RECORDS_PER_BATCH)
        ]
        if len(responses) == 1:
            return responses[0]
        return {
            "record_count": len(normalized_records),
            "batch_count": len(responses),
            "responses": responses,
        }
    except Exception as exc:
        raise PineconeUpsertError(f"Pinecone 문맥 업서트에 실패했습니다: {exc}") from exc


def _response_hits(response: Any) -> list[Any]:
    result = _read_value(response, "result", {})
    hits = _read_value(result, "hits", [])
    return list(hits or [])


def _batches(records: list[CompanyContextRecord], size: int) -> Iterable[list[CompanyContextRecord]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


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


def _unique_hits_by_record_id(
    hits: Iterable[CompanyContextHit],
) -> list[CompanyContextHit]:
    unique: dict[str, CompanyContextHit] = {}
    for hit in hits:
        key = hit.record_id or f"{hit.document_id}:{len(unique)}"
        current = unique.get(key)
        if current is None or hit.score > current.score:
            unique[key] = hit
    return sorted(unique.values(), key=lambda hit: hit.score, reverse=True)


def _map_rerank_results(
    response: Any,
    candidates: list[CompanyContextHit],
) -> list[CompanyContextHit]:
    data = list(_read_value(response, "data", []) or [])
    mapped: list[CompanyContextHit] = []
    used_indexes: set[int] = set()
    for item in data:
        index = int(_read_value(item, "index", -1))
        if index < 0 or index >= len(candidates) or index in used_indexes:
            continue
        score = float(_read_value(item, "score", 0.0) or 0.0)
        mapped.append(replace(candidates[index], rerank_score=score))
        used_indexes.add(index)
    mapped.extend(
        candidate
        for index, candidate in enumerate(candidates)
        if index not in used_indexes
    )
    return mapped


def _normalize_upsert_record(record: Mapping[str, Any], text_field: str) -> CompanyContextRecord:
    raw = _as_dict(record)
    if not raw:
        raise ValueError("Pinecone 업서트 레코드는 비어 있을 수 없습니다.")

    metadata = _as_dict(raw.get("metadata", {}))
    for key, value in raw.items():
        if key in {"_id", "id", "record_id", "metadata", text_field}:
            continue
        if key == "text" and text_field != "text":
            continue
        metadata[key] = value

    record_text = raw.get(text_field)
    if record_text is None and text_field != "text":
        record_text = raw.get("text")
    normalized_text = str(record_text or "").strip()
    if not normalized_text:
        raise ValueError("Pinecone 업서트 레코드 text는 비어 있을 수 없습니다.")

    record_id = (
        raw.get("_id")
        or raw.get("record_id")
        or raw.get("id")
        or _stable_record_id(metadata)
    )
    if not str(record_id or "").strip():
        raise ValueError("Pinecone 업서트 레코드 _id를 만들 수 없습니다.")

    normalized: CompanyContextRecord = {
        "_id": str(record_id).strip(),
        text_field: normalized_text,
    }
    safe_metadata = _json_serializable_dict(metadata)
    safe_metadata.pop("_id", None)
    safe_metadata.pop(text_field, None)
    normalized.update(safe_metadata)
    return normalized


def _stable_record_id(metadata: Mapping[str, Any]) -> str:
    document_id = str(metadata.get("document_id") or "").strip()
    if not document_id:
        return ""

    chunk_index = metadata.get("chunk_index")
    if chunk_index is None:
        return document_id

    try:
        return f"{document_id}__chunk_{int(chunk_index):03d}"
    except (TypeError, ValueError):
        return f"{document_id}__chunk_{str(chunk_index).strip()}"


def _json_serializable_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Pinecone 업서트 metadata는 JSON 직렬화 가능해야 합니다.") from exc


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


def _environment_boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise PineconeConfigurationError(f"{name}은 true 또는 false여야 합니다.")
