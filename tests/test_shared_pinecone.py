from __future__ import annotations

from types import SimpleNamespace

import pytest

from DATA_Analyst_Assistant_Agent.shared.pinecone import (
    PineconeUpsertError,
    PineconeConfigurationError,
    PineconeSearchError,
    PineconeSettings,
    get_pinecone_index,
    search_company_context,
    upsert_company_context,
)


def _settings(**overrides) -> PineconeSettings:
    values = {
        "api_key": "test-key",
        "index_name": "olist",
        "namespace": "olist-rag-v1",
        "text_field": "text",
        "top_k": 5,
        "timeout_seconds": 10.0,
    }
    values.update(overrides)
    return PineconeSettings(**values)


def test_settings_load_pinecone_environment(monkeypatch) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "env-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "olist-context")
    monkeypatch.setenv("PINECONE_NAMESPACE", "company-v2")
    monkeypatch.setenv("PINECONE_TEXT_FIELD", "content")
    monkeypatch.setenv("PINECONE_TOP_K", "7")
    monkeypatch.setenv("PINECONE_TIMEOUT_SECONDS", "3.5")

    settings = PineconeSettings.from_env()

    assert settings.api_key == "env-key"
    assert settings.index_name == "olist-context"
    assert settings.namespace == "company-v2"
    assert settings.text_field == "content"
    assert settings.top_k == 7
    assert settings.timeout_seconds == 3.5


def test_settings_reject_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    with pytest.raises(PineconeConfigurationError, match="PINECONE_API_KEY"):
        PineconeSettings.from_env()


def test_get_pinecone_index_reuses_client_for_same_settings(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    index = object()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def Index(self, index_name: str):
            calls.append((self.api_key, index_name))
            return index

    import DATA_Analyst_Assistant_Agent.shared.pinecone as pinecone_module

    monkeypatch.setattr(pinecone_module, "Pinecone", FakeClient)
    pinecone_module._get_cached_index.cache_clear()

    assert get_pinecone_index(_settings()) is index
    assert get_pinecone_index(_settings()) is index
    assert calls == [("test-key", "olist")]
    pinecone_module._get_cached_index.cache_clear()


def test_upsert_company_context_uses_integrated_embedding_records() -> None:
    class FakeIndex:
        def __init__(self) -> None:
            self.kwargs = None

        def upsert_records(self, **kwargs):
            self.kwargs = kwargs
            return {"upserted_count": len(kwargs["records"])}

    index = FakeIndex()

    result = upsert_company_context(
        [
            {
                "document_id": "analysis-rule-001",
                "title": "매출 분석 규칙",
                "text": "매출 분석은 주문 상태와 취소 여부를 먼저 확인한다.",
                "query_type": "analysis_rule",
                "rule_id": "revenue-status-rule",
                "chunk_index": 0,
            }
        ],
        settings=_settings(),
        index=index,
    )

    assert result == {"upserted_count": 1}
    assert index.kwargs["namespace"] == "olist-rag-v1"
    assert index.kwargs["records"] == [
        {
            "_id": "analysis-rule-001__chunk_000",
            "text": "매출 분석은 주문 상태와 취소 여부를 먼저 확인한다.",
            "document_id": "analysis-rule-001",
            "title": "매출 분석 규칙",
            "query_type": "analysis_rule",
            "rule_id": "revenue-status-rule",
            "chunk_index": 0,
        }
    ]


def test_upsert_company_context_accepts_nested_metadata_and_custom_text_field() -> None:
    class FakeIndex:
        def __init__(self) -> None:
            self.records = None

        def upsert_records(self, **kwargs):
            self.records = kwargs["records"]
            return None

    index = FakeIndex()

    upsert_company_context(
        [
            {
                "record_id": "rule-custom-id",
                "content": "분석 규칙 본문",
                "metadata": {
                    "document_id": "analysis-rule-002",
                    "rule": {"metric": "gmv", "grain": "monthly"},
                    "enabled": True,
                },
            }
        ],
        settings=_settings(text_field="content"),
        index=index,
    )

    assert index.records == [
        {
            "_id": "rule-custom-id",
            "content": "분석 규칙 본문",
            "document_id": "analysis-rule-002",
            "rule": {"metric": "gmv", "grain": "monthly"},
            "enabled": True,
        }
    ]


def test_upsert_company_context_batches_records_at_pinecone_limit() -> None:
    class FakeIndex:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def upsert_records(self, **kwargs):
            self.calls.append(kwargs)
            return {"upserted_count": len(kwargs["records"])}

    index = FakeIndex()
    records = [{"_id": f"record-{number}", "text": "rule"} for number in range(97)]

    result = upsert_company_context(records, settings=_settings(), index=index)

    assert [len(call["records"]) for call in index.calls] == [96, 1]
    assert result["record_count"] == 97
    assert result["batch_count"] == 2


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"document_id": "doc", "chunk_index": 0, "text": "   "},
        {"text": "본문만 있고 식별자가 없음"},
    ],
)
def test_upsert_company_context_rejects_invalid_records(record: dict) -> None:
    with pytest.raises(ValueError):
        upsert_company_context([record], settings=_settings(), index=object())


def test_upsert_company_context_wraps_pinecone_errors() -> None:
    class FailingIndex:
        def upsert_records(self, **_kwargs):
            raise RuntimeError("network down")

    with pytest.raises(PineconeUpsertError, match="Pinecone 문맥 업서트에 실패") as exc_info:
        upsert_company_context(
            [{"_id": "record-1", "text": "본문"}],
            settings=_settings(),
            index=FailingIndex(),
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_search_uses_integrated_embedding_and_normalizes_hits() -> None:
    class FakeIndex:
        def __init__(self) -> None:
            self.kwargs = None

        def search(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                result=SimpleNamespace(
                    hits=[
                        SimpleNamespace(
                            id="olist-logistics-001__chunk_000",
                            score=0.87,
                            fields={
                                "text": "배송 지연은 약속 준수 관점에서 해석한다.",
                                "document_id": "olist-logistics-001",
                                "title": "Olist 물류 의사결정 원칙",
                                "department": "logistics",
                            },
                        )
                    ]
                )
            )

    index = FakeIndex()
    metadata_filter = {"department": {"$eq": "logistics"}}

    hits = search_company_context(
        "배송 지연을 분석할 때 무엇을 고려해야 해?",
        top_k=3,
        metadata_filter=metadata_filter,
        settings=_settings(),
        index=index,
    )

    assert index.kwargs["namespace"] == "olist-rag-v1"
    assert index.kwargs["inputs"] == {"text": "배송 지연을 분석할 때 무엇을 고려해야 해?"}
    assert index.kwargs["top_k"] == 3
    assert index.kwargs["filter"] == metadata_filter
    assert index.kwargs["timeout"] == 10.0
    assert "text" in index.kwargs["fields"]
    assert hits[0].record_id == "olist-logistics-001__chunk_000"
    assert hits[0].document_id == "olist-logistics-001"
    assert hits[0].title == "Olist 물류 의사결정 원칙"
    assert hits[0].text == "배송 지연은 약속 준수 관점에서 해석한다."
    assert hits[0].metadata["department"] == "logistics"


def test_search_requests_only_the_configured_text_field() -> None:
    class FakeIndex:
        def __init__(self) -> None:
            self.fields = []

        def search(self, **kwargs):
            self.fields = kwargs["fields"]
            return SimpleNamespace(result=SimpleNamespace(hits=[]))

    index = FakeIndex()

    search_company_context("배송", settings=_settings(text_field="content"), index=index)

    assert "content" in index.fields
    assert "text" not in index.fields


def test_search_preserves_analysis_rule_metadata_with_top_k_one() -> None:
    class FakeIndex:
        def __init__(self) -> None:
            self.kwargs = None

        def search(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                result=SimpleNamespace(
                    hits=[
                        {
                            "id": "analysis-rule-001__chunk_000",
                            "score": 0.91,
                            "fields": {
                                "text": "매출 분석은 취소 주문 제외 여부를 명시한다.",
                                "document_id": "analysis-rule-001",
                                "title": "매출 분석 규칙",
                                "query_type": "analysis_rule",
                                "rule_id": "revenue-status-rule",
                                "rule": {"metric": "revenue", "exclude_cancelled": True},
                            },
                        }
                    ]
                )
            )

    index = FakeIndex()

    hits = search_company_context(
        "매출 분석 규칙",
        top_k=1,
        metadata_filter={"query_type": {"$eq": "analysis_rule"}},
        settings=_settings(),
        index=index,
    )

    assert index.kwargs["top_k"] == 1
    assert "query_type" in index.kwargs["fields"]
    assert "rule_id" in index.kwargs["fields"]
    assert "rule" in index.kwargs["fields"]
    assert hits[0].metadata["query_type"] == "analysis_rule"
    assert hits[0].metadata["rule_id"] == "revenue-status-rule"
    assert hits[0].metadata["rule"] == {"metric": "revenue", "exclude_cancelled": True}


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(query: str) -> None:
    with pytest.raises(ValueError, match="검색어"):
        search_company_context(query, settings=_settings(), index=object())


def test_search_wraps_pinecone_errors() -> None:
    class FailingIndex:
        def search(self, **_kwargs):
            raise RuntimeError("network down")

    with pytest.raises(PineconeSearchError, match="Pinecone 문맥 검색에 실패") as exc_info:
        search_company_context("배송", settings=_settings(), index=FailingIndex())

    assert isinstance(exc_info.value.__cause__, RuntimeError)
