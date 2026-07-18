from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import olist_analysis_rules as rules


EXPECTED_FILES = [
    "sales_orders.md",
    "purchase_frequency.md",
    "customer_value.md",
    "delivery_delay.md",
    "product_category.md",
    "seller_performance.md",
    "review_satisfaction.md",
    "payment_behavior.md",
    "regional_analysis.md",
]


QUERY_TYPES = [
    "sales_orders",
    "purchase_frequency",
    "customer_value",
    "delivery_delay",
    "product_category",
    "seller_performance",
    "review_satisfaction",
    "payment_behavior",
    "regional_analysis",
]


def _write_doc(root: Path, filename: str, query_type: str, *, source_tables: list[str] | None = None) -> Path:
    source_tables = source_tables or ["customers", "orders"]
    section_body = "\n".join(f"- {query_type} rule item {idx}" for idx in range(1, 4))
    constraints = "\n".join(f"- {query_type} deterministic constraint {idx}" for idx in range(1, 11))
    ambiguous = "\n".join(f"- {query_type} ambiguous query {idx}" for idx in range(1, 4))
    text = f"""---
document_id: {query_type}
doc_type: analysis_query_rule
title: {query_type} rules
language: ko
version: "1.0"
query_type: {query_type}
business_entities: [orders]
source_tables: [{", ".join(source_tables)}]
grounding_level: observed
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---

# {query_type} rules

## definition
- {query_type} definition.

## supported_intents
{section_body}

## default_metrics
{section_body}

## entity_grain
{section_body}

## time_basis
{section_body}

## required_tables
{section_body}

## join_constraints
{section_body}

## status_and_null_rules
{section_body}

## constraints
{constraints}

## clarify_when
{section_body}

## ambiguous_examples
{ambiguous}

## prohibited_interpretations
{section_body}

## unsupported_requests
{section_body}

## limitations
{section_body}

## positive_examples
{section_body}

## negative_examples
{section_body}
"""
    path = root / filename
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def rule_dir(tmp_path: Path) -> Path:
    for filename, query_type in zip(EXPECTED_FILES, QUERY_TYPES, strict=True):
        _write_doc(tmp_path, filename, query_type)
    return tmp_path


def test_validate_requires_exactly_nine_independent_rule_docs(rule_dir: Path) -> None:
    docs = rules.validate_rule_directory(rule_dir)

    assert [doc.path.name for doc in docs] == EXPECTED_FILES
    assert [doc.front_matter["query_type"] for doc in docs] == QUERY_TYPES

    (rule_dir / "common.md").write_text("---\ndocument_id: common\n---\n", encoding="utf-8")
    with pytest.raises(rules.RuleValidationError, match="unexpected Markdown files"):
        rules.validate_rule_directory(rule_dir)


def test_validate_rejects_multiple_query_types(rule_dir: Path) -> None:
    path = rule_dir / "purchase_frequency.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "query_type: purchase_frequency", "query_type: [purchase_frequency, customer_value]"
        ),
        encoding="utf-8",
    )

    with pytest.raises(rules.RuleValidationError, match="query_type"):
        rules.validate_rule_directory(rule_dir)


def test_validate_rejects_unknown_source_tables(rule_dir: Path) -> None:
    path = rule_dir / "sales_orders.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "source_tables: [customers, orders]", "source_tables: [customers, marketing_spend]"
        ),
        encoding="utf-8",
    )

    with pytest.raises(rules.RuleValidationError, match="unknown source_tables"):
        rules.validate_rule_directory(rule_dir)


def test_validate_requires_constraint_and_ambiguous_example_minimums(rule_dir: Path) -> None:
    path = rule_dir / "delivery_delay.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("- delivery_delay deterministic constraint 10\n", ""), encoding="utf-8")

    with pytest.raises(rules.RuleValidationError, match="constraints"):
        rules.validate_rule_directory(rule_dir)

    _write_doc(rule_dir, "delivery_delay.md", "delivery_delay")
    path.write_text(
        path.read_text(encoding="utf-8").replace("- delivery_delay ambiguous query 3\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(rules.RuleValidationError, match="ambiguous_examples"):
        rules.validate_rule_directory(rule_dir)


def test_build_jsonl_is_stable_one_record_per_markdown(rule_dir: Path, tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    records = rules.build_jsonl(rule_dir, first)
    rules.build_jsonl(rule_dir, second)

    assert len(records) == 9
    assert first.read_bytes() == second.read_bytes()
    lines = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert [line["_id"] for line in lines] == QUERY_TYPES
    assert all(line["doc_type"] == "analysis_query_rule" for line in lines)
    assert all(line["chunk_index"] == 0 and line["chunk_count"] == 1 for line in lines)
    assert all("PINECONE_API_KEY" not in json.dumps(line) for line in lines)


def test_ingest_uses_stable_record_ids_and_namespace(rule_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "rules.jsonl"
    rules.build_jsonl(rule_dir, output)

    calls: list[dict] = []

    class FakeIndex:
        def upsert_records(self, **kwargs):
            calls.append(kwargs)
            return {"upserted_count": len(kwargs["records"])}

    result = rules.ingest_jsonl(output, index=FakeIndex(), namespace="test-rules", text_field="content")

    assert result == {"upserted_count": 9}
    assert calls[0]["namespace"] == "test-rules"
    assert [record["_id"] for record in calls[0]["records"]] == QUERY_TYPES
    assert all("content" in record and "text" not in record for record in calls[0]["records"])


def test_search_smoke_uses_analysis_rule_filter_and_top_k_one(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_search(query: str, **kwargs):
        calls.append({"query": query, **kwargs})
        return [
            rules.SearchHit(
                record_id="purchase_frequency",
                score=0.91,
                text="rules",
                document_id="purchase_frequency",
                title="Purchase frequency",
                metadata={"query_type": "purchase_frequency"},
            )
        ]

    monkeypatch.setattr(rules, "search_company_context", fake_search)

    hits = rules.search_smoke("자주 구매하는 고객", namespace="test-rules")

    assert hits[0].document_id == "purchase_frequency"
    assert calls == [
        {
            "query": "자주 구매하는 고객",
            "top_k": 1,
            "metadata_filter": {"doc_type": {"$eq": "analysis_query_rule"}},
            "settings": rules.PineconeRuntimeSettings(namespace="test-rules"),
        }
    ]
