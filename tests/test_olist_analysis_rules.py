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

## clarify_when
{section_body}

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

## search_aliases
- {query_type}
- {query_type} Korean alias

## when_to_use
- [prefer] Use this rule for {query_type} questions.

## when_not_to_use
- [prefer] Use another rule when {query_type} is not the main intent.

## metric_definitions
- [default] {query_type} default metric uses `orders.order_id`.

## grain_guidance
- [must] {query_type} grain is grounded in `orders.order_id`.

## table_and_join_guidance
- [must] Join `orders.customer_id` to `customers.customer_id` when customer context is needed.
- [avoid] Do not use `orders.missing_column` as a real schema column.

## default_assumptions
- [default] Use the observed dataset when no period is supplied.

## soft_guidance
- [prefer] State assumptions in the final answer.

## clarification_triggers
- [ask_if_missing] Ask only when the missing choice changes the metric meaning.

## related_rule_types
- sales_orders
- purchase_frequency
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


def test_validate_schema_mismatches_are_warnings_not_blocking(rule_dir: Path) -> None:
    docs = rules.validate_rule_directory(rule_dir)

    delivery = next(doc for doc in docs if doc.front_matter["query_type"] == "delivery_delay")
    assert any("unknown referenced column" in warning for warning in delivery.warnings)


def test_build_jsonl_is_stable_section_rule_and_example_records(rule_dir: Path, tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    records = rules.build_jsonl(rule_dir, first)
    rules.build_jsonl(rule_dir, second)

    assert len(records) > 9
    assert first.read_bytes() == second.read_bytes()
    lines = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert all(line["doc_type"] == "analysis_query_rule" for line in lines)
    assert {"section_chunk", "rule_atom", "example"} <= {line["record_type"] for line in lines}
    assert any(line["rule_strength"] == "must" for line in lines)
    assert any("schema_warnings" in line for line in lines)
    assert all("PINECONE_API_KEY" not in json.dumps(line) for line in lines)


def test_record_metadata_scopes_referenced_columns_to_its_own_chunk(rule_dir: Path) -> None:
    records = rules.build_records(rule_dir)

    metric_atom = next(
        record
        for record in records
        if record["_id"] == "sales_orders__rule__default__metric_definitions__001"
    )
    join_atom = next(
        record
        for record in records
        if record["_id"] == "sales_orders__rule__must__table_and_join_guidance__001"
    )

    assert metric_atom["referenced_columns"] == ["orders.order_id"]
    assert join_atom["referenced_columns"] == ["customers.customer_id", "orders.customer_id"]


def test_record_id_segment_is_ascii_for_korean_section_names() -> None:
    segment = rules._record_id_segment("조작적 정의")

    assert segment.startswith("section-")
    assert segment.isascii()


def test_ingest_uses_stable_record_ids_and_namespace(rule_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "rules.jsonl"
    rules.build_jsonl(rule_dir, output)

    calls: list[dict] = []

    class FakeIndex:
        def upsert_records(self, **kwargs):
            calls.append(kwargs)
            return {"upserted_count": len(kwargs["records"])}

    result = rules.ingest_jsonl(output, index=FakeIndex(), namespace="test-rules", text_field="content")

    assert result == {"upserted_count": len(calls[0]["records"])}
    assert calls[0]["namespace"] == "test-rules"
    assert all("content" in record and "text" not in record for record in calls[0]["records"])


def test_search_smoke_uses_analysis_rule_filter_and_top_k_twelve(monkeypatch) -> None:
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
            "top_k": 12,
            "metadata_filter": {"doc_type": {"$in": ["analysis_query_rule", "analysis_foundation", "analysis_integrity_caution"]}},
            "settings": rules.PineconeRuntimeSettings(namespace="test-rules"),
        }
    ]
