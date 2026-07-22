from __future__ import annotations

import json

from scripts import evaluate_olist_semantic_search as evaluation


def test_quality_fixture_contains_twenty_natural_questions() -> None:
    questions = json.loads(evaluation.DEFAULT_FIXTURE.read_text(encoding="utf-8"))

    assert len(questions) == 20
    assert len({question["id"] for question in questions}) == 20
    assert all(question["primary_document_id"] in question["required_document_ids"] for question in questions)
    assert all(question["query"].strip() for question in questions)


def test_metrics_calculate_hit_and_recall_at_five_and_ten() -> None:
    result = {
        "primary_document_id": "sales-orders",
        "required_document_ids": ["sales-orders", "table-orders"],
        "ranked_document_ids": ["sales-orders", "table-orders"],
    }

    metrics = evaluation._metrics([result])

    assert metrics == {
        "hit_at_5": 1.0,
        "recall_at_5": 1.0,
        "hit_at_10": 1.0,
        "recall_at_10": 1.0,
    }


def test_saved_baseline_records_pre_sync_namespace_metrics() -> None:
    baseline = json.loads(evaluation.DEFAULT_BASELINE.read_text(encoding="utf-8"))

    assert baseline["source_namespace"] == "olist-rag-v1"
    assert baseline["source_record_count"] == 542
    assert baseline["question_count"] == 20
    assert baseline["metrics"]["hit_at_5"] >= 0.90
    assert baseline["metrics"]["hit_at_10"] >= 0.95
    assert baseline["metrics"]["recall_at_10"] >= 0.95
