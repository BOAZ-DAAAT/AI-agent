from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.analysis.result_contract import (
    fatal_result_contract_errors,
    normalize_result_payload,
)


def test_normalize_evidence_table_name_to_title() -> None:
    payload = {
        "evidence_tables": [
            {"name": "risk_segment_summary", "columns": ["seg"], "rows": []}
        ]
    }

    normalized = normalize_result_payload(payload)

    assert normalized["evidence_tables"][0]["title"] == "risk_segment_summary"
    assert "Normalized evidence_tables[0].name to title." in normalized["method_notes"]


def test_normalize_integer_valued_float_sample_size() -> None:
    payload = {"hypothesis_tests": [{"hypothesis": "x", "test_name": "t", "n": 87448.0}]}

    normalized = normalize_result_payload(payload)

    assert normalized["hypothesis_tests"][0]["n"] == 87448
    assert "Normalized hypothesis_tests[0].n from integer-valued float." in normalized["method_notes"]


def test_clear_non_integer_float_sample_size_without_rounding() -> None:
    payload = {"hypothesis_tests": [{"hypothesis": "x", "test_name": "spearman", "n": 87448.08854062065}]}

    normalized = normalize_result_payload(payload)

    assert normalized["hypothesis_tests"][0]["n"] is None
    assert (
        "Cleared hypothesis_tests[0].n because sample size was a non-integer float."
        in normalized["method_notes"]
    )


def test_normalize_hypothesis_caveats_string_to_list() -> None:
    payload = {
        "hypothesis_tests": [
            {
                "hypothesis": "seller averages differ",
                "test_name": "spearman",
                "caveats": "Aggregated seller averages can hide within-seller variance.",
            }
        ]
    }

    normalized = normalize_result_payload(payload)

    assert normalized["hypothesis_tests"][0]["caveats"] == [
        "Aggregated seller averages can hide within-seller variance."
    ]
    assert (
        "Normalized hypothesis_tests[0].caveats from string to list."
        in normalized["method_notes"]
    )


def test_recoverable_alias_is_not_fatal_but_broken_rows_are() -> None:
    assert fatal_result_contract_errors({"evidence_tables": [{"name": "summary", "rows": []}]}) == []

    errors = fatal_result_contract_errors({"evidence_tables": [{"title": "summary", "rows": "bad"}]})

    assert errors == ["result['evidence_tables'][0]['rows'] must be a list of row dicts."]
