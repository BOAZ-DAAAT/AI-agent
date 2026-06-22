from __future__ import annotations

from DATA_Analyst_Assistant_Agent.shared.contracts import LocalCheck


def run_analysis_self_check(result: dict) -> list[LocalCheck]:
    return [
        LocalCheck(name="method_summary_present", passed=bool(result.get("method_summary"))),
        LocalCheck(name="key_findings_present", passed=bool(result.get("key_findings"))),
        LocalCheck(name="limitations_present", passed=bool(result.get("limitations"))),
    ]
