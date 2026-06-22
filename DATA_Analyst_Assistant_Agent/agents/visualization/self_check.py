from __future__ import annotations

from DATA_Analyst_Assistant_Agent.shared.contracts import LocalCheck


def run_visualization_self_check(config: dict) -> list[LocalCheck]:
    return [
        LocalCheck(name="chart_type_present", passed=bool(config.get("chart_type"))),
        LocalCheck(name="encoding_present", passed=bool(config.get("encoding"))),
        LocalCheck(
            name="data_reference_present",
            passed=bool(config.get("data_reference")),
            severity="warning" if not config.get("data_reference") else "info",
        ),
    ]
