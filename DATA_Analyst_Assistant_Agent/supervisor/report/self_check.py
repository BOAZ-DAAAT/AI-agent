from __future__ import annotations

from DATA_Analyst_Assistant_Agent.shared.contracts import LocalCheck


def run_report_self_check(markdown: str) -> list[LocalCheck]:
    return [
        LocalCheck(name="summary_present", passed="## Summary" in markdown),
        LocalCheck(name="evidence_present", passed="## Evidence" in markdown),
        LocalCheck(name="limitations_present", passed="## Limitations" in markdown),
        LocalCheck(name="next_actions_present", passed="## Next Actions" in markdown),
    ]
