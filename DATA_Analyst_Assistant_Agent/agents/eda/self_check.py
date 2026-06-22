from __future__ import annotations

from DATA_Analyst_Assistant_Agent.shared.contracts import LocalCheck


def run_eda_self_check(source_artifact_ids: list[str], profile: dict) -> list[LocalCheck]:
    return [
        LocalCheck(
            name="source_artifact_present",
            passed=bool(source_artifact_ids),
            severity="error" if not source_artifact_ids else "info",
            detail="EDA requires at least one upstream SQL artifact.",
        ),
        LocalCheck(
            name="profile_generated",
            passed=bool(profile),
            severity="error" if not profile else "info",
            detail="EDA profile payload was generated.",
        ),
        LocalCheck(
            name="columns_present",
            passed=bool(profile.get("columns")),
            severity="warning" if not profile.get("columns") else "info",
            detail="Column metadata should be available for downstream analysis.",
        ),
    ]
