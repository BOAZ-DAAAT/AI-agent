"""Olist semantic search 원격 품질 평가 CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DATA_Analyst_Assistant_Agent.supervisor.graph import (  # noqa: E402
    _default_analysis_rule_rerank,
    _default_analysis_rule_search,
    _retrieve_diverse_analysis_hits,
)


DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "olist_semantic_search_questions.json"
DEFAULT_BASELINE = REPO_ROOT / "tests" / "fixtures" / "olist_semantic_search_baseline.json"
QUALITY_THRESHOLDS = {
    "hit_at_5": 0.90,
    "hit_at_10": 0.95,
    "recall_at_10": 0.95,
}


def evaluate_questions(questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for question in questions:
        hits, rerank = _retrieve_diverse_analysis_hits(
            _default_analysis_rule_search,
            str(question["query"]),
            rerank=_default_analysis_rule_rerank,
        )
        ranked_document_ids = _ordered_unique(
            str(getattr(hit, "document_id", "")) for hit in hits
        )
        primary = str(question["primary_document_id"])
        required = [str(item) for item in question["required_document_ids"]]
        results.append(
            {
                "id": str(question["id"]),
                "query": str(question["query"]),
                "primary_document_id": primary,
                "required_document_ids": required,
                "ranked_document_ids": ranked_document_ids,
                "primary_rank": (
                    ranked_document_ids.index(primary) + 1
                    if primary in ranked_document_ids
                    else None
                ),
                "rerank": rerank,
            }
        )

    metrics = _metrics(results)
    return {"question_count": len(results), "metrics": metrics, "questions": results}


def _metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    count = len(results)
    required_total = sum(len(result["required_document_ids"]) for result in results)
    metrics: dict[str, float] = {}
    for k in (5, 10):
        primary_hits = 0
        required_hits = 0
        for result in results:
            ranked = list(result["ranked_document_ids"])[:k]
            primary_hits += int(result["primary_document_id"] in ranked)
            required_hits += len(set(result["required_document_ids"]) & set(ranked))
        metrics[f"hit_at_{k}"] = primary_hits / count if count else 0.0
        metrics[f"recall_at_{k}"] = required_hits / required_total if required_total else 0.0
    return metrics


def assess_quality(result: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> list[str]:
    metrics = dict(result.get("metrics") or {})
    failures = [
        f"{name}={metrics.get(name, 0.0):.4f} < {minimum:.4f}"
        for name, minimum in QUALITY_THRESHOLDS.items()
        if float(metrics.get(name, 0.0)) < minimum
    ]
    if baseline is not None:
        baseline_metrics = dict(baseline.get("metrics") or {})
        for name in ("hit_at_5", "hit_at_10", "recall_at_5", "recall_at_10"):
            current = float(metrics.get(name, 0.0))
            previous = float(baseline_metrics.get(name, 0.0))
            if current < previous:
                failures.append(f"{name}={current:.4f} < baseline={previous:.4f}")
    return failures


def _ordered_unique(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--namespace", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    questions = json.loads(args.fixture.read_text(encoding="utf-8"))
    baseline = (
        json.loads(args.baseline.read_text(encoding="utf-8"))
        if args.baseline
        else None
    )
    result = evaluate_questions(questions)
    failures = [] if args.baseline_only else assess_quality(result, baseline)
    result["quality"] = {"passed": not failures, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for question in result["questions"]:
        print(
            f"{question['id']}: primary_rank={question['primary_rank']} "
            f"documents={question['ranked_document_ids']}"
        )
    print(json.dumps(result["metrics"], ensure_ascii=False, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"실패: {failure}", file=sys.stderr)
        if args.backup is not None:
            from scripts.olist_analysis_rules import restore_namespace

            restored = restore_namespace(args.backup, namespace=args.namespace)
            print(f"백업 복원 완료: {restored}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
