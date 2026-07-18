"""Deterministic Olist analysis-rule document validator and Pinecone record builder."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULE_DIR = REPO_ROOT / "docs" / "olist_rag_context" / "analysis_rules"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "DATA_Analyst_Assistant_Agent" / "agents" / "sql" / "data" / "db_schema.json"

EXPECTED_DOC_FILES = (
    "sales_orders.md",
    "purchase_frequency.md",
    "customer_value.md",
    "delivery_delay.md",
    "product_category.md",
    "seller_performance.md",
    "review_satisfaction.md",
    "payment_behavior.md",
    "regional_analysis.md",
)

ALLOWED_QUERY_TYPES = tuple(path.removesuffix(".md") for path in EXPECTED_DOC_FILES)

REQUIRED_FRONT_MATTER = {
    "document_id": str,
    "doc_type": str,
    "title": str,
    "language": str,
    "version": str,
    "query_type": str,
    "business_entities": list,
    "source_tables": list,
    "grounding_level": str,
    "intended_use": str,
    "prohibited_use": str,
}

REQUIRED_SECTIONS = (
    "definition",
    "supported_intents",
    "default_metrics",
    "entity_grain",
    "time_basis",
    "required_tables",
    "join_constraints",
    "status_and_null_rules",
    "clarify_when",
    "prohibited_interpretations",
    "unsupported_requests",
    "limitations",
    "positive_examples",
    "negative_examples",
)

DOC_TYPE = "analysis_query_rule"
MIN_CONSTRAINTS = 10
MIN_AMBIGUOUS_EXAMPLES = 3


class RuleValidationError(ValueError):
    """Raised when local Olist analysis-rule Markdown violates the contract."""


@dataclass(frozen=True)
class RuleDocument:
    path: Path
    front_matter: dict[str, Any]
    sections: dict[str, str]
    body: str


@dataclass(frozen=True)
class PineconeRuntimeSettings:
    """Small settings shim for CLI overrides without changing shared.pinecone.py."""

    namespace: str | None = None


@dataclass(frozen=True)
class SearchHit:
    record_id: str
    score: float
    text: str
    document_id: str
    title: str
    metadata: dict[str, Any]


def load_source_table_allowlist(schema_path: Path = DEFAULT_SCHEMA_PATH) -> set[str]:
    try:
        raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuleValidationError(f"schema file does not exist: {schema_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuleValidationError(f"schema file is not valid JSON: {schema_path}") from exc

    if not isinstance(raw_schema, Mapping):
        raise RuleValidationError("schema root must be an object keyed by table name")
    return {str(table) for table in raw_schema}


def validate_rule_directory(
    rule_dir: Path = DEFAULT_RULE_DIR,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> list[RuleDocument]:
    rule_dir = Path(rule_dir)
    if not rule_dir.exists():
        raise RuleValidationError(f"rule directory does not exist: {rule_dir}")
    if not rule_dir.is_dir():
        raise RuleValidationError(f"rule path is not a directory: {rule_dir}")

    markdown_files = sorted(path.name for path in rule_dir.glob("*.md"))
    expected_files = list(EXPECTED_DOC_FILES)
    if set(markdown_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(markdown_files))
        unexpected = sorted(set(markdown_files) - set(expected_files))
        details = []
        if missing:
            details.append(f"missing Markdown files: {missing}")
        if unexpected:
            details.append(f"unexpected Markdown files: {unexpected}")
        raise RuleValidationError(
            f"analysis rules must contain exactly {len(EXPECTED_DOC_FILES)} independent docs; "
            + "; ".join(details)
        )

    allowed_tables = load_source_table_allowlist(schema_path)
    documents = [_parse_markdown(rule_dir / filename) for filename in EXPECTED_DOC_FILES]
    _validate_documents(documents, allowed_tables)
    return documents


def build_records(rule_dir: Path = DEFAULT_RULE_DIR, *, schema_path: Path = DEFAULT_SCHEMA_PATH) -> list[dict[str, Any]]:
    documents = validate_rule_directory(rule_dir, schema_path=schema_path)
    return [_record_from_document(document) for document in documents]


def build_jsonl(
    rule_dir: Path = DEFAULT_RULE_DIR,
    output: Path | None = None,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> list[dict[str, Any]]:
    records = build_records(rule_dir, schema_path=schema_path)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for record in records]
    content = "\n".join(lines) + "\n"
    if output is None:
        sys.stdout.write(content)
    else:
        Path(output).write_text(content, encoding="utf-8")
    return records


def ingest_jsonl(
    jsonl_path: Path,
    *,
    index: Any | None = None,
    namespace: str | None = None,
    text_field: str = "text",
) -> Any:
    records = [_record_for_text_field(record, text_field) for record in _read_jsonl(jsonl_path)]
    if not records:
        raise ValueError("JSONL file has no records to ingest")

    if index is None:
        from DATA_Analyst_Assistant_Agent.shared.pinecone import PineconeSettings, get_pinecone_index

        settings = PineconeSettings.from_env()
        if namespace is not None:
            settings = PineconeSettings(
                api_key=settings.api_key,
                index_name=settings.index_name,
                namespace=namespace,
                text_field=text_field,
                top_k=settings.top_k,
                timeout_seconds=settings.timeout_seconds,
            )
        index = get_pinecone_index(settings)
        resolved_namespace = settings.namespace
    else:
        resolved_namespace = namespace or "olist-rag-v1"

    return index.upsert_records(namespace=resolved_namespace, records=records)


def search_company_context(query: str, **kwargs: Any) -> list[Any]:
    from DATA_Analyst_Assistant_Agent.shared.pinecone import PineconeSettings, search_company_context as shared_search

    settings = kwargs.pop("settings", None)
    if isinstance(settings, PineconeRuntimeSettings):
        if settings.namespace is None:
            settings = None
        else:
            base = PineconeSettings.from_env()
            settings = PineconeSettings(
                api_key=base.api_key,
                index_name=base.index_name,
                namespace=settings.namespace,
                text_field=base.text_field,
                top_k=base.top_k,
                timeout_seconds=base.timeout_seconds,
            )
    return shared_search(query, settings=settings, **kwargs)


def search_smoke(query: str, *, namespace: str | None = None) -> list[Any]:
    normalized = query.strip()
    if not normalized:
        raise ValueError("search query cannot be empty")
    return search_company_context(
        normalized,
        top_k=1,
        metadata_filter={"doc_type": {"$eq": DOC_TYPE}},
        settings=PineconeRuntimeSettings(namespace=namespace),
    )


def _parse_markdown(path: Path) -> RuleDocument:
    text = path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(text, path)
    sections = _parse_sections(body)
    return RuleDocument(path=path, front_matter=front_matter, sections=sections, body=body.strip())


def _split_front_matter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise RuleValidationError(f"{path.name}: missing YAML front matter")
    try:
        end_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise RuleValidationError(f"{path.name}: unterminated YAML front matter") from exc
    return _parse_simple_front_matter(lines[1:end_index], path), "\n".join(lines[end_index + 1 :])


def _parse_simple_front_matter(lines: Sequence[str], path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_key: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("  - "):
            if current_key is None or not isinstance(data.get(current_key), list):
                raise RuleValidationError(f"{path.name}: invalid list item in front matter")
            data[current_key].append(_parse_scalar(stripped[2:].strip()))
            continue
        if ":" not in line:
            raise RuleValidationError(f"{path.name}: invalid front matter line: {line}")
        key, value = line.split(":", 1)
        current_key = key.strip()
        raw_value = value.strip()
        if not current_key:
            raise RuleValidationError(f"{path.name}: empty front matter key")
        data[current_key] = [] if raw_value == "" else _parse_front_matter_value(raw_value)
    return data


def _parse_front_matter_value(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    return _parse_scalar(value)


def _parse_scalar(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _parse_sections(body: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in body.splitlines():
        if line.startswith("## "):
            current = _normalize_section(line[3:])
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def _normalize_section(title: str) -> str:
    return title.strip().lower().replace(" ", "_").replace("-", "_")


def _validate_documents(documents: Sequence[RuleDocument], allowed_tables: set[str]) -> None:
    seen_document_ids: set[str] = set()
    seen_query_types: set[str] = set()

    for expected_file, expected_query_type, document in zip(
        EXPECTED_DOC_FILES, ALLOWED_QUERY_TYPES, documents, strict=True
    ):
        if document.path.name != expected_file:
            raise RuleValidationError(f"{document.path.name}: expected file order/name {expected_file}")
        _validate_front_matter(document, expected_query_type, allowed_tables)
        _validate_sections(document)

        document_id = document.front_matter["document_id"]
        query_type = document.front_matter["query_type"]
        if document_id in seen_document_ids:
            raise RuleValidationError(f"{document.path.name}: duplicate document_id {document_id}")
        if query_type in seen_query_types:
            raise RuleValidationError(f"{document.path.name}: duplicate query_type {query_type}")
        seen_document_ids.add(document_id)
        seen_query_types.add(query_type)


def _validate_front_matter(document: RuleDocument, expected_query_type: str, allowed_tables: set[str]) -> None:
    front_matter = document.front_matter
    for key, expected_type in REQUIRED_FRONT_MATTER.items():
        if key not in front_matter:
            raise RuleValidationError(f"{document.path.name}: missing front matter field {key}")
        if not isinstance(front_matter[key], expected_type):
            raise RuleValidationError(f"{document.path.name}: front matter {key} must be {expected_type.__name__}")

    if front_matter["doc_type"] != DOC_TYPE:
        raise RuleValidationError(f"{document.path.name}: doc_type must be {DOC_TYPE}")
    if front_matter["language"] not in {"ko", "en"}:
        raise RuleValidationError(f"{document.path.name}: language must be ko or en")
    if front_matter["query_type"] != expected_query_type:
        raise RuleValidationError(f"{document.path.name}: query_type must be {expected_query_type}")
    expected_document_ids = {expected_query_type, expected_query_type.replace("_", "-")}
    if front_matter["document_id"] not in expected_document_ids:
        raise RuleValidationError(
            f"{document.path.name}: document_id must match query_type slug"
        )
    if not front_matter["source_tables"]:
        raise RuleValidationError(f"{document.path.name}: source_tables cannot be empty")
    if not front_matter["business_entities"]:
        raise RuleValidationError(f"{document.path.name}: business_entities cannot be empty")

    unknown_tables = sorted(set(front_matter["source_tables"]) - allowed_tables)
    if unknown_tables:
        raise RuleValidationError(f"{document.path.name}: unknown source_tables {unknown_tables}")


def _validate_sections(document: RuleDocument) -> None:
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in document.sections]
    if missing_sections:
        raise RuleValidationError(f"{document.path.name}: missing sections {missing_sections}")

    for section in REQUIRED_SECTIONS:
        if not document.sections[section].strip():
            raise RuleValidationError(f"{document.path.name}: section {section} cannot be empty")

    constraint_section_names = (
        ("constraints",)
        if "constraints" in document.sections
        else (
            "join_constraints",
            "status_and_null_rules",
            "prohibited_interpretations",
            "unsupported_requests",
            "limitations",
        )
    )
    constraints = [
        item
        for section in constraint_section_names
        for item in _bullet_items(document.sections.get(section, ""))
    ]
    if len(constraints) < MIN_CONSTRAINTS:
        raise RuleValidationError(
            f"{document.path.name}: constraints requires at least {MIN_CONSTRAINTS} bullet items"
        )

    ambiguous_section_names = (
        ("ambiguous_examples",)
        if "ambiguous_examples" in document.sections
        else ("clarify_when", "negative_examples")
    )
    ambiguous_examples = [
        item
        for section in ambiguous_section_names
        for item in _bullet_items(document.sections.get(section, ""))
    ]
    if len(ambiguous_examples) < MIN_AMBIGUOUS_EXAMPLES:
        raise RuleValidationError(
            f"{document.path.name}: ambiguous_examples requires at least {MIN_AMBIGUOUS_EXAMPLES} bullet items"
        )


def _bullet_items(text: str) -> list[str]:
    return [line.strip()[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


def _record_from_document(document: RuleDocument) -> dict[str, Any]:
    metadata = dict(sorted(document.front_matter.items()))
    resolved_path = document.path.resolve()
    relative_path = resolved_path.relative_to(REPO_ROOT) if resolved_path.is_relative_to(REPO_ROOT) else document.path
    record = {
        "_id": str(document.front_matter["document_id"]),
        "text": document.body,
        "chunk_count": 1,
        "chunk_index": 0,
        "source_path": str(relative_path),
        **metadata,
    }
    return dict(sorted(record.items()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL record") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: JSONL record must be an object")
        records.append(record)
    return records


def _record_for_text_field(record: Mapping[str, Any], text_field: str) -> dict[str, Any]:
    if not text_field.strip():
        raise ValueError("text_field cannot be empty")
    if "_id" not in record:
        raise ValueError("record is missing _id")
    if "text" not in record or not str(record["text"]).strip():
        raise ValueError(f"record {record.get('_id')} is missing non-empty text")
    normalized = dict(record)
    text = normalized.pop("text")
    normalized[text_field] = text
    return dict(sorted(normalized.items()))


def _print_validation_result(documents: Sequence[RuleDocument]) -> None:
    print(f"validated {len(documents)} Olist analysis rule documents")


def _print_search_hits(hits: Iterable[Any]) -> None:
    for hit in hits:
        if isinstance(hit, Mapping):
            payload = dict(hit)
        else:
            payload = {
                "record_id": getattr(hit, "record_id", ""),
                "document_id": getattr(hit, "document_id", ""),
                "title": getattr(hit, "title", ""),
                "score": getattr(hit, "score", 0.0),
                "metadata": getattr(hit, "metadata", {}),
            }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--rules-dir", type=Path, default=DEFAULT_RULE_DIR)
    validate_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--rules-dir", type=Path, default=DEFAULT_RULE_DIR)
    build_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    build_parser.add_argument("--output", type=Path, required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--input", type=Path, required=True)
    ingest_parser.add_argument("--namespace", default=None)
    ingest_parser.add_argument("--text-field", default="text")

    smoke_parser = subparsers.add_parser("search-smoke")
    smoke_parser.add_argument("query")
    smoke_parser.add_argument("--namespace", default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            _print_validation_result(validate_rule_directory(args.rules_dir, schema_path=args.schema))
        elif args.command == "build":
            records = build_jsonl(args.rules_dir, args.output, schema_path=args.schema)
            print(f"wrote {len(records)} records to {args.output}")
        elif args.command == "ingest":
            result = ingest_jsonl(args.input, namespace=args.namespace, text_field=args.text_field)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
        elif args.command == "search-smoke":
            _print_search_hits(search_smoke(args.query, namespace=args.namespace))
        else:
            parser.error(f"unknown command: {args.command}")
    except (RuleValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
