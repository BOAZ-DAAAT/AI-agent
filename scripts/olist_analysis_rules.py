"""Deterministic Olist analysis-rule document validator and Pinecone record builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RULE_DIR = REPO_ROOT / "docs" / "olist_rag_context" / "analysis_rules"
DEFAULT_FOUNDATION_DIR = REPO_ROOT / "docs" / "olist_rag_context" / "analysis_foundations"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "DATA_Analyst_Assistant_Agent" / "agents" / "sql" / "data" / "db_schema.json"
DEFAULT_INTEGRITY_PATH = REPO_ROOT / "DATA_Analyst_Assistant_Agent" / "agents" / "sql" / "data" / "db_integrity_result.json"

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
FOUNDATION_DOC_FILES = (
    "metric_definitions.md",
    "entity_grain_definitions.md",
    "time_status_population_rules.md",
    "join_cardinality_rules.md",
    "table_contracts.md",
    "operational_definition_policy.md",
    "table_customers.md",
    "table_geolocation.md",
    "table_order_items.md",
    "table_order_payments.md",
    "table_order_reviews.md",
    "table_orders.md",
    "table_product_category_name_translation.md",
    "table_products.md",
    "table_sellers.md",
    "sales_order_metrics.md",
    "customer_metrics.md",
    "payment_metrics.md",
    "delivery_metrics.md",
    "review_metrics.md",
)

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

QUALITY_SECTIONS = (
    "search_aliases",
    "when_to_use",
    "when_not_to_use",
    "metric_definitions",
    "grain_guidance",
    "table_and_join_guidance",
    "default_assumptions",
    "soft_guidance",
    "clarification_triggers",
    "related_rule_types",
    "positive_examples",
    "negative_examples",
)

DOC_TYPE = "analysis_query_rule"
FOUNDATION_DOC_TYPE = "analysis_foundation"
INTEGRITY_DOC_TYPE = "analysis_integrity_caution"
RULE_STRENGTHS = ("must", "avoid", "default", "prefer", "ask_if_missing")
RULE_STRENGTH_ORDER = {name: index for index, name in enumerate(RULE_STRENGTHS)}
ISSUE_INTEGRITY_STATUSES = {"FAIL", "FAILED", "ERROR", "ACTION_REQUIRED", "STALE"}
TABLE_COLUMN_RE = re.compile(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\.`?([A-Za-z_][A-Za-z0-9_]*)`?")
RULE_TAG_RE = re.compile(r"^\[(must|avoid|default|prefer|ask_if_missing)\]\s*", re.IGNORECASE)


class RuleValidationError(ValueError):
    """Raised when local Olist analysis-rule Markdown violates the contract."""


@dataclass(frozen=True)
class RuleDocument:
    path: Path
    front_matter: dict[str, Any]
    sections: dict[str, str]
    body: str
    warnings: list[str]


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


def load_schema_catalog(schema_path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, set[str]]:
    try:
        raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuleValidationError(f"schema file does not exist: {schema_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuleValidationError(f"schema file is not valid JSON: {schema_path}") from exc

    if not isinstance(raw_schema, Mapping):
        raise RuleValidationError("schema root must be an object keyed by table name")
    tables = raw_schema.get("tables") if isinstance(raw_schema.get("tables"), Mapping) else raw_schema
    catalog: dict[str, set[str]] = {}
    for table_name, table in tables.items():
        columns = table.get("columns", []) if isinstance(table, Mapping) else []
        catalog[str(table_name)] = {
            str(column.get("name"))
            for column in columns
            if isinstance(column, Mapping) and column.get("name")
        }
    return catalog


def load_source_table_allowlist(schema_path: Path = DEFAULT_SCHEMA_PATH) -> set[str]:
    return set(load_schema_catalog(schema_path))


def validate_rule_directory(
    rule_dir: Path = DEFAULT_RULE_DIR,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    integrity_path: Path = DEFAULT_INTEGRITY_PATH,
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

    schema_catalog = load_schema_catalog(schema_path)
    integrity_statuses = _load_integrity_statuses(integrity_path)
    documents = [_parse_markdown(rule_dir / filename) for filename in EXPECTED_DOC_FILES]
    documents = _validate_documents(documents, schema_catalog, integrity_statuses)
    return documents


def build_records(
    rule_dir: Path = DEFAULT_RULE_DIR,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    integrity_path: Path = DEFAULT_INTEGRITY_PATH,
    foundation_dir: Path | None = None,
) -> list[dict[str, Any]]:
    documents = validate_rule_directory(rule_dir, schema_path=schema_path, integrity_path=integrity_path)
    resolved_rule_dir = Path(rule_dir).resolve()
    resolved_foundation_dir = foundation_dir
    if resolved_foundation_dir is None and resolved_rule_dir == DEFAULT_RULE_DIR.resolve():
        resolved_foundation_dir = DEFAULT_FOUNDATION_DIR
    if resolved_foundation_dir is not None:
        documents += validate_foundation_directory(
            Path(resolved_foundation_dir), schema_path=schema_path, integrity_path=integrity_path
        )
    records: list[dict[str, Any]] = []
    for document in documents:
        records.extend(_records_from_document(document))
    if resolved_rule_dir == DEFAULT_RULE_DIR.resolve():
        records.extend(_integrity_caution_records(integrity_path))
    return _with_chunk_counts(records)


def build_jsonl(
    rule_dir: Path = DEFAULT_RULE_DIR,
    output: Path | None = None,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    integrity_path: Path = DEFAULT_INTEGRITY_PATH,
    foundation_dir: Path | None = None,
) -> list[dict[str, Any]]:
    records = build_records(
        rule_dir,
        schema_path=schema_path,
        integrity_path=integrity_path,
        foundation_dir=foundation_dir,
    )
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
        top_k=12,
        metadata_filter={"doc_type": {"$in": [DOC_TYPE, FOUNDATION_DOC_TYPE, INTEGRITY_DOC_TYPE]}},
        settings=PineconeRuntimeSettings(namespace=namespace),
    )


def _parse_markdown(path: Path) -> RuleDocument:
    text = path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(text, path)
    sections = _parse_sections(body)
    return RuleDocument(path=path, front_matter=front_matter, sections=sections, body=body.strip(), warnings=[])


def validate_foundation_directory(
    foundation_dir: Path = DEFAULT_FOUNDATION_DIR,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    integrity_path: Path = DEFAULT_INTEGRITY_PATH,
) -> list[RuleDocument]:
    foundation_dir = Path(foundation_dir)
    filenames = sorted(path.name for path in foundation_dir.glob("*.md")) if foundation_dir.is_dir() else []
    if set(filenames) != set(FOUNDATION_DOC_FILES):
        missing = sorted(set(FOUNDATION_DOC_FILES) - set(filenames))
        unexpected = sorted(set(filenames) - set(FOUNDATION_DOC_FILES))
        raise RuleValidationError(f"foundation documents mismatch; missing={missing}; unexpected={unexpected}")

    schema_catalog = load_schema_catalog(schema_path)
    integrity_statuses = _load_integrity_statuses(integrity_path)
    documents: list[RuleDocument] = []
    for filename in FOUNDATION_DOC_FILES:
        document = _parse_markdown(foundation_dir / filename)
        _validate_foundation_front_matter(document, set(schema_catalog))
        warnings = _schema_warnings(document, schema_catalog) + _integrity_warnings(document, integrity_statuses)
        documents.append(
            RuleDocument(
                path=document.path,
                front_matter=document.front_matter,
                sections=document.sections,
                body=document.body,
                warnings=warnings,
            )
        )
    return documents


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
    normalized = title.strip().lower().replace(" ", "_").replace("-", "_")
    korean_aliases = {
        "검색_별칭": "search_aliases",
        "정의와_사용_시점": "when_to_use",
        "정의": "when_to_use",
        "지원하는_질문": "when_to_use",
        "지표_정의": "metric_definitions",
        "기본_지표": "metric_definitions",
        "grain과_조인": "grain_guidance",
        "시간·상태·기본_가정": "default_assumptions",
        "시간·결측·기본_가정": "default_assumptions",
        "상태·결측·기본_가정": "default_assumptions",
        "기본_가정과_해석": "default_assumptions",
        "기본_가정과_표현": "default_assumptions",
        "확인이_필요한_경우와_예시": "clarification_triggers",
        "확인이_필요한_경우": "clarification_triggers",
        "관련_규칙": "related_rule_types",
        "사용하지_않는_경우": "when_not_to_use",
        "테이블_및_조인_가이드": "table_and_join_guidance",
        "소프트_가이드": "soft_guidance",
        "긍정_예시": "positive_examples",
        "부정_예시": "negative_examples",
    }
    return korean_aliases.get(normalized, normalized)


def _validate_documents(
    documents: Sequence[RuleDocument],
    schema_catalog: dict[str, set[str]],
    integrity_statuses: dict[str, str],
) -> list[RuleDocument]:
    seen_document_ids: set[str] = set()
    seen_query_types: set[str] = set()
    validated: list[RuleDocument] = []

    for expected_file, expected_query_type, document in zip(
        EXPECTED_DOC_FILES, ALLOWED_QUERY_TYPES, documents, strict=True
    ):
        if document.path.name != expected_file:
            raise RuleValidationError(f"{document.path.name}: expected file order/name {expected_file}")
        warnings: list[str] = []
        _validate_front_matter(document, expected_query_type, set(schema_catalog))
        warnings.extend(_quality_warnings(document))
        warnings.extend(_schema_warnings(document, schema_catalog))
        warnings.extend(_integrity_warnings(document, integrity_statuses))

        document_id = document.front_matter["document_id"]
        query_type = document.front_matter["query_type"]
        if document_id in seen_document_ids:
            raise RuleValidationError(f"{document.path.name}: duplicate document_id {document_id}")
        if query_type in seen_query_types:
            raise RuleValidationError(f"{document.path.name}: duplicate query_type {query_type}")
        seen_document_ids.add(document_id)
        seen_query_types.add(query_type)
        validated.append(
            RuleDocument(
                path=document.path,
                front_matter=document.front_matter,
                sections=document.sections,
                body=document.body,
                warnings=warnings,
            )
        )
    return validated


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


def _validate_foundation_front_matter(document: RuleDocument, allowed_tables: set[str]) -> None:
    front_matter = document.front_matter
    required_fields = {
        "document_id": str,
        "doc_type": str,
        "query_type": str,
        "title": str,
        "language": str,
        "version": str,
        "source_tables": list,
        "grounding_level": str,
        "intended_use": str,
        "prohibited_use": str,
    }
    for key, expected_type in required_fields.items():
        if not isinstance(front_matter.get(key), expected_type):
            raise RuleValidationError(f"{document.path.name}: foundation front matter {key} must be {expected_type.__name__}")
    if front_matter["doc_type"] != FOUNDATION_DOC_TYPE:
        raise RuleValidationError(f"{document.path.name}: doc_type must be {FOUNDATION_DOC_TYPE}")
    if front_matter["language"] not in {"ko", "en"}:
        raise RuleValidationError(f"{document.path.name}: language must be ko or en")
    if not front_matter["source_tables"]:
        raise RuleValidationError(f"{document.path.name}: source_tables cannot be empty")
    unknown_tables = sorted(set(front_matter["source_tables"]) - allowed_tables)
    if unknown_tables:
        raise RuleValidationError(f"{document.path.name}: unknown source_tables {unknown_tables}")


def _bullet_items(text: str) -> list[str]:
    return [line.strip()[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


def _quality_warnings(document: RuleDocument) -> list[str]:
    warnings = [
        f"missing quality section: {section}"
        for section in QUALITY_SECTIONS
        if not document.sections.get(section, "").strip()
    ]
    tagged_rules = [
        item
        for section in document.sections.values()
        for item in _bullet_items(section)
        if RULE_TAG_RE.match(item)
    ]
    if not tagged_rules:
        warnings.append("no rule strength tags found")
    return warnings


def _schema_warnings(document: RuleDocument, schema_catalog: dict[str, set[str]]) -> list[str]:
    warnings: list[str] = []
    referenced = _referenced_columns(document.body)
    for table, column in referenced:
        if table not in schema_catalog:
            warnings.append(f"unknown referenced table in body: {table}")
        elif column not in schema_catalog[table]:
            warnings.append(f"unknown referenced column in body: {table}.{column}")

    return warnings


def _integrity_warnings(document: RuleDocument, integrity_statuses: dict[str, str]) -> list[str]:
    if not integrity_statuses:
        return ["integrity snapshot unavailable"]
    warnings: list[str] = []
    for table in document.front_matter.get("source_tables", []):
        status = integrity_statuses.get(str(table))
        if status is None:
            warnings.append(f"integrity coverage missing: {table}")
        elif status.upper() in ISSUE_INTEGRITY_STATUSES:
            warnings.append(f"integrity caution: {table} status={status}")
    return warnings


def _referenced_columns(text: str) -> list[tuple[str, str]]:
    return sorted({(match.group(1), match.group(2)) for match in TABLE_COLUMN_RE.finditer(text)})


def _load_integrity_statuses(path: Path) -> dict[str, str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    tables = data.get("tables") if isinstance(data, Mapping) else {}
    if not isinstance(tables, Mapping):
        return {}
    statuses: dict[str, str] = {}
    for table, payload in tables.items():
        if isinstance(payload, Mapping):
            statuses[str(table)] = str(payload.get("status") or "")
        elif isinstance(payload, list):
            statuses[str(table)] = "PASS"
    return statuses


def _integrity_caution_records(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    tables = data.get("tables") if isinstance(data, Mapping) else None
    if not isinstance(tables, Mapping):
        return []

    records: list[dict[str, Any]] = []
    for table, payload in sorted(tables.items()):
        if not isinstance(payload, Mapping):
            continue
        status = str(payload.get("table_status") or "").upper()
        if status not in ISSUE_INTEGRITY_STATUSES:
            continue
        failing_columns = sorted(
            {
                str(check.get("column"))
                for check in payload.get("checks", [])
                if isinstance(check, Mapping)
                and str(check.get("status") or "").upper() in ISSUE_INTEGRITY_STATUSES
                and str(check.get("column") or "") not in {"", "Table-Level"}
            }
        )
        columns_text = ", ".join(failing_columns) if failing_columns else "table-level checks"
        records.append(
            {
                "_id": f"integrity-caution__{table}",
                "text": f"Integrity caution for {table}: status={status}; affected columns: {columns_text}. Planning caution only; do not block execution.",
                "doc_type": INTEGRITY_DOC_TYPE,
                "document_id": f"integrity-caution-{table}",
                "source_document_id": f"integrity-caution-{table}",
                "title": f"Olist integrity caution: {table}",
                "query_type": "integrity_caution",
                "record_type": "integrity_caution",
                "section": "integrity_caution",
                "source_tables": [str(table)],
                "referenced_columns": [f"{table}.{column}" for column in failing_columns],
                "integrity_cautions": [f"integrity caution: {table} status={status}"],
                "schema_warnings": [],
                "grounding_level": "integrity_grounded",
                "intended_use": "SQL planning caution retrieval",
                "prohibited_use": "execution blocking or live metric evidence",
                "language": "en",
                "version": "1.0",
                "rule_strength": "prefer",
                "rule_strength_rank": RULE_STRENGTH_ORDER["prefer"],
            }
        )
    return records


def _base_metadata(document: RuleDocument) -> dict[str, Any]:
    metadata = dict(sorted(document.front_matter.items()))
    resolved_path = document.path.resolve()
    relative_path = resolved_path.relative_to(REPO_ROOT) if resolved_path.is_relative_to(REPO_ROOT) else document.path
    return {
        "source_path": str(relative_path),
        "source_document_id": str(document.front_matter["document_id"]),
        "schema_warnings": sorted(set(document.warnings)),
        "integrity_cautions": sorted({warning for warning in document.warnings if warning.startswith("integrity ")}),
        **metadata,
    }


def _records_from_document(document: RuleDocument) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    base = _base_metadata(document)
    document_id = str(document.front_matter["document_id"])

    for section_name in sorted(document.sections):
        text = document.sections[section_name].strip()
        if not text:
            continue
        section_id = _record_id_segment(section_name)
        records.append(
            _record(
                base,
                record_id=f"{document_id}__section__{section_id}",
                text=f"{section_name}\n{text}",
                record_type="section_chunk",
                section=section_name,
            )
        )
        for item_index, item in enumerate(_bullet_items(text), start=1):
            strength = _rule_strength(item)
            clean_item = RULE_TAG_RE.sub("", item).strip()
            if strength:
                records.append(
                    _record(
                        base,
                        record_id=f"{document_id}__rule__{strength}__{section_id}__{item_index:03d}",
                        text=clean_item,
                        record_type="rule_atom",
                        section=section_name,
                        rule_strength=strength,
                    )
                )
            elif section_name in {"positive_examples", "negative_examples"}:
                polarity = "positive" if section_name == "positive_examples" else "negative"
                records.append(
                    _record(
                        base,
                        record_id=f"{document_id}__example__{polarity}__{section_id}__{item_index:03d}",
                        text=clean_item,
                        record_type="example",
                        section=section_name,
                    )
                )
    return sorted(records, key=lambda item: str(item["_id"]))


def _rule_strength(text: str) -> str:
    match = RULE_TAG_RE.match(text.strip())
    return match.group(1).lower() if match else ""


def _record_id_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-").lower()
    if normalized:
        return normalized
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"section-{digest}"


def _record(
    base: Mapping[str, Any],
    *,
    record_id: str,
    text: str,
    record_type: str,
    section: str,
    rule_strength: str = "",
) -> dict[str, Any]:
    referenced_columns = [f"{table}.{column}" for table, column in _referenced_columns(text)]
    record = {
        **dict(base),
        "_id": record_id,
        "text": " ".join(text.split()),
        "referenced_columns": referenced_columns,
        "record_type": record_type,
        "section": section,
        "rule_strength": rule_strength,
        "rule_strength_rank": RULE_STRENGTH_ORDER.get(rule_strength, 99),
    }
    return dict(sorted(record.items()))


def _with_chunk_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, int] = {}
    counters: dict[str, int] = {}
    for record in records:
        doc_id = str(record.get("source_document_id") or record.get("document_id") or "")
        totals[doc_id] = totals.get(doc_id, 0) + 1
    normalized: list[dict[str, Any]] = []
    for record in records:
        doc_id = str(record.get("source_document_id") or record.get("document_id") or "")
        index = counters.get(doc_id, 0)
        counters[doc_id] = index + 1
        normalized.append(dict(sorted({**record, "chunk_count": totals.get(doc_id, 1), "chunk_index": index}.items())))
    return normalized


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
    for document in documents:
        for warning in document.warnings:
            print(f"warning: {document.path.name}: {warning}", file=sys.stderr)


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
    build_parser.add_argument("--integrity", type=Path, default=DEFAULT_INTEGRITY_PATH)
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
            records = build_jsonl(args.rules_dir, args.output, schema_path=args.schema, integrity_path=args.integrity)
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
