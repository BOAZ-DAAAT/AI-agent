import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

from DATA_Analyst_Assistant_Agent.shared.config import sql_metadata_dir

DATA_DIR = sql_metadata_dir()

SCHEMA_JSON_PATH = DATA_DIR / "db_schema.json"
INTEGRITY_JSON_PATH = DATA_DIR / "db_integrity_result.json"

_ISSUE_STATUSES = {"FAIL", "FAILED", "ERROR", "WARNING", "WARN", "ACTION_REQUIRED", "STALE"}
_FAIL_STATUSES = {"FAIL", "FAILED", "ERROR", "ACTION_REQUIRED"}  # fail_only 모드: 실패만
_PASS_STATUSES = {"PASS", "PASSED", "SUCCESS", "OK"}
_MAX_INTEGRITY_LINES = 60
_MAX_DETAIL_CHARS = 260


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_schema_json():
    return _read_json(SCHEMA_JSON_PATH)


def load_integrity_json():
    return _read_json(INTEGRITY_JSON_PATH)


def load_schema_text():
    data = load_schema_json()
    return json.dumps(data, ensure_ascii=False, indent=2)


def _schema_tables(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    tables = data.get("tables")
    return tables if isinstance(tables, dict) else data


def _schema_with_tables(data: dict[str, Any], tables: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data.get("tables"), dict):
        return {**data, "tables": tables}
    return tables


def _compact_description(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    first_line = value.splitlines()[0] if value.splitlines() else ""
    return " ".join(first_line.split())[:160]


def _foreign_key_columns(foreign_keys: Any) -> set[str]:
    columns: set[str] = set()
    if not isinstance(foreign_keys, list):
        return columns
    for foreign_key in foreign_keys:
        if not isinstance(foreign_key, dict):
            continue
        for key in ("column", "columns", "local_column", "local_columns"):
            value = foreign_key.get(key)
            values = value if isinstance(value, list) else [value]
            columns.update(str(item) for item in values if item not in (None, ""))
    return columns


def _catalog_columns(table: dict[str, Any]) -> list[dict[str, Any]]:
    columns = table.get("columns")
    if not isinstance(columns, list):
        return []
    primary_key = table.get("primary_key")
    priority_names = {
        str(column) for column in (primary_key if isinstance(primary_key, list) else [primary_key])
        if column not in (None, "")
    }
    priority_names.update(_foreign_key_columns(table.get("foreign_keys")))
    ordered = [column for column in columns if isinstance(column, dict) and str(column.get("name") or "") in priority_names]
    ordered.extend(
        column for column in columns
        if isinstance(column, dict) and str(column.get("name") or "") not in priority_names
    )
    return [
        {"name": column.get("name"), "type": column.get("type")}
        for column in ordered[:5]
    ]


def load_schema_catalog_text() -> str:
    """계획 단계용 축약 스키마 카탈로그를 반환한다."""
    data = load_schema_json()
    catalog: dict[str, Any] = {}
    for table_name, table in _schema_tables(data).items():
        if not isinstance(table, dict):
            continue
        catalog[table_name] = {
            "description": _compact_description(table.get("description")),
            "primary_key": table.get("primary_key", []),
            "foreign_keys": table.get("foreign_keys", []),
            "columns": _catalog_columns(table),
        }
    return json.dumps(_schema_with_tables(data, catalog), ensure_ascii=False, indent=2)


def load_scoped_schema_text(tables: Iterable[str] | None) -> str:
    """선택 테이블의 상세 스키마를 샘플 데이터 없이 반환한다."""
    selected_tables = _table_filter(tables)
    if not selected_tables:
        return ""

    data = load_schema_json()
    scoped: dict[str, Any] = {}
    for table_name, table in _schema_tables(data).items():
        if table_name not in selected_tables or not isinstance(table, dict):
            continue
        columns = table.get("columns")
        scoped[table_name] = {
            "description": table.get("description", ""),
            "primary_key": table.get("primary_key", []),
            "foreign_keys": table.get("foreign_keys", []),
            "columns": [
                {
                    "name": column.get("name"),
                    "type": column.get("type"),
                    "nullable": column.get("nullable"),
                    "description": column.get("description", ""),
                }
                for column in columns if isinstance(columns, list) and isinstance(column, dict)
            ],
        }
    if not scoped:
        return ""
    return json.dumps(_schema_with_tables(data, scoped), ensure_ascii=False, indent=2)


def load_integrity_text():
    data = load_integrity_json()
    return json.dumps(data, ensure_ascii=False, indent=2)


def load_all_metadata():
    schema_json = load_schema_json()
    integrity_json = load_integrity_json()
    return {
        "schema_json": schema_json,
        "integrity_json": integrity_json,
        "schema_text": json.dumps(schema_json, ensure_ascii=False, indent=2),
        "integrity_text": compact_integrity_summary_text(integrity_json),
    }


_SCOPED_MAX_LINES = 100  # 실측(9테이블 fail_only 무스코핑=29줄) 대비 넉넉한 안전망 — 평소엔 안 걸리고 이상상황(스키마 드리프트 등)만 방어(#123)


def load_scoped_integrity_text(
    tables: Iterable[str] | None,
    *,
    fail_only: bool = True,
    max_lines: int | None = _SCOPED_MAX_LINES,
) -> str:
    """미리 생성해둔 정적 db_integrity_result.json 을 planned tables 로 스코핑해 프롬프트 텍스트를 만든다.

    로컬/오프라인(백엔드 정합성 서비스 미가동)에서 쓰는 경로. 파일이 없거나 비어 있으면 ""
    를 돌려 파이프라인을 막지 않는다. 기본은 fail_only=True(실패 검사만) + max_lines=100
    — 스코핑·fail_only로 실사용은 이미 유계(실측 29줄)지만, 100은 평소엔 절대 안 걸리는
    최후 방어선으로만 둔다(#123).

    tables 가 비어 있으면(None/[]) 스코프할 테이블이 없다는 뜻이므로 ""를 돌린다 — 여기서
    전체를 덤프하면 하위 스코핑 파서의 "필터 없음=전체" 규칙과 만나 토큰폭탄이 된다(#130).
    """
    if not tables:
        return ""
    try:
        data = load_integrity_json()
    except FileNotFoundError:
        return ""
    return compact_integrity_summary_text(
        data, tables=tables, fail_only=fail_only, max_lines=max_lines
    )


def preload_backend_integrity_summary(dataset_name: str) -> dict[str, Any]:
    service = _load_integrity_service()
    if service is None:
        return {"status": "unavailable", "reason": "integrity_service_not_configured", "integrity_text": ""}
    try:
        summary_result = _post_backend_json(
            service,
            "/integrity/summary",
            {"dataset_name": dataset_name, "include_pass": False},
        )
        summary_data = summary_result.get("data") if summary_result.get("ok") else None
        if not summary_data:
            return {
                "status": "error",
                "reason": (summary_result.get("error") or {}).get("message", "integrity summary unavailable"),
                "integrity_text": "",
            }
        return {
            "status": "prefetched",
            "summary": summary_data,
            "integrity_text": compact_integrity_summary_text(summary_data, include_pass=False),
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "integrity_text": ""}


def _normalize_table_name(table: Any) -> str:
    normalized = str(table or "").strip().strip("`")
    return normalized.split(".")[-1] if normalized else ""


def _table_filter(tables: Iterable[str] | None) -> set[str] | None:
    if not tables:
        return None
    if isinstance(tables, str):
        tables = [tables]
    filtered = {_normalize_table_name(table) for table in tables if _normalize_table_name(table)}
    return filtered or None


def _stringify_detail(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    text = " ".join(text.split())
    if len(text) > _MAX_DETAIL_CHARS:
        return text[: _MAX_DETAIL_CHARS - 1] + "…"
    return text


def _status_from_dict(item: dict[str, Any]) -> str | None:
    if item.get("stale") is True or item.get("is_stale") is True:
        return "STALE"

    for key in ("status", "result", "severity", "level"):
        raw = item.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().upper()

    if item.get("success") is False or item.get("passed") is False:
        return "FAIL"
    if item.get("success") is True or item.get("passed") is True:
        return "PASS"
    return None


def _is_prompt_relevant_status(status: str | None, include_pass: bool, fail_only: bool = False) -> bool:
    if not status:
        return False
    normalized = status.upper()
    if fail_only:
        return normalized in _FAIL_STATUSES
    return normalized in _ISSUE_STATUSES or (include_pass and normalized in _PASS_STATUSES)


def _extract_column(item: dict[str, Any]) -> str:
    if item.get("column"):
        return str(item["column"])
    kwargs = item.get("kwargs")
    if isinstance(kwargs, dict) and kwargs.get("column"):
        return str(kwargs["column"])
    config = item.get("expectation_config")
    if isinstance(config, dict):
        cfg_kwargs = config.get("kwargs")
        if isinstance(cfg_kwargs, dict) and cfg_kwargs.get("column"):
            return str(cfg_kwargs["column"])
    return ""


def _extract_check_name(item: dict[str, Any]) -> str:
    for key in ("check", "name", "expectation_type", "type", "intent"):
        if item.get(key):
            return str(item[key])
    config = item.get("expectation_config")
    if isinstance(config, dict):
        for key in ("expectation_type", "type"):
            if config.get(key):
                return str(config[key])
    return "integrity_check"


def _extract_detail(item: dict[str, Any]) -> str:
    candidates = [
        item.get("observed"),
        item.get("detail"),
        item.get("message"),
        item.get("reason"),
        item.get("description"),
        item.get("exception_info"),
    ]
    nested_summary = item.get("summary")
    if isinstance(nested_summary, dict):
        candidates.extend([
            nested_summary.get("message"),
            nested_summary.get("detail"),
            nested_summary.get("reason"),
            nested_summary.get("description"),
        ])
    meta = item.get("meta")
    if isinstance(meta, dict):
        candidates.extend([meta.get("desc"), meta.get("description")])
    result = item.get("result")
    if isinstance(result, dict):
        candidates.extend([
            result.get("unexpected_count"),
            result.get("unexpected_percent"),
            result.get("observed_value"),
        ])
    return next((text for text in (_stringify_detail(value) for value in candidates) if text), "")


def _iter_integrity_findings(
    value: Any,
    *,
    active_table: str = "",
    tables: set[str] | None = None,
    include_pass: bool = False,
    fail_only: bool = False,
):
    if isinstance(value, list):
        for item in value:
            yield from _iter_integrity_findings(
                item,
                active_table=active_table,
                tables=tables,
                include_pass=include_pass,
                fail_only=fail_only,
            )
        return

    if not isinstance(value, dict):
        return

    table_name = _normalize_table_name(
        value.get("table")
        or value.get("table_name")
        or value.get("data_asset_name")
        or active_table
    )
    if tables is not None and table_name and table_name not in tables:
        return

    status = _status_from_dict(value)
    if _is_prompt_relevant_status(status, include_pass, fail_only):
        yield {
            "table": table_name,
            "column": _extract_column(value),
            "status": status or "UNKNOWN",
            "check": _extract_check_name(value),
            "detail": _extract_detail(value),
        }

    for key, child in value.items():
        if key in {"result", "meta", "kwargs", "expectation_config"}:
            continue
        child_table = table_name
        if isinstance(child, (dict, list)) and key not in {
            "tables",
            "checks",
            "results",
            "expectations",
            "validations",
            "summary",
        }:
            child_table = _normalize_table_name(key) or child_table
        yield from _iter_integrity_findings(
            child,
            active_table=child_table,
            tables=tables,
            include_pass=include_pass,
            fail_only=fail_only,
        )


def compact_integrity_summary_text(
    summary: Any,
    *,
    tables: Iterable[str] | None = None,
    include_pass: bool = False,
    fail_only: bool = False,
    max_lines: int | None = _MAX_INTEGRITY_LINES,
) -> str:
    """Build compact prompt context from legacy, service, or GE-shaped payloads.

    By default only failures, warnings, and stale checks are included so SQL
    prompts do not receive full passing validation dumps.

    fail_only=True 면 실패(FAIL/ERROR/ACTION_REQUIRED)만 남긴다(경고/stale 제외).
    max_lines=None 이면 줄 상한을 두지 않는다(스코핑·fail_only 로 이미 유계일 때 사용).
    """
    table_filter = _table_filter(tables)
    findings: list[dict[str, str]] = []
    seen = set()
    for finding in _iter_integrity_findings(
        summary, tables=table_filter, include_pass=include_pass, fail_only=fail_only
    ):
        key = tuple(finding.get(field, "") for field in ("table", "column", "status", "check", "detail"))
        if key in seen:
            continue
        seen.add(key)
        findings.append(finding)
        if max_lines is not None and len(findings) >= max_lines:
            break

    if not findings:
        return ""

    header = (
        "Integrity context (only FAILED checks):"
        if fail_only
        else "Integrity context (only failures/warnings/stale checks):"
    )
    lines = [header]
    for finding in findings:
        location = finding["table"] or "dataset"
        if finding["column"]:
            location = f"{location}.{finding['column']}"
        detail = f" — {finding['detail']}" if finding["detail"] else ""
        lines.append(f"- [{finding['status']}] {location}: {finding['check']}{detail}")
    return "\n".join(lines)


def _load_integrity_service():
    base_url = os.getenv("DATA_AGENT_BACKEND_URL", "").strip().rstrip("/")
    if not base_url:
        return None
    return base_url


def _post_backend_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url=f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def notify_backend_dataset_update(
    dataset_name: str,
    tables: Iterable[str] | None = None,
    source_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    service = _load_integrity_service()
    table_list = list(tables or [])
    if service is None:
        return {"status": "unavailable", "reason": "integrity_service_not_configured", "tables": table_list}
    try:
        result = _post_backend_json(
            service,
            "/integrity/notify-dataset-update",
            {
                "dataset_name": dataset_name,
                "tables": table_list or None,
                "source_version": source_version,
                "metadata": metadata,
            },
        )
        return {
            "status": "notified" if result.get("ok") else "error",
            "dataset_name": dataset_name,
            "tables": table_list,
            "result": result.get("data"),
            "error": (result.get("error") or {}).get("message"),
        }
    except URLError as exc:
        return {"status": "error", "error": str(exc), "dataset_name": dataset_name, "tables": table_list}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "dataset_name": dataset_name, "tables": table_list}


def refresh_backend_integrity_summary(
    dataset_name: str,
    tables: Iterable[str],
    *,
    wait_timeout_s: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    table_list = [_normalize_table_name(table) for table in tables if _normalize_table_name(table)]
    if not table_list:
        return {"status": "skipped", "reason": "no_tables", "integrity_text": "", "tables": []}

    service = _load_integrity_service()
    if service is None:
        return {
            "status": "unavailable",
            "reason": "integrity_service_not_configured",
            "integrity_text": "",
            "tables": table_list,
        }

    try:
        ensure_result = _post_backend_json(
            service,
            "/integrity/ensure-tables-ready",
            {
                "dataset_name": dataset_name,
                "tables": table_list,
                "wait_timeout_s": wait_timeout_s,
                "metadata": metadata,
            },
        )
        summary_result = _post_backend_json(
            service,
            "/integrity/summary",
            {
                "dataset_name": dataset_name,
                "tables": table_list,
                "include_pass": False,
            },
        )
        summary_data = summary_result.get("data") if summary_result.get("ok") else None
        if not summary_data:
            return {
                "status": "error",
                "error": (summary_result.get("error") or {}).get("message", "integrity summary unavailable"),
                "dataset_name": dataset_name,
                "tables": table_list,
                "integrity_text": "",
            }
        ready = bool((ensure_result.get("data") or {}).get("ready")) if ensure_result.get("ok") else False
        status = "refreshed" if ready else "queued"
        return {
            "status": status,
            "dataset_name": dataset_name,
            "tables": table_list,
            "ready": ready,
            "integrity_text": compact_integrity_summary_text(summary_data, tables=table_list, include_pass=False),
            "summary": summary_data,
            "ensure_result": ensure_result.get("data") if ensure_result.get("ok") else None,
        }
    except URLError as exc:
        return {
            "status": "error",
            "error": str(exc),
            "dataset_name": dataset_name,
            "tables": table_list,
            "integrity_text": "",
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "dataset_name": dataset_name,
            "tables": table_list,
            "integrity_text": "",
        }
