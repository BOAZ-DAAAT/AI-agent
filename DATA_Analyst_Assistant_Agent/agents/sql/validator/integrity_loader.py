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


def _is_prompt_relevant_status(status: str | None, include_pass: bool) -> bool:
    if not status:
        return False
    normalized = status.upper()
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
):
    if isinstance(value, list):
        for item in value:
            yield from _iter_integrity_findings(
                item,
                active_table=active_table,
                tables=tables,
                include_pass=include_pass,
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
    if _is_prompt_relevant_status(status, include_pass):
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
        )


def compact_integrity_summary_text(
    summary: Any,
    *,
    tables: Iterable[str] | None = None,
    include_pass: bool = False,
) -> str:
    """Build compact prompt context from legacy, service, or GE-shaped payloads.

    By default only failures, warnings, and stale checks are included so SQL
    prompts do not receive full passing validation dumps.
    """
    table_filter = _table_filter(tables)
    findings: list[dict[str, str]] = []
    seen = set()
    for finding in _iter_integrity_findings(summary, tables=table_filter, include_pass=include_pass):
        key = tuple(finding.get(field, "") for field in ("table", "column", "status", "check", "detail"))
        if key in seen:
            continue
        seen.add(key)
        findings.append(finding)
        if len(findings) >= _MAX_INTEGRITY_LINES:
            break

    if not findings:
        return ""

    lines = ["Integrity context (only failures/warnings/stale checks):"]
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
