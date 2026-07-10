from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from backend.config import StorageMySQL
from DATA_Analyst_Assistant_Agent.shared.config import sql_metadata_dir


@contextmanager
def _dataset_env(dataset_name: str):
    keys = {
        "DB_HOST": StorageMySQL.HOST,
        "DB_PORT": str(StorageMySQL.PORT),
        "DB_USER": StorageMySQL.USER,
        "DB_PASSWORD": StorageMySQL.PASSWORD,
        "DB_NAME": dataset_name,
        "MYSQL_HOST": StorageMySQL.HOST,
        "MYSQL_PORT": str(StorageMySQL.PORT),
        "MYSQL_USERNAME": StorageMySQL.USER,
        "MYSQL_PASSWORD": StorageMySQL.PASSWORD,
        "MYSQL_DATABASE": dataset_name,
    }
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(keys)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _create_engine(dataset_name: str):
    database_url = (
        f"mysql+pymysql://{StorageMySQL.USER}:{StorageMySQL.PASSWORD}@"
        f"{StorageMySQL.HOST}:{StorageMySQL.PORT}/{dataset_name}"
    )
    return create_engine(database_url)


def _create_ge_context(base_dir: Path):
    import great_expectations as gx

    base_dir.mkdir(parents=True, exist_ok=True)
    return gx.get_context(project_root_dir=str(base_dir))


def _get_schema_info_minimal(engine) -> dict[str, Any]:
    schema_data: dict[str, Any] = {}
    inspector = inspect(engine)
    for table in inspector.get_table_names():
        pk_info = inspector.get_pk_constraint(table)
        schema_data[table] = {
            "primary_key": pk_info.get("constrained_columns", []),
            "columns": [],
            "foreign_keys": [],
        }
        for col in inspector.get_columns(table):
            schema_data[table]["columns"].append(
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"],
                }
            )
        for fk in inspector.get_foreign_keys(table):
            schema_data[table]["foreign_keys"].append(
                {
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"],
                    "constrained_columns": fk["constrained_columns"],
                }
            )
    return schema_data


def _normalize_scope(schema_info: dict[str, Any], table_scope: list[str] | None) -> list[str]:
    if not table_scope:
        return sorted(schema_info.keys())
    allowed = {str(table).strip().strip("`").split(".")[-1] for table in table_scope if str(table).strip()}
    return [table for table in sorted(schema_info.keys()) if table in allowed]


def _get_validator_and_batch(context, table_name: str, df):
    from great_expectations.core.batch import RuntimeBatchRequest

    datasource_name = "my_datasource"
    suite_name = f"{table_name}_suite"
    if suite_name in context.list_expectation_suite_names():
        context.delete_expectation_suite(suite_name)
    context.add_expectation_suite(expectation_suite_name=suite_name)

    existing_ds = [d["name"] for d in context.list_datasources()]
    if datasource_name not in existing_ds:
        context.add_datasource(
            name=datasource_name,
            class_name="Datasource",
            execution_engine={"class_name": "PandasExecutionEngine"},
            data_connectors={
                "runtime_data_connector": {
                    "class_name": "RuntimeDataConnector",
                    "batch_identifiers": ["default_identifier_name"],
                }
            },
        )

    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
        data_connector_name="runtime_data_connector",
        data_asset_name=table_name,
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default_id"},
    )
    validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)
    return validator, batch_request, suite_name


def _format_physical_results(run_id: str, all_results: list[Any]) -> dict[str, Any]:
    import datetime

    final_report = {
        "run_id": run_id,
        "summary": {"total_tables": len(all_results), "status": "PASS", "tested_at": datetime.datetime.now().isoformat()},
        "tables": {},
    }

    for result in all_results:
        run_results = result.run_results[next(iter(result.run_results))]
        table_name = run_results["validation_result"]["meta"]["active_batch_definition"]["data_asset_name"]
        final_report["tables"][table_name] = {"status": "PASS", "checks": []}

        for validation_result in result.list_validation_results():
            for check in validation_result.results:
                is_pass = check.success
                if not is_pass:
                    final_report["summary"]["status"] = "ACTION_REQUIRED"
                    final_report["tables"][table_name]["status"] = "ACTION_REQUIRED"

                res = check.result
                exp_type = check.expectation_config.expectation_type
                raw_val = res.get("unexpected_count") if exp_type == "expect_column_values_to_not_be_null" else res.get("observed_value")
                observed_val = str(raw_val) if raw_val is not None else "N/A"
                memo = check.expectation_config.meta.get("desc", "상세 설명 없음")
                final_report["tables"][table_name]["checks"].append(
                    {
                        "layer": "physical",
                        "column": check.expectation_config.kwargs.get("column", "Table-Level"),
                        "status": "PASS" if is_pass else "FAIL",
                        "intent": memo,
                        "observed": observed_val,
                    }
                )

    return final_report


def _run_scoped_physical_checks(engine, context, schema_info: dict[str, Any], tables: list[str], run_id: str) -> dict[str, Any]:
    all_results = []
    checkpoint_name = "integrity_checkpoint"
    context.add_or_update_checkpoint(
        name=checkpoint_name,
        class_name="Checkpoint",
        config_version=1,
        action_list=[
            {"name": "store_validation_result", "action": {"class_name": "StoreValidationResultAction"}},
            {"name": "update_data_docs", "action": {"class_name": "UpdateDataDocsAction"}},
        ],
    )

    for table_name in tables:
        table = schema_info[table_name]
        df = pd.read_sql(text(f"SELECT * FROM `{table_name}`"), engine)
        if df.empty:
            continue
        validator, batch_request, suite_name = _get_validator_and_batch(context, table_name, df)
        pks = table.get("primary_key", [])

        for pk in pks:
            if pk in df.columns:
                validator.expect_column_values_to_be_unique(
                    pk,
                    meta={"desc": f"식별자 중복 검사: '{pk}'는 테이블의 기본키이므로 값이 유일해야 합니다."},
                )
                validator.expect_column_values_to_not_be_null(
                    pk,
                    meta={"desc": f"식별자 누락 검사: 기본키 '{pk}'는 절대 비어있을 수 없습니다."},
                )

        for col in table["columns"]:
            col_name = col["name"]
            if col_name not in df.columns:
                continue

            if not col["nullable"]:
                validator.expect_column_values_to_not_be_null(
                    col_name,
                    meta={"desc": f"필수값 누락 검사: '{col_name}'은 설계상 반드시 값이 존재해야 하는 컬럼입니다."},
                )

            db_type = str(col["type"]).upper()
            target_type = None
            if any(t in db_type for t in ["INT", "BIT"]):
                target_type = "int64"
            elif any(t in db_type for t in ["FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL"]):
                target_type = "float64"
            elif any(t in db_type for t in ["CHAR", "TEXT", "STRING"]):
                target_type = "str"
            elif any(t in db_type for t in ["DATE", "TIME", "STAMP"]):
                target_type = "datetime64[ns]"

            if target_type:
                validator.expect_column_values_to_be_of_type(
                    column=col_name,
                    type_=target_type,
                    meta={"desc": f"물리적 타입 일치성: 원본 DB의 '{db_type}' 형식이 로드 과정에서 변조되지 않고 '{target_type}'으로 유지되었는지 검증합니다."},
                )

            if col_name not in pks:
                validator.expect_column_values_to_not_be_null(
                    col_name,
                    mostly=0.0,
                    meta={"desc": f"데이터 점유 현황: '{col_name}'의 실제 데이터 점유율을 파악해 결측치 수를 산출합니다."},
                )

            validator.expect_column_unique_value_count_to_be_between(
                col_name,
                min_value=0,
                meta={"desc": f"데이터 다양성 지표: '{col_name}' 내 고유값 개수를 통해 범주형 여부를 진단합니다."},
            )

            if target_type in ["int64", "float64"]:
                validator.expect_column_min_to_be_between(
                    col_name,
                    min_value=-999999999999,
                    max_value=999999999999,
                    meta={"desc": f"데이터 최소값: '{col_name}'의 하한선을 확인합니다."},
                )
                validator.expect_column_max_to_be_between(
                    col_name,
                    min_value=-999999999999,
                    max_value=999999999999,
                    meta={"desc": f"데이터 최대값: '{col_name}'의 상한선을 확인합니다."},
                )

        schema_cols = [c["name"] for c in table["columns"]]
        validator.expect_table_columns_to_match_set(
            schema_cols,
            meta={"desc": "스키마 구성 검사: 실제 테이블 컬럼이 DB 설계도와 일치하는지 확인합니다."},
        )

        if "foreign_keys" in table:
            for fk in table["foreign_keys"]:
                try:
                    ref_table = fk.get("ref_table") or fk.get("referred_table")
                    ref_col = (fk.get("ref_column") or fk.get("referred_columns"))[0]
                    child_col = fk.get("column") or fk.get("constrained_columns")[0]
                    if child_col in df.columns:
                        parent_ids = pd.read_sql(text(f"SELECT DISTINCT `{ref_col}` FROM `{ref_table}`"), engine)[ref_col].tolist()
                        validator.expect_column_values_to_be_in_set(
                            column=child_col,
                            value_set=parent_ids,
                            meta={"desc": f"참조 정합성(FK) 검사: '{child_col}'의 값이 부모 테이블('{ref_table}')에 실존하는지 확인합니다."},
                        )
                except Exception:
                    pass

        with engine.connect() as conn:
            db_count = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar()

        validator.expect_table_row_count_to_equal(
            db_count,
            meta={"desc": f"데이터 유실 검사: 원본 DB({db_count}건)와 로드된 데이터 수가 정확히 일치하는지 검증합니다."},
        )

        context.add_or_update_expectation_suite(expectation_suite=validator.expectation_suite)
        result = context.run_checkpoint(
            checkpoint_name=checkpoint_name,
            run_name=run_id,
            validations=[{"batch_request": batch_request, "expectation_suite_name": suite_name}],
        )
        all_results.append(result)

    return _format_physical_results(run_id, all_results)


def _run_scoped_semantic_checks(engine, context, physical_report: dict[str, Any], tables: list[str], run_id: str) -> dict[str, Any]:
    from DATA_Analyst_Assistant_Agent.agents.sql.db.integrity_semantic import SemanticValidator

    validator = SemanticValidator(engine, context)
    inspector = inspect(engine)
    all_table_names = inspector.get_table_names()
    target_tables = set(tables)
    all_results: dict[str, list[dict[str, Any]]] = {}

    for table in validator.get_tables_without_pk():
        if table not in target_tables:
            continue
        history = physical_report["tables"].get(table, {}).get("checks", [])
        if not history:
            continue
        hypothesis = validator.hypothesizer.analyze_pk_hypothesis(table, history)
        all_results[table] = [validator.validate_pk_integrity(table, hypothesis)]

    for table in validator.get_tables_without_fk():
        if table not in target_tables:
            continue
        history = physical_report["tables"].get(table, {}).get("checks", [])
        if not history:
            continue
        columns = [c["column"] for c in history if c.get("column") != "Table-Level"]
        fk_hypotheses = validator.hypothesizer.analyze_fk_hypothesis(table, columns, all_table_names)
        fk_results = validator.validate_fk_integrity(table, fk_hypotheses)
        if table in all_results:
            all_results[table].extend(fk_results)
        else:
            all_results[table] = fk_results

    return validator.get_formatted_report(run_id, all_results)


def _table_summary(physical_checks: list[dict[str, Any]], semantic_checks: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [check for check in [*physical_checks, *semantic_checks] if str(check.get("status", "")).upper() != "PASS"]
    physical_status = "fail" if any(str(check.get("status", "")).upper() == "FAIL" for check in physical_checks) else "pass"
    semantic_status = "fail" if any(str(check.get("status", "")).upper() == "FAIL" for check in semantic_checks) else "pass"
    overall = "fail" if failures else "pass"
    return {
        "status": overall,
        "physical_status": physical_status,
        "semantic_status": semantic_status,
        "failures": failures,
        "temporary_fix_artifacts": [],
        "root_cause_summary": [check.get("intent", "") for check in failures[:5] if check.get("intent")],
    }


def run_dataset_integrity_checks(
    *,
    dataset_name: str,
    table_scope: list[str] | None,
    source_version: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import great_expectations as gx  # noqa: F401  # imported to fail fast when missing

    checked_at = __import__("datetime").datetime.now().isoformat()
    run_id = f"run_{dataset_name}_{checked_at.replace(':', '').replace('-', '')}"
    ge_root = sql_metadata_dir() / ".ge_runtime" / dataset_name

    with _dataset_env(dataset_name):
        engine = _create_engine(dataset_name)
        context = _create_ge_context(ge_root)
        schema_info = _get_schema_info_minimal(engine)
        tables = _normalize_scope(schema_info, table_scope)
        physical_report = _run_scoped_physical_checks(engine, context, schema_info, tables, run_id)

        semantic_warning = None
        try:
            semantic_report = _run_scoped_semantic_checks(engine, context, physical_report, tables, run_id)
        except Exception as exc:
            semantic_report = {"run_id": run_id, "summary": {"total_tables": len(tables), "status": "WARNING", "tested_at": checked_at}, "tables": {}}
            semantic_warning = f"{type(exc).__name__}: {exc}"

        metadata_dir = sql_metadata_dir()
        metadata_dir.mkdir(parents=True, exist_ok=True)
        run_dir = metadata_dir / "integrity_runs" / dataset_name / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        physical_path = metadata_dir / "db_integrity_result_physical.json"
        semantic_path = metadata_dir / "db_integrity_result_semantic.json"
        immutable_physical_path = run_dir / "physical.json"
        immutable_semantic_path = run_dir / "semantic.json"
        physical_path.write_text(__import__("json").dumps(physical_report, ensure_ascii=False, indent=2), encoding="utf-8")
        semantic_path.write_text(__import__("json").dumps(semantic_report, ensure_ascii=False, indent=2), encoding="utf-8")
        immutable_physical_path.write_text(__import__("json").dumps(physical_report, ensure_ascii=False, indent=2), encoding="utf-8")
        immutable_semantic_path.write_text(__import__("json").dumps(semantic_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_tables: dict[str, Any] = {}
    for table in tables:
        physical_entry = physical_report.get("tables", {}).get(table, {})
        semantic_entry = semantic_report.get("tables", {}).get(table, [])
        semantic_checks = semantic_entry if isinstance(semantic_entry, list) else semantic_entry.get("checks", [])
        summary = _table_summary(physical_entry.get("checks", []), semantic_checks)
        if semantic_warning and not semantic_checks:
            summary["semantic_status"] = "warning"
            summary["status"] = "warning" if summary["status"] == "pass" else summary["status"]
            summary["failures"].append(
                {
                    "layer": "semantic",
                    "status": "WARNING",
                    "intent": "Semantic integrity checks were unavailable.",
                    "observed": semantic_warning,
                }
            )
            summary["root_cause_summary"].append("Semantic integrity checks were unavailable.")
        summary_tables[table] = summary

    return {
        "dataset_name": dataset_name,
        "source_version": source_version,
        "checked_at": checked_at,
        "artifact_refs": [
            {"path": str(immutable_physical_path), "kind": "physical_report"},
            {"path": str(immutable_semantic_path), "kind": "semantic_report"},
        ],
        "physical_report": physical_report,
        "semantic_report": semantic_report,
        "summary_report": {"tables": summary_tables, "semantic_warning": semantic_warning},
        "tables": summary_tables,
    }
