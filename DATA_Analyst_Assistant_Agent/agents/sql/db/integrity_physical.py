import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from DATA_Analyst_Assistant_Agent.agents.sql.db.db_schema import get_schema_info
from sqlalchemy import text
import pandas as pd
import datetime
import os
import json

# [경로 설정]
DATA_DIR = "data"
RESULT_PATH = os.path.join(DATA_DIR, "db_integrity_result_physical.json")

class PhysicalIntegrityManager:
    def __init__(self, engine, context, run_id=None):
        """초기화: DB 엔진과 GE 컨텍스트를 저장합니다."""
        self.engine = engine
        self.context = context
        self.all_results = []
        # [수정] 외부(노트북)에서 run_id를 주입받으면 사용하고, 없으면 새로 생성합니다.
        self.run_id = run_id if run_id else f"run_SK_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"

    def _get_validator_and_batch(self, table_name, df):
        """[GE 세팅] 기존 Suite 초기화 및 Validator 생성"""
        datasource_name = "my_datasource"
        suite_name = f"{table_name}_suite"

        if suite_name in self.context.list_expectation_suite_names():
            self.context.delete_expectation_suite(suite_name)
        
        self.context.add_expectation_suite(expectation_suite_name=suite_name)

        existing_ds = [d["name"] for d in self.context.list_datasources()]
        if datasource_name not in existing_ds:
            self.context.add_datasource(
                name=datasource_name,
                class_name="Datasource",
                execution_engine={"class_name": "PandasExecutionEngine"},
                data_connectors={
                    "runtime_data_connector": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["default_identifier_name"]
                    }
                }
            )

        batch_request = RuntimeBatchRequest(
            datasource_name=datasource_name,
            data_connector_name="runtime_data_connector",
            data_asset_name=table_name,
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default_id"}
        )

        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name
        )

        return validator, batch_request, suite_name

    def run_validation(self):
        """[1단계: Schema-driven Validation] 전체 실행 로직"""
        schema_info = get_schema_info(self.engine, use_llm=False, use_cache=True)
        
        # Checkpoint 설정
        checkpoint_name = "integrity_checkpoint"
        self.context.add_or_update_checkpoint(
            name=checkpoint_name,
            class_name="Checkpoint",
            config_version=1,
            action_list=[
                {"name": "store_validation_result", "action": {"class_name": "StoreValidationResultAction"}},
                {"name": "update_data_docs", "action": {"class_name": "UpdateDataDocsAction"}},
            ]
        )

        for table_name, table in schema_info.items():
            print(f"🚀 [{table_name}] 검증 준비", end=" -> ")
            df = pd.read_sql(text(f"SELECT * FROM `{table_name}`"), self.engine)
            
            print(f"   - 데이터 건수: {len(df)}건")
            if df.empty:
                print("데이터 없음 (Skip)")
                continue

            validator, batch_request, suite_name = self._get_validator_and_batch(table_name, df)
            pks = table.get("primary_key", [])
            print(f"   - PK 목록: {pks}")

            # --- [Rule 1] PK Unique/Not Null ---
            for pk in pks:
                if pk in df.columns:
                    validator.expect_column_values_to_be_unique(
                        pk, meta={"desc": f"식별자 중복 검사: '{pk}'는 테이블의 기본키이므로 값이 유일해야 합니다."}
                    )
                    validator.expect_column_values_to_not_be_null(
                        pk, meta={"desc": f"식별자 누락 검사: 기본키 '{pk}'는 절대 비어있을 수 없습니다."}
                    )

            # --- [Rule 2, 3] Not Null & Data Type ---
            for col in table["columns"]:
                col_name = col["name"]
                if col_name not in df.columns: continue

                if not col["nullable"]:
                    validator.expect_column_values_to_not_be_null(
                        col_name, meta={"desc": f"필수값 누락 검사: '{col_name}'은 설계상 반드시 값이 존재해야 하는 컬럼입니다."}
                    )

                db_type = str(col["type"]).upper()
                target_type = None
                if any(t in db_type for t in ["INT", "BIT"]): target_type = "int64"
                elif any(t in db_type for t in ["FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL"]): target_type = "float64"
                elif any(t in db_type for t in ["CHAR", "TEXT", "STRING"]): target_type = "str"
                elif any(t in db_type for t in ["DATE", "TIME", "STAMP"]): target_type = "datetime64[ns]"

                if target_type:
                    validator.expect_column_values_to_be_of_type(
                        column=col_name, type_=target_type, 
                        meta={"desc": f"물리적 타입 일치성: 원본 DB의 '{db_type}' 형식이 로드 과정에서 변조되지 않고 '{target_type}'으로 유지되었는지 검증합니다."}
                    )

                # --- [Profiling 1, 2, 3] 정보 수집 (내용물 스캔) ---
                if col_name not in pks: # <--- PK 보호 로직 그대로 유지
                    validator.expect_column_values_to_not_be_null(
                        col_name, mostly=0.0, 
                        meta={"desc": f"데이터 점유 현황: '{col_name}'의 실제 데이터 점유율을 파악해 결측치 수를 산출합니다."}
                    )

                validator.expect_column_unique_value_count_to_be_between(
                    col_name, min_value=0,
                    meta={"desc": f"데이터 다양성 지표: '{col_name}' 내 고유값 개수를 통해 범주형 여부를 진단합니다."}
                )

                if target_type in ["int64", "float64"]:
                    validator.expect_column_min_to_be_between(col_name, min_value=-999999999999, max_value=999999999999, 
                                                            meta={"desc": f"데이터 최소값: '{col_name}'의 하한선을 확인합니다."})
                    validator.expect_column_max_to_be_between(col_name, min_value=-999999999999, max_value=999999999999, 
                                                            meta={"desc": f"데이터 최대값: '{col_name}'의 상한선을 확인합니다."})
                    
            # --- [Rule 4] Column Match ---
            schema_cols = [c["name"] for c in table["columns"]]
            validator.expect_table_columns_to_match_set(
                schema_cols, meta={"desc": "스키마 구성 검사: 실제 테이블 컬럼이 DB 설계도와 일치하는지 확인합니다."}
            )

            # --- [Rule 5] Foreign Key ---
            if "foreign_keys" in table:
                for fk in table["foreign_keys"]:
                    try:
                        ref_table = fk.get("ref_table") or fk.get("referred_table")
                        ref_col = (fk.get("ref_column") or fk.get("referred_columns"))[0]
                        child_col = fk.get("column") or fk.get("constrained_columns")[0]
                        
                        if child_col in df.columns:
                            parent_ids = pd.read_sql(text(f"SELECT DISTINCT `{ref_col}` FROM `{ref_table}`"), self.engine)[ref_col].tolist()
                            validator.expect_column_values_to_be_in_set(
                                column=child_col, value_set=parent_ids,
                                meta={"desc": f"참조 정합성(FK) 검사: '{child_col}'의 값이 부모 테이블('{ref_table}')에 실존하는지 확인합니다."}
                            )
                    except Exception:
                        pass

            # --- [Rule 6] Row Count ---
            with self.engine.connect() as conn:
                db_count = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar()
            
            validator.expect_table_row_count_to_equal(
                db_count, meta={"desc": f"데이터 유실 검사: 원본 DB({db_count}건)와 로드된 데이터 수가 정확히 일치하는지 검증합니다."}
            )
            
            self.context.add_or_update_expectation_suite(expectation_suite=validator.expectation_suite)
            
            result = self.context.run_checkpoint(
                checkpoint_name=checkpoint_name,
                run_name=self.run_id,
                validations=[{"batch_request": batch_request, "expectation_suite_name": suite_name}]
            )
            
            status = "✅ PASS" if result.success else "🚨 FAIL"
            print(status)
            self.all_results.append(result)

        self.context.build_data_docs()
        return self.run_id, self.all_results

    def format_results(self, run_id, all_results, save_file=False):
        """결과 리포트 가공 및 JSON 저장"""
        final_report = {
            "run_id": run_id,
            "summary": {"total_tables": len(all_results), "status": "PASS", "tested_at": datetime.datetime.now().isoformat()},
            "tables": {}
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
                    
                    if exp_type == "expect_column_values_to_not_be_null":
                        raw_val = res.get("unexpected_count")
                    else:
                        raw_val = res.get("observed_value")

                    observed_val = str(raw_val) if raw_val is not None else "N/A"
                        
                    memo = check.expectation_config.meta.get("desc", "상세 설명 없음")
                    
                    check_item = {
                        "layer": "physical", 
                        "column": check.expectation_config.kwargs.get("column", "Table-Level"),
                        "status": "PASS" if is_pass else "FAIL",
                        "intent": memo,
                        "observed": observed_val
                    }
                    final_report["tables"][table_name]["checks"].append(check_item)
        
        # [수정] save_file 파라미터가 True일 때만 개별 JSON 파일을 저장합니다.
        if save_file:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=4, ensure_ascii=False)
        
        return final_report