import os
import json
import pandas as pd
from datetime import datetime
from sqlalchemy import inspect, text
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from langchain_core.prompts import ChatPromptTemplate
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
from DATA_Analyst_Assistant_Agent.agents.sql.db.db_schema import get_schema_info

class SemanticHypothesizer:
    """[1단계] 🧠 추론 Layer: 물리 지표 기반 PK 및 FK 후보 선정"""
    def __init__(self):
        self.llm = get_chat_model(temperature=0)

    def _build_column_stats(self, physical_history):
        column_stats = {}
        total_count = 0
        for check in physical_history:
            col = check["column"]
            val = check["observed"]
            if col == "Table-Level" and "유실 검사" in check["intent"]:
                total_count = int(val) if str(val).isdigit() else 0
                continue
            if col == "Table-Level": continue
            column_stats.setdefault(col, {"column": col})
            if "결측치" in check["intent"] or "점유" in check["intent"]:
                column_stats[col]["null_count"] = int(val) if str(val).isdigit() else 0
            elif "고유값" in check["intent"]:
                column_stats[col]["unique_count"] = int(val) if str(val).isdigit() else 0
        for col in column_stats:
            column_stats[col]["total_count"] = total_count
        return list(column_stats.values())

    def analyze_pk_hypothesis(self, table_name, physical_history):
        """PK(기본키) 후보 추론"""
        stats = self._build_column_stats(physical_history)
        prompt = ChatPromptTemplate.from_template("""
            너는 15년 경력의 베테랑 '데이터 아키텍트'다. 
            '{table_name}' 테이블의 물리적 통계 지표를 분석하여 최적의 PK(기본키) 가설을 세워라.

            [컬럼별 물리 통계]
            {stats}

            [감사 지침]
            1. 'id', 'pk' 포함 컬럼을 우선하되 실제 고유성(Unique)과 결측치(Null) 수치를 최우선 신뢰하라.
            2. 단일 컬럼으로 부족하면 비즈니스 맥락상 유효한 '복합키'를 제안하라.
            3. 위도, 경도, 시간값 등은 식별자로 부적합하다고 판정하라.

            [결과 작성 지침]
            - 'reasoning' 필드에는 선정 근거를 반드시 "~합니다" 또는 "~습니다"로 끝나는 정중한 문장으로 작성하라.

            반드시 아래 JSON으로만 응답하라:
            {{
                "column": "선정된 컬럼명 또는 리스트",
                "reasoning": "선정 근거 (반드시 ~습니다 로 끝나는 문장)"
            }}
        """)
        chain = prompt | self.llm
        res = chain.invoke({"table_name": table_name, "stats": json.dumps(stats, ensure_ascii=False)})
        return json.loads(res.content.replace("```json", "").replace("```", ""))

    def analyze_fk_hypothesis(self, table_name, columns, all_table_names):
        """FK(외래키) 관계 추론"""
        prompt = ChatPromptTemplate.from_template("""
            너는 데이터 모델링 전문가다. '{table_name}' 테이블의 컬럼 목록과 전체 테이블 명단을 보고, 
            타 테이블을 참조할 가능성이 높은 외래키(FK) 후보를 모두 찾아라.

            [현재 컬럼] {columns}
            [전체 테이블] {all_tables}

            [결과 작성 지침]
            - 'reasoning' 필드에는 참조 근거를 반드시 "~합니다" 또는 "~습니다"로 끝나는 정중한 문장으로 작성하라.

            반드시 아래 형식의 JSON 리스트로만 응답하라:
            [ {{ "column": "현재컬럼", "ref_table": "대상테이블", "ref_column": "대상컬럼", "reasoning": "참조 근거 (~습니다)" }} ]
        """)
        chain = prompt | self.llm
        res = chain.invoke({"table_name": table_name, "columns": columns, "all_tables": all_table_names})
        return json.loads(res.content.replace("```json", "").replace("```", ""))

class SemanticValidator:
    """[2단계] 💪 검증 Layer: 가설 확증 및 리포트 생성"""
    def __init__(self, engine, context):
        self.engine = engine
        self.context = context
        self.hypothesizer = SemanticHypothesizer() # 통합 실행을 위해 내부에서 생성

    def get_tables_without_pk(self):
        inspector = inspect(self.engine)
        return [t for t in inspector.get_table_names() if not inspector.get_pk_constraint(t).get('constrained_columns')]

    def get_tables_without_fk(self):
        """DB 스키마에 FK(외래키)가 하나도 정의되지 않은 테이블 목록 추출"""
        inspector = inspect(self.engine)
        all_tables = inspector.get_table_names()
        return [t for t in all_tables if not inspector.get_foreign_keys(t)]

    def run_all_semantic_logic(self, phys_report):
        """
        [추가/통합] 노트북의 조잡한 루프 로직을 파일 내부로 캡슐화합니다.
        물리 리포트(phys_report)를 입력받아 전체 테이블의 PK/FK 추론 및 검증을 수행합니다.
        """
        all_results = {}
        all_tables = list(phys_report['tables'].keys())
        
        # 1. PK 부재 테이블 추론 및 검증
        target_pk_tables = self.get_tables_without_pk()
        for table in target_pk_tables:
            history = phys_report['tables'].get(table, {}).get('checks', [])
            hypothesis = self.hypothesizer.analyze_pk_hypothesis(table, history)
            # 결과를 리스트로 감싸서 저장 (FK 결과와 규격 통일)
            all_results[table] = [self.validate_pk_integrity(table, hypothesis)]

        # 2. FK 부재 테이블 추론 및 검증
        target_fk_tables = self.get_tables_without_fk()
        for table in target_fk_tables:
            # 현재 테이블 컬럼 추출 (Table-Level 제외)
            columns = [c['column'] for c in phys_report['tables'][table]['checks'] if c['column'] != 'Table-Level']
            fk_hypotheses = self.hypothesizer.analyze_fk_hypothesis(table, columns, all_tables)
            fk_results = self.validate_fk_integrity(table, fk_hypotheses)
            
            # 기존 PK 결과가 있으면 확장(extend), 없으면 신규 저장
            if table in all_results:
                all_results[table].extend(fk_results)
            else:
                all_results[table] = fk_results
                
        return all_results

    def validate_pk_integrity(self, table_name, hypothesis):
        """PK 정합성 검증"""
        candidate = hypothesis["column"]
        reasoning = hypothesis["reasoning"]
        df = pd.read_sql(text(f"SELECT * FROM `{table_name}`"), self.engine)
        
        is_success, details = self._run_ge_pk_checks(table_name, df, candidate)
        final_status = "PASS" if is_success else "FAIL"

        if final_status == "PASS":
            observed_msg = f"{reasoning} 이에 따라 {candidate} 컬럼을 최적의 식별자로 선정하였으며, 실측 데이터 검증 결과 모든 정합성이 완벽히 확보되어 최종 PASS 되었습니다."
        else:
            observed_msg = f"{reasoning} 이러한 분석을 바탕으로 {candidate} 컬럼을 식별자 후보로 선정하였으나, 실측 과정에서 중복 데이터 {details['duplicate_count']}건 등이 식별되어 검증을 통과하지 못하였습니다. 이는 원천 데이터의 품질 개선이 시급한 상태임을 의미합니다."

        return {
            "layer": "semantic",
            "column": str(candidate),
            "status": final_status,
            "intent": "PK 부재 테이블 식별자 추론 및 GE 확증",
            "observed": observed_msg
        }

    def validate_fk_integrity(self, table_name, fk_hypotheses):
        """FK 참조 정합성 검증"""
        df_child = pd.read_sql(text(f"SELECT * FROM `{table_name}`"), self.engine)
        results = []

        for fk in fk_hypotheses:
            col, ref_t, ref_c = fk["column"], fk["ref_table"], fk["ref_column"]
            try:
                df_parent = pd.read_sql(text(f"SELECT DISTINCT `{ref_c}` FROM `{ref_t}`"), self.engine)
                parent_ids = df_parent[ref_c].tolist()

                batch_request = RuntimeBatchRequest(
                    datasource_name="my_datasource",
                    data_connector_name="runtime_data_connector",
                    data_asset_name=f"fk_check_{table_name}_{ref_t}",
                    runtime_parameters={"batch_data": df_child},
                    batch_identifiers={"default_identifier_name": "default_id"}
                )
                
                suite_name = f"fk_{table_name}_suite"
                if suite_name not in self.context.list_expectation_suite_names():
                    self.context.add_expectation_suite(suite_name)

                validator = self.context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)
                res = validator.expect_column_values_to_be_in_set(col, parent_ids)
                
                status = "PASS" if res.success else "FAIL"
                
                if status == "PASS":
                    msg = f"{fk['reasoning']} 이에 따라 {ref_t} 테이블을 참조하는 외래키로 판단하였으며, 실측 결과 모든 데이터가 부모 테이블에 실존함을 확인하여 PASS 되었습니다."
                else:
                    msg = f"{fk['reasoning']} 이러한 분석을 바탕으로 {ref_t} 테이블의 참조 키로 추론하였으나, 실측 과정에서 부모 데이터가 존재하지 않는 고아 데이터 {res.result.get('unexpected_count', 0)}건이 식별되어 실패하였습니다. 이는 원천 데이터의 품질 개선이 시급한 상태임을 의미합니다."

                results.append({
                    "layer": "semantic",
                    "column": col,
                    "status": status,
                    "intent": "FK 부재 테이블 식별자 추론 및 GE 확증",
                    "observed": msg
                })
            except: continue
        return results

    def _run_ge_pk_checks(self, table_name, df, columns):
        """PK 실측을 수행합니다."""
        batch_request = RuntimeBatchRequest(
            datasource_name="my_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name=f"semantic_pk_{table_name}", 
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default_id"}
        )
        
        suite_name = f"pk_{table_name}_suite"
        if suite_name not in self.context.list_expectation_suite_names():
            self.context.add_expectation_suite(suite_name)

        validator = self.context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)
        cols = [columns] if isinstance(columns, str) else columns
        try:
            res_u = validator.expect_compound_columns_to_be_unique(cols) if len(cols) > 1 else validator.expect_column_values_to_be_unique(cols[0])
            duplicate_count = res_u.result.get("unexpected_count", 0)
            null_count = sum(validator.expect_column_values_to_not_be_null(c).result.get("unexpected_count", 0) for c in cols)
            return (duplicate_count == 0 and null_count == 0), {"duplicate_count": duplicate_count, "null_count": null_count}
        except: return False, {"duplicate_count": "Unknown", "null_count": "Unknown"}

    def get_formatted_report(self, run_id, all_results):
        """[통합용] KeyError 방지 및 통합 구조를 위한 포맷터"""
        has_fail = False
        for table_res in all_results.values():
            if isinstance(table_res, list):
                if any(r.get("status") == "FAIL" for r in table_res):
                    has_fail = True
                    break
            elif isinstance(table_res, dict):
                if table_res.get("status") == "FAIL":
                    has_fail = True
                    break

        report = {
            "run_id": run_id,
            "summary": {
                "total_tables": len(all_results),
                "status": "ACTION_REQUIRED" if has_fail else "PASS",
                "tested_at": datetime.now().isoformat()
            },
            "tables": all_results
        }
        return report

    def save_results(self, all_results, run_id, path="data/db_integrity_result_semantic.json"):
        """[통합용] 공통 run_id를 사용하여 결과를 저장합니다."""
        report = self.get_formatted_report(run_id, all_results)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        return report
