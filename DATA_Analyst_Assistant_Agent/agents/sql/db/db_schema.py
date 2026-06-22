import os
import json
import datetime
from sqlalchemy import inspect
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
from DATA_Analyst_Assistant_Agent.shared.config import sql_metadata_dir
from DATA_Analyst_Assistant_Agent.shared.db import get_db_engine, get_table_samples

# 메타데이터 캐시 경로(패키지 기준, CWD 비의존). .env 로드/별칭 정규화는 shared.config 가 수행.
SCHEMA_CACHE_PATH = str(sql_metadata_dir() / "db_schema.json")
sql_metadata_dir().mkdir(parents=True, exist_ok=True)

def enrich_schema_descriptions(engine, schema_data): # engine 인자 추가
    """
    수정 사항: 샘플 데이터 10개를 직접 추출하여 프롬프트에 포함하고, 
    결과 구조에도 sample_data를 추가합니다.
    """
    model_name = os.getenv("SCHEMA_LLM_MODEL") or os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL")

    if not model_name:
        raise ValueError("🚨 .env에 'SCHEMA_LLM_MODEL'이 설정되지 않았습니다.")

    llm = get_chat_model(model=model_name, temperature=0)
    print(f"🤖 {model_name} 모델이 샘플 데이터를 분석 중입니다... (비용 발생)")

    for table_name, details in schema_data.items():
        # 1. 회의 내용 반영: 직접 데이터 샘플 10개 접근
        samples = get_table_samples(engine, table_name, sample_count=10)
        details["sample_data"] = samples # 최종 결과에 샘플 데이터 포함

        col_info = [{"name": col["name"], "type": col["type"]} for col in details["columns"]]
        
        # 2. 프롬프트 수정: 샘플 데이터를 기반으로 의미를 파악하도록 지시
        prompt = (
            f"너는 데이터베이스 전문가야. 다음 테이블의 '컬럼 정보'와 '실제 데이터 샘플'을 보고 "
            f"테이블의 용도와 각 컬럼의 의미를 한글로 상세히 설명해줘.\n\n"
            f"테이블명: {table_name}\n"
            f"컬럼 목록: {col_info}\n"
            f"데이터 샘플(10개): {samples}\n\n"
            f"--- 반드시 다음 JSON 형식으로만 응답해 ---\n"
            f"{{\n"
            f"  \"table_desc\": \"테이블 전체 역할에 대한 한 문장 설명\",\n"
            f"  \"column_descs\": {{\n"
            f"    \"컬럼명\": \"데이터 샘플을 토대로 파악한 해당 컬럼의 역할 설명\"\n"
            f"  }}\n"
            f"}}"
        )
        
        try:
            response = llm.invoke(prompt)
            res_json = json.loads(response.content.replace("```json", "").replace("```", "").strip())
            
            details["description"] = res_json.get("table_desc", "설명이 없습니다.")
            for col in details["columns"]:
                col_name = col["name"]
                col["description"] = res_json.get("column_descs", {}).get(col_name, "설명이 없습니다.")
                
        except Exception as e:
            print(f"⚠️ {table_name} 설명 생성 실패: {e}")
            details["description"] = "설명 생성 중 오류가 발생했습니다."
            
    return schema_data


def get_schema_info(engine, use_llm=False, use_cache=True): # use_cache 추가
    """
    회의 내용 반영: use_cache가 True면 기존에 생성된 파일이 있을 때 그걸 읽어옵니다.
    """
    # 1. 캐시 확인 (비용 방어)
    if use_cache and os.path.exists(SCHEMA_CACHE_PATH):
        print(f"📦 기존에 생성된 스키마 정보({SCHEMA_CACHE_PATH})를 활용합니다.")
        with open(SCHEMA_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    inspector = inspect(engine)
    schema_data = {}
    
    table_names = inspector.get_table_names()
    
    for table in table_names:
        pk_info = inspector.get_pk_constraint(table)
        schema_data[table] = {
            "primary_key": pk_info.get("constrained_columns", []),
            "description": "LLM 설명이 비활성화되어 있습니다.", 
            "columns": [],
            "foreign_keys": []
        }
        
        columns = inspector.get_columns(table)
        for col in columns:
            schema_data[table]["columns"].append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col["nullable"],
                "description": "" 
            })
            
        fk_relations = inspector.get_foreign_keys(table)
        for fk in fk_relations:
            schema_data[table]["foreign_keys"].append({
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"],
                "constrained_columns": fk["constrained_columns"]
            })
            
    # LLM 설명 보강
    if use_llm:
        schema_data = enrich_schema_descriptions(engine, schema_data) 
        
        def json_default(obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()  # 날짜를 "2026-03-22" 같은 문자열로 변환
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(SCHEMA_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(schema_data, f, indent=4, ensure_ascii=False, default=json_default)
            print(f"💾 스키마 정보가 {SCHEMA_CACHE_PATH}에 저장되었습니다.")
            
    return schema_data


if __name__ == "__main__":
    test_engine = get_db_engine()
    if test_engine:
        # 처음 실행 시에는 use_llm=True, 그 다음부터는 저장된 파일을 쓰게 됩니다.
        info = get_schema_info(test_engine, use_llm=True, use_cache=True)
        print("✅ 스키마 처리 완료")
