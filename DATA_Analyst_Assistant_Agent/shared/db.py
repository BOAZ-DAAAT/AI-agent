"""공통 DB 접속 계층.

`agents/sql/db/db_connect.py` 와 `agents/eda/db/db_connect.py` 에 중복돼 있던
`get_db_engine` 을 단일화한다. import 시 `shared.config` 를 끌어와 `.env` 로드 +
`DB_*`/`MYSQL_*` 별칭 정규화가 먼저 수행되도록 보장한다.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, text

# .env 로드 + DB_*/MYSQL_* 별칭 정규화 (import 부작용으로 1회 수행).
import DATA_Analyst_Assistant_Agent.shared.config  # noqa: F401


def get_db_engine():
    """MySQL SQLAlchemy 엔진 생성. 연결 테스트 후 성공 시 엔진, 실패 시 None 반환."""
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME")

    database_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    try:
        engine = create_engine(database_url)
        with engine.connect() as connection:  # 실제 연결 확인
            print(f"✅ DB 연결 성공: {db_name}")
        return engine
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return None


def get_table_samples(engine, table_name, sample_count=10):
    """테이블에서 랜덤 N개 샘플을 list[dict] 로 추출(LLM 스키마 설명용)."""
    try:
        with engine.connect() as connection:
            query = text(f"SELECT * FROM `{table_name}` ORDER BY RAND() LIMIT {sample_count}")
            df = pd.read_sql(query, connection)
            return df.to_dict(orient="records")
    except Exception as e:
        print(f"⚠️ {table_name} 샘플 추출 실패: {e}")
        return []


def load_mart(engine, table_name: str) -> pd.DataFrame:
    """테이블 전체를 DataFrame 으로 적재."""
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM `{table_name}`"), conn)
    return df
