import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def _get_mysql_config() -> dict[str, str]:
    """MYSQL_*를 표준으로 사용하고, DB_*는 하위 호환 fallback으로만 허용합니다."""
    username = os.getenv("MYSQL_USERNAME") or os.getenv("DB_USER") or ""
    password = os.getenv("MYSQL_PASSWORD") or os.getenv("DB_PASSWORD") or ""
    host = os.getenv("MYSQL_HOST") or os.getenv("DB_HOST") or ""
    port = os.getenv("MYSQL_PORT") or os.getenv("DB_PORT") or "3306"
    database = os.getenv("MYSQL_DATABASE") or os.getenv("DB_NAME") or ""
    return {
        "username": username,
        "password": password,
        "host": host,
        "port": port,
        "database": database,
    }


def get_db_engine():
    """
    MySQL 데이터베이스 연결을 위한 SQLAlchemy Engine을 생성하여 반환합니다.
    이 엔진은 GE 검증이나 SQL 실행 시 공통으로 사용됩니다.
    """
    config = _get_mysql_config()
    database_url = (
        f"mysql+pymysql://{config['username']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['database']}"
    )

    try:
        engine = create_engine(database_url)
        with engine.connect() as connection:
            print(f"✅ DB 연결 성공: {config['database']}")
        return engine
    except Exception as e:
        print(
            f"❌ DB 연결 실패: {e}\n"
            "필요한 환경변수: MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, MYSQL_USERNAME, MYSQL_PASSWORD"
        )
        return None


def get_table_samples(engine, table_name, sample_count=10):
    """테이블에서 랜덤 샘플을 추출합니다."""
    try:
        with engine.connect() as connection:
            query = text(f"SELECT * FROM `{table_name}` ORDER BY RAND() LIMIT {sample_count}")
            df = pd.read_sql(query, connection)
            return df.to_dict(orient="records")
    except Exception as e:
        print(f"⚠️ {table_name} 샘플 추출 실패: {e}")
        return []


if __name__ == "__main__":
    engine = get_db_engine()

    if engine:
        test_table = "users"
        samples = get_table_samples(engine, test_table, 10)

        print(f"\n🔍 '{test_table}' 테이블 샘플 데이터 (최대 10개):")
        for i, row in enumerate(samples, 1):
            print(f"{i}: {row}")
    else:
        print("❌ 테스트 실패: 엔진을 생성할 수 없습니다.")
