import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def _get_mysql_config() -> dict[str, str]:
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
    config = _get_mysql_config()
    database_url = (
        f"mysql+pymysql://{config['username']}:{config['password']}@"
        f"{config['host']}:{config['port']}/{config['database']}"
    )
    return create_engine(database_url)


def load_mart(engine, table_name: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM `{table_name}`"), conn)
