from __future__ import annotations

from backend.config import StorageMySQL
from backend.mysql import db

META_DATABASE = "daaat_meta"
SESSIONS_TABLE = "sessions"


def storage_connect(database: str | None = None):
    return db.connect(
        StorageMySQL.HOST,
        StorageMySQL.PORT,
        StorageMySQL.USER,
        StorageMySQL.PASSWORD,
        database=database,
    )


def ensure_session_store() -> None:
    conn = storage_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{META_DATABASE}` CHARACTER SET utf8mb4")
        conn.commit()
    finally:
        conn.close()

    meta = storage_connect(META_DATABASE)
    try:
        with meta.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS `{SESSIONS_TABLE}` (
                    id CHAR(8) PRIMARY KEY,
                    username VARCHAR(64) NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    source_host VARCHAR(255) NOT NULL,
                    source_port INT NOT NULL,
                    source_user VARCHAR(255) NOT NULL,
                    source_database VARCHAR(255) NOT NULL,
                    session_db VARCHAR(64) NOT NULL,
                    mart_db VARCHAR(64) NOT NULL,
                    status VARCHAR(16) NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    last_opened_at DATETIME NULL,
                    UNIQUE KEY uq_session_db (session_db),
                    UNIQUE KEY uq_mart_db (mart_db),
                    INDEX idx_sessions_username_created (username, created_at)
                ) CHARACTER SET utf8mb4
                """
            )
        meta.commit()
    finally:
        meta.close()