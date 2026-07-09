from __future__ import annotations

import pymysql

from backend.config import INGEST_CHUNK_SIZE, INGEST_ROW_LIMIT, StorageMySQL
from backend.mysql import db




# 사본 저장용 로컬 MySQL 연결.
def _local_connect(database: str | None = None):
    return db.connect(
        StorageMySQL.HOST, StorageMySQL.PORT,
        StorageMySQL.USER, StorageMySQL.PASSWORD,
        database=database,
    )

# 원격 DB의 뷰를 제외한 실제 테이블만 나열한다
def _list_base_tables(host: str, port: int, user: str, password: str, database: str) -> list[str]:
    conn = db.connect(host, port, user, password, database=database)
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW FULL TABLES WHERE Table_type = 'BASE TABLE'")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

# 사용자 서버가 우리 저장소 서버와 같은지 확인
def _same_server(host: str, port: int) -> bool:
    def norm(h: str) -> str:
        return "127.0.0.1" if h == "localhost" else h
    return norm(host) == norm(StorageMySQL.HOST) and port == StorageMySQL.PORT

# 원격 DB의 모든 테이블을 로컬 MySQL로 복사한다.
def ingest_database(
    host: str, port: int, user: str, password: str, database: str,
    target_database: str | None = None,
) -> dict[str, int]:
    db.quote_identifier(database)
    target = target_database or database
    db.quote_identifier(target)

    # 원본과 사본이 같은 서버의 같은 DB면 차단한다.
    if _same_server(host, port) and target == database:
        raise ValueError("원본 서버가 저장소 서버와 같습니다. target_database로 다른 이름을 지정하세요.")

    quoted_db = f"`{target}`"

    # 원격의 뷰를 제외한 테이블 목록
    tables = db.list_base_tables(host, port, user, password, database)

    # 로컬에 사본 DB를 새로 만든다
    local = _local_connect()
    try:
        with local.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {quoted_db}")
            cur.execute(f"CREATE DATABASE {quoted_db} CHARACTER SET utf8mb4")
    finally:
        local.close()

    # 테이블 하나씩 복사
    copied: dict[str, int] = {}
    for table in tables:
        copied[table] = _copy_table(host, port, user, password, database, table, target)
    return copied


# 테이블 하나를 복사
def _copy_table(host: str, port: int, user: str, password: str, database: str, table: str, target: str) -> int:
    quoted = db.quote_identifier(table)

    remote = db.connect(host, port, user, password, database=database)
    local = _local_connect(database=target)
    try:
        # 복사 중에는 외래키 검사 끔 (테이블 생성/입력 순서 문제 방지)
        with local.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            cur.execute("SET SESSION sql_mode = ''") 

        with remote.cursor() as cur:
            cur.execute(f"SHOW CREATE TABLE {quoted}")
            create_sql = cur.fetchone()[1]
            
        with local.cursor() as cur:
            cur.execute(create_sql)

        copied = 0
        with remote.cursor(pymysql.cursors.SSCursor) as rcur:
            rcur.execute(f"SELECT * FROM {quoted} LIMIT %s", (INGEST_ROW_LIMIT,))
            n_cols = len(rcur.description)
            placeholders = ", ".join(["%s"] * n_cols)
            insert_sql = f"INSERT INTO {quoted} VALUES ({placeholders})"

            while True:
                rows = rcur.fetchmany(INGEST_CHUNK_SIZE)
                if not rows:
                    break
                with local.cursor() as lcur:
                    lcur.executemany(insert_sql, rows)
                local.commit()  # 청크마다 확정 (pymysql은 자동커밋 꺼져 있음)
                copied += len(rows)
        return copied
    finally:
        remote.close()
        local.close()