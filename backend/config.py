from __future__ import annotations

import os

from dotenv import load_dotenv

# 레포 루트의 .env 파일을 읽어 환경변수로 등록한다.
# override=False: 이미 진짜 환경변수가 있으면 .env보다 그걸 우선한다 (배포 환경 대비).
load_dotenv(override=False)


class StorageMySQL:
    """사본 저장용 로컬 MySQL 접속정보. 우리 인프라라서 서버 자신이 env로 안다."""
    HOST = os.getenv("STORAGE_MYSQL_HOST", "127.0.0.1")
    PORT = int(os.getenv("STORAGE_MYSQL_PORT", "3306"))
    USER = os.getenv("STORAGE_MYSQL_USER", "root")
    PASSWORD = os.getenv("STORAGE_MYSQL_PASSWORD", "")


# ingest 안전장치
INGEST_ROW_LIMIT = int(os.getenv("INGEST_ROW_LIMIT", "1000000"))   # 테이블당 최대 행
INGEST_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "10000"))   # 한 번에 옮기는 행 수