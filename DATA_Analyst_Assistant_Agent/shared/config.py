"""공통 설정 계층.

- `.env`를 1회 로드한다(레포 루트 기준, 멱등).
- `DB_*` ↔ `MYSQL_*` / `GOOGLE_API_KEY` ↔ `GEMINI_API_KEY` 별칭을 정규화한다.
- SQL 에이전트 메타데이터 디렉터리 경로를 CWD 비의존으로 단일 제공한다.

기존에 `run.py`/`scripts/run_orchestration_demo.py`/각 모듈에 흩어져 있던
`load_dotenv()` 와 alias 정규화 로직을 한곳으로 모은 것이다.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# shared/ -> DATA_Analyst_Assistant_Agent/ -> repo root(ADV)
_REPO_ROOT = Path(__file__).resolve().parents[2]

# .env 로드(이미 설정된 환경변수는 덮어쓰지 않음 = 멱등).
load_dotenv(_REPO_ROOT / ".env")


def normalize_env_aliases() -> None:
    """`DB_*`가 없고 `MYSQL_*`만 있으면 채워준다(역도 아님). 코드가 읽는 이름은 `DB_*`."""
    aliases = {
        "DB_HOST": "MYSQL_HOST",
        "DB_USER": "MYSQL_USERNAME",
        "DB_NAME": "MYSQL_DATABASE",
        "DB_PORT": "MYSQL_PORT",
        "DB_PASSWORD": "MYSQL_PASSWORD",
        "GOOGLE_API_KEY": "GEMINI_API_KEY",
    }
    for target, source in aliases.items():
        if not os.getenv(target) and os.getenv(source):
            os.environ[target] = os.getenv(source, "")
    os.environ.setdefault("DB_PASSWORD", "")


# import 시점에 별칭 정규화를 1회 수행(이 모듈을 import하면 DB_*가 채워진 상태가 됨).
normalize_env_aliases()


def sql_metadata_dir() -> Path:
    """`db_schema.json` / `db_integrity_result.json` 가 사는 디렉터리(패키지 기준, CWD 비의존)."""
    # shared/ -> DATA_Analyst_Assistant_Agent/  ; 그 아래 agents/sql/data
    return Path(__file__).resolve().parent.parent / "agents" / "sql" / "data"
