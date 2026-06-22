import json
from pathlib import Path

from DATA_Analyst_Assistant_Agent.shared.config import sql_metadata_dir

DATA_DIR = sql_metadata_dir()

SCHEMA_JSON_PATH = DATA_DIR / "db_schema.json"
INTEGRITY_JSON_PATH = DATA_DIR / "db_integrity_result.json"


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_schema_json():
    return _read_json(SCHEMA_JSON_PATH)


def load_integrity_json():
    return _read_json(INTEGRITY_JSON_PATH)


def load_schema_text():
    data = load_schema_json()
    return json.dumps(data, ensure_ascii=False, indent=2)


def load_integrity_text():
    data = load_integrity_json()
    return json.dumps(data, ensure_ascii=False, indent=2)


def load_all_metadata():
    return {
        "schema_json": load_schema_json(),
        "integrity_json": load_integrity_json(),
        "schema_text": load_schema_text(),
        "integrity_text": load_integrity_text(),
    }