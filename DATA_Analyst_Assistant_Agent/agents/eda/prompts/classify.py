"""컬럼 의미 분류 프롬프트 (load 노드용)."""

from __future__ import annotations

import json


def classify_columns_prompt(all_columns: list, measure_cols: list, sample: dict) -> str:
    return f"""
아래 데이터마트의 컬럼 의미를 파악하라.

[전체 컬럼] {all_columns}
[measure 컬럼 (분석 지표, 분류 불필요)] {measure_cols}
[분류 대상 컬럼 + 샘플값]
{json.dumps(sample, ensure_ascii=False, default=str)}

아래 두 가지를 판단하라:
1. time_columns: 날짜/시간/기간을 나타내는 컬럼 목록 (없으면 빈 리스트)
   - 예: created_at, order_time, month, period, dt, ts, 주문일자 등
2. count_column: 표본 수 또는 볼륨을 나타내는 컬럼 1개 (없으면 빈 문자열)
   - 예: volume, qty, n_records, num_transactions, 주문건수 등
   - measure_cols 안에 있어도 됨

반드시 아래 JSON 형식으로만 응답하라. 설명 없이 JSON만 출력하라.
{{
  "time_columns": [...],
  "count_column": "..."
}}
"""
