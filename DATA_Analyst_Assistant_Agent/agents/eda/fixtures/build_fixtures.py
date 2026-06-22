"""olist 원본에서 grain별 fixture(CSV + 입력계약 JSON)를 생성한다 (LLM 없음).

실행: PYTHONUTF8=1 .venv/Scripts/python.exe -m DATA_Analyst_Assistant_Agent.agents.eda.fixtures.build_fixtures

한 번 뽑아 두면 이후 DB 없이 fixture로 EDA를 개발·테스트할 수 있다.
fixture는 실제 olist 쿼리 결과 = "SQL이 GROUP BY를 이렇게(또는 덜) 하면 나올 형태".
"""

from __future__ import annotations

import json
import os

import pandas as pd

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# 공통 조인(배송완료 + 대표 카테고리 + 주별 합계 + 리뷰). 주문 1건 = 1행이 되도록 item/review는 미리 집계.
_BASE = """
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN (
    SELECT order_id, SUM(price) AS order_price, SUM(freight_value) AS order_freight,
           MIN(product_id) AS rep_product
    FROM order_items GROUP BY order_id
) it ON o.order_id = it.order_id
JOIN products p ON it.rep_product = p.product_id
JOIN product_category_name_translation cat ON p.product_category_name = cat.product_category_name
JOIN (
    SELECT order_id, AVG(review_score) AS review_score FROM order_reviews GROUP BY order_id
) rv ON o.order_id = rv.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND cat.product_category_name_english IS NOT NULL
"""

# F1: 주문 단위(raw) — 그룹당 여러 행, 분포/회귀/교차 가능. 결정론적 8000 샘플.
Q_F1 = f"""
SELECT
    o.order_id,
    cat.product_category_name_english AS product_category,
    c.customer_state,
    o.order_purchase_timestamp,
    ROUND(it.order_price, 2)   AS order_price,
    ROUND(it.order_freight, 2) AS order_freight,
    DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp) AS delivery_days,
    ROUND(rv.review_score, 2)  AS review_score
{_BASE}
ORDER BY o.order_id
LIMIT 8000
"""

# F2: 카테고리 단위(aggregated) — 그룹당 1행, 비교용. total_orders = 표본수.
Q_F2 = f"""
SELECT
    cat.product_category_name_english AS product_category,
    ROUND(AVG(rv.review_score), 3)  AS avg_review_score,
    ROUND(AVG(DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp)), 2) AS avg_delivery_days,
    ROUND(AVG(it.order_price), 2)   AS avg_order_price,
    COUNT(DISTINCT o.order_id)      AS total_orders
{_BASE}
GROUP BY cat.product_category_name_english
ORDER BY total_orders DESC
"""

# F3: 카테고리 × 월(2축, aggregated) — 다중 추세선/시간×범주.
Q_F3 = f"""
SELECT
    cat.product_category_name_english AS product_category,
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS order_month,
    ROUND(SUM(it.order_price), 2)  AS monthly_revenue,
    COUNT(DISTINCT o.order_id)     AS monthly_orders,
    ROUND(AVG(rv.review_score), 3) AS avg_review_score
{_BASE}
GROUP BY cat.product_category_name_english, DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m')
ORDER BY product_category, order_month
"""

FIXTURES = {
    "f1_order_level": {
        "query": Q_F1,
        "contract": {
            "name": "f1_order_level",
            "question": "배송일이 리뷰점수에 영향을 주는지 분석해줘",
            "question_type": "relationship",
            "grain": {"grain_hint": "one row per order", "level": "raw"},
            "columns": {
                "numeric": ["order_price", "order_freight", "delivery_days", "review_score"],
                "categorical": ["product_category", "customer_state"],
                "datetime": ["order_purchase_timestamp"],
                "id": ["order_id"],
                "count": None,
                "target_candidates": ["review_score"],
            },
        },
    },
    "f2_category_level": {
        "query": Q_F2,
        "contract": {
            "name": "f2_category_level",
            "question": "카테고리별 성과(리뷰·배송·매출)를 비교해줘",
            "question_type": "comparison",
            "grain": {"grain_hint": "one row per product_category", "level": "aggregated"},
            "columns": {
                "numeric": ["avg_review_score", "avg_delivery_days", "avg_order_price", "total_orders"],
                "categorical": ["product_category"],
                "datetime": [],
                "id": [],
                "count": "total_orders",
                "target_candidates": ["avg_review_score"],
            },
        },
    },
    "f3_category_month": {
        "query": Q_F3,
        "contract": {
            "name": "f3_category_month",
            "question": "카테고리별 월매출 추세를 비교해줘",
            "question_type": "time",
            "grain": {"grain_hint": "one row per (product_category, month)", "level": "aggregated"},
            "columns": {
                "numeric": ["monthly_revenue", "monthly_orders", "avg_review_score"],
                "categorical": ["product_category"],
                "datetime": ["order_month"],
                "count": "monthly_orders",
                "target_candidates": ["monthly_revenue", "avg_review_score"],
            },
        },
    },
}


def _ensure_env() -> None:
    os.environ.setdefault("MYSQL_HOST", "localhost")
    os.environ.setdefault("MYSQL_USER", "root")
    os.environ.setdefault("MYSQL_PASSWORD", "1234")
    os.environ.setdefault("MYSQL_DB", "olist")
    os.environ.setdefault("MYSQL_DATABASE", "olist")
    os.environ.setdefault("MYSQL_PORT", "3306")


def main() -> None:
    _ensure_env()
    from sqlalchemy import text
    from DATA_Analyst_Assistant_Agent.shared.db import get_db_engine

    engine = get_db_engine()
    if engine is None:
        raise SystemExit("DB 연결 실패 — MySQL84 서비스/.env 확인")

    with engine.connect() as conn:
        for name, spec in FIXTURES.items():
            df = pd.read_sql(text(spec["query"]), conn)
            # 원본 olist는 범주값에 \r(윈도우 줄바꿈)이 섞여 있음 → 문자열 컬럼 정리
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].str.strip()
            csv_path = os.path.join(FIXTURE_DIR, f"{name}.csv")
            json_path = os.path.join(FIXTURE_DIR, f"{name}.contract.json")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(spec["contract"], f, ensure_ascii=False, indent=2)
            print(f"[{name}] rows={len(df)}, cols={list(df.columns)} -> {os.path.basename(csv_path)}")

    print("DONE")


if __name__ == "__main__":
    main()
