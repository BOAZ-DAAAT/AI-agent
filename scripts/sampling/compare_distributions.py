"""샘플 DB가 원본 분포(특히 결측 비율)를 심하게 훼손하지 않았는지 확인한다.

sample_olist.py로 DB를 만든 직후에 돌려서, 원본과 샘플의 핵심 지표를 나란히 비교한다.
숫자만 보여주고 판단은 사람이 한다 — 자동으로 실패 처리하지 않는다.

사용법:
    python scripts/sampling/compare_distributions.py
    python scripts/sampling/compare_distributions.py --source-db olist --target-db olist_sampling
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")


def _engine(database: str) -> Engine:
    username = os.getenv("MYSQL_USERNAME") or os.getenv("DB_USER") or "root"
    password = os.getenv("MYSQL_PASSWORD") or os.getenv("DB_PASSWORD") or ""
    host = os.getenv("MYSQL_HOST") or os.getenv("DB_HOST") or "localhost"
    port = os.getenv("MYSQL_PORT") or os.getenv("DB_PORT") or "3306"
    return create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")


def _scalar(engine: Engine, sql: str) -> float | None:
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


def _null_rate(engine: Engine, table: str, column: str) -> float:
    total = _scalar(engine, f"SELECT COUNT(*) FROM {table}") or 0
    if total == 0:
        return 0.0
    nulls = _scalar(engine, f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL") or 0
    return nulls / total


def _quantiles(engine: Engine, sql: str) -> pd.Series:
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df.iloc[:, 0].quantile([0.25, 0.5, 0.75, 0.95])


CHECKS: list[tuple[str, str, str]] = [
    # (표시 이름, 테이블, NULL 체크할 컬럼)
    ("배송완료일 결측", "orders", "order_delivered_customer_date"),
    ("배송승인일 결측", "orders", "order_approved_at"),
    ("리뷰 코멘트제목 결측", "order_reviews", "review_comment_title"),
    ("리뷰 코멘트본문 결측", "order_reviews", "review_comment_message"),
    ("상품 카테고리 결측", "products", "product_category_name"),
]


def compare(source_db: str, target_db: str) -> None:
    src = _engine(source_db)
    tgt = _engine(target_db)

    print(f"=== 기본 규모 비교 ({source_db} vs {target_db}) ===")
    for table in (
        "customers",
        "orders",
        "order_items",
        "order_payments",
        "order_reviews",
        "products",
        "sellers",
    ):
        src_n = _scalar(src, f"SELECT COUNT(*) FROM {table}")
        tgt_n = _scalar(tgt, f"SELECT COUNT(*) FROM {table}")
        ratio = (tgt_n / src_n * 100) if src_n else 0.0
        print(f"  {table:28s} 원본 {src_n:>8,}  ->  샘플 {tgt_n:>7,}  ({ratio:5.1f}%)")

    print(f"\n=== 결측 비율 비교 (원본 대비 몇 %p 차이나는지) ===")
    for label, table, column in CHECKS:
        src_rate = _null_rate(src, table, column)
        tgt_rate = _null_rate(tgt, table, column)
        diff = (tgt_rate - src_rate) * 100
        flag = "  ⚠ 3%p 이상 차이" if abs(diff) >= 3.0 else ""
        print(f"  {label:20s} 원본 {src_rate:6.2%}  샘플 {tgt_rate:6.2%}  (차이 {diff:+.2f}%p){flag}")

    print("\n=== 고객당 결제액(monetary) 분위수 비교 ===")
    payment_sql = """
        SELECT SUM(op.payment_value) AS total_payment
        FROM orders o JOIN order_payments op ON op.order_id = o.order_id
        GROUP BY o.customer_id
    """
    src_q = _quantiles(src, payment_sql)
    tgt_q = _quantiles(tgt, payment_sql)
    for q in src_q.index:
        print(f"  Q{int(q * 100):>2d}  원본 {src_q[q]:>10,.2f}  샘플 {tgt_q[q]:>10,.2f}")

    print("\n=== 리뷰 점수 분포 비교 ===")
    review_sql = "SELECT review_score, COUNT(*) AS n FROM order_reviews GROUP BY review_score ORDER BY review_score"
    with src.connect() as conn:
        src_reviews = pd.read_sql(text(review_sql), conn).set_index("review_score")["n"]
    with tgt.connect() as conn:
        tgt_reviews = pd.read_sql(text(review_sql), conn).set_index("review_score")["n"]
    src_pct = src_reviews / src_reviews.sum() * 100
    tgt_pct = tgt_reviews / tgt_reviews.sum() * 100
    for score in sorted(set(src_pct.index) | set(tgt_pct.index)):
        print(f"  {score}점  원본 {src_pct.get(score, 0):5.1f}%  샘플 {tgt_pct.get(score, 0):5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="원본 DB와 샘플 DB의 분포를 비교한다.")
    parser.add_argument("--source-db", default=os.getenv("MYSQL_DATABASE", "olist"))
    parser.add_argument("--target-db", default="olist_sampling")
    args = parser.parse_args()
    compare(args.source_db, args.target_db)


if __name__ == "__main__":
    main()
