"""Olist 원본 DB에서 customer_unique_id 기준 층화 샘플링으로 개발용 축소 DB를 만든다.

핵심 원칙:
- customer_id(PK)가 아니라 customer_unique_id(진짜 사람)를 기준으로 샘플링한다.
  customer_id는 주문마다 새로 발급되는 계정이라, 이걸 기준으로 뽑으면 같은 사람의
  주문 이력이 쪼개져서 RFM의 frequency/monetary가 왜곡된다.
- customers -> orders -> order_items/order_payments/order_reviews 순서로
  "부모가 뽑힌 것만 자식도 따라간다"는 계단식 구조로 복사한다. 각 테이블을
  독립적으로 다시 샘플링하지 않는다 (그러면 조인이 깨진다).
- products/sellers/product_category_name_translation/geolocation은 카디널리티가
  낮아 원본을 그대로 복사한다(참조 무결성 걱정 없음).

사용법:
    python scripts/sampling/sample_olist.py
    python scripts/sampling/sample_olist.py --target-customers 2000 --seed 7
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_SQL = _REPO_ROOT / "seed" / "00_schema.sql"
_STAGING_TABLE = "_sample_customer_ids"

load_dotenv(_REPO_ROOT / ".env")


def _mysql_config() -> dict[str, str]:
    return {
        "username": os.getenv("MYSQL_USERNAME") or os.getenv("DB_USER") or "root",
        "password": os.getenv("MYSQL_PASSWORD") or os.getenv("DB_PASSWORD") or "",
        "host": os.getenv("MYSQL_HOST") or os.getenv("DB_HOST") or "localhost",
        "port": os.getenv("MYSQL_PORT") or os.getenv("DB_PORT") or "3306",
    }


def _engine(database: str | None = None) -> Engine:
    cfg = _mysql_config()
    db_part = f"/{database}" if database else ""
    url = f"mysql+pymysql://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}{db_part}"
    return create_engine(url)


def create_target_schema(target_db: str) -> None:
    """seed/00_schema.sql의 CREATE TABLE 구문을 target_db에 그대로 적용한다(원본 스키마와 동일)."""
    sql_text = _SCHEMA_SQL.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    statements = [s for s in statements if not s.upper().startswith(("CREATE DATABASE", "USE "))]

    server_engine = _engine()
    with server_engine.begin() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS `{target_db}`"))
        conn.execute(text(f"CREATE DATABASE `{target_db}` CHARACTER SET utf8mb4"))

    target_engine = _engine(target_db)
    with target_engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    print(f"  [schema] {target_db}에 테이블 {len(statements)}개 생성")


def load_customer_profile(source_engine: Engine) -> pd.DataFrame:
    """고객(customer_unique_id) 단위로 층화에 쓸 집계값을 한 번에 뽑는다."""
    query = """
        SELECT
            c.customer_unique_id,
            COUNT(DISTINCT o.order_id) AS order_count,
            COALESCE(SUM(op.order_payment_sum), 0) AS total_payment,
            AVG(r.order_avg_review) AS avg_review_score,
            MIN(o.order_purchase_timestamp) AS first_order_ts
        FROM customers c
        JOIN orders o ON o.customer_id = c.customer_id
        LEFT JOIN (
            SELECT order_id, SUM(payment_value) AS order_payment_sum
            FROM order_payments GROUP BY order_id
        ) op ON op.order_id = o.order_id
        LEFT JOIN (
            SELECT order_id, AVG(review_score) AS order_avg_review
            FROM order_reviews GROUP BY order_id
        ) r ON r.order_id = o.order_id
        GROUP BY c.customer_unique_id
    """
    with source_engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def build_strata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_count_bucket"] = pd.cut(
        df["order_count"], bins=[0, 1, 3, np.inf], labels=["1건", "2~3건", "4건+"]
    )
    df["amount_bucket"] = pd.qcut(
        df["total_payment"].rank(method="first"), 5, labels=[f"Q{i + 1}" for i in range(5)]
    )

    def _review_bucket(value: float) -> str:
        if pd.isna(value):
            return "리뷰없음"
        if value <= 2:
            return "1~2점"
        if value < 4:
            return "3점"
        return "4~5점"

    df["review_bucket"] = df["avg_review_score"].apply(_review_bucket)
    df["quarter_bucket"] = pd.to_datetime(df["first_order_ts"]).dt.to_period("Q").astype(str)
    df["stratum"] = (
        df["order_count_bucket"].astype(str)
        + "|"
        + df["amount_bucket"].astype(str)
        + "|"
        + df["review_bucket"]
        + "|"
        + df["quarter_bucket"]
    )
    return df


def select_sample(
    df: pd.DataFrame, target_customers: int, representative_ratio: float, seed: int
) -> pd.DataFrame:
    rep_n = int(target_customers * representative_ratio)
    rare_n = target_customers - rep_n

    # 대표성 그룹: stratum 조합별로 원본 비율만큼 뽑는다.
    parts = []
    for _, group in df.groupby("stratum", observed=True):
        n = max(1, round(len(group) / len(df) * rep_n))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    rep_sample = pd.concat(parts) if parts else df.iloc[0:0]

    # 희귀케이스 그룹: 고액 상위 5% / 다회구매(4건+) / 리뷰없음 / 저평점(1~2점) 중 하나라도 해당.
    # 랜덤 샘플링에서 씹혀나가기 쉬운 케이스라 별도로 의도적으로 오버샘플한다.
    high_value_cutoff = df["total_payment"].quantile(0.95)
    rare_mask = (
        (df["total_payment"] >= high_value_cutoff)
        | (df["order_count_bucket"] == "4건+")
        | (df["review_bucket"].isin(["리뷰없음", "1~2점"]))
    )
    rare_pool = df[rare_mask & ~df["customer_unique_id"].isin(rep_sample["customer_unique_id"])]
    rare_sample = rare_pool.sample(n=min(rare_n, len(rare_pool)), random_state=seed)

    combined = pd.concat([rep_sample, rare_sample]).drop_duplicates("customer_unique_id")
    if len(combined) > target_customers:
        combined = combined.sample(n=target_customers, random_state=seed)
    return combined


def cascade_copy(source_db: str, target_db: str, customer_unique_ids: list[str]) -> dict[str, int]:
    """샘플된 customer_unique_id를 기준으로 부모->자식 순서로 계단식 복사한다."""
    target_engine = _engine(target_db)
    id_frame = pd.DataFrame({"customer_unique_id": customer_unique_ids})
    with target_engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{target_db}`.`{_STAGING_TABLE}`"))
    id_frame.to_sql(_STAGING_TABLE, target_engine, if_exists="replace", index=False)

    engine = _engine()  # 서버 레벨 연결 — 크로스 DB 쿼리용
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                INSERT INTO `{target_db}`.customers
                SELECT c.* FROM `{source_db}`.customers c
                JOIN `{target_db}`.`{_STAGING_TABLE}` s
                  ON s.customer_unique_id = c.customer_unique_id
                """
            )
        )
        counts["customers"] = conn.execute(
            text(f"SELECT COUNT(*) FROM `{target_db}`.customers")
        ).scalar()

        conn.execute(
            text(
                f"""
                INSERT INTO `{target_db}`.orders
                SELECT o.* FROM `{source_db}`.orders o
                JOIN `{target_db}`.customers c ON c.customer_id = o.customer_id
                """
            )
        )
        counts["orders"] = conn.execute(text(f"SELECT COUNT(*) FROM `{target_db}`.orders")).scalar()

        for child in ("order_items", "order_payments", "order_reviews"):
            conn.execute(
                text(
                    f"""
                    INSERT INTO `{target_db}`.{child}
                    SELECT t.* FROM `{source_db}`.{child} t
                    JOIN `{target_db}`.orders o ON o.order_id = t.order_id
                    """
                )
            )
            counts[child] = conn.execute(text(f"SELECT COUNT(*) FROM `{target_db}`.{child}")).scalar()

        for full_table in ("products", "sellers", "product_category_name_translation", "geolocation"):
            conn.execute(
                text(f"INSERT INTO `{target_db}`.{full_table} SELECT * FROM `{source_db}`.{full_table}")
            )
            counts[full_table] = conn.execute(
                text(f"SELECT COUNT(*) FROM `{target_db}`.{full_table}")
            ).scalar()

        conn.execute(text(f"DROP TABLE IF EXISTS `{target_db}`.`{_STAGING_TABLE}`"))

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Olist 원본 DB에서 customer_unique_id 기준 층화 샘플 DB를 만든다."
    )
    parser.add_argument("--source-db", default=os.getenv("MYSQL_DATABASE", "olist"))
    parser.add_argument("--target-db", default="olist_sampling")
    parser.add_argument("--target-customers", type=int, default=10000)
    parser.add_argument("--representative-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[1/5] {args.source_db} 연결, 고객 프로필 집계 중...")
    source_engine = _engine(args.source_db)
    profile = load_customer_profile(source_engine)
    # MySQL은 ORDER BY 없이는 반환 순서를 보장하지 않는다 — 그런데 아래 rank(method="first")의
    # 동점 처리와 df.sample(random_state=seed)는 DataFrame 행 순서에 의존한다. customer_unique_id로
    # 정렬해 순서를 고정해야, 팀원 각자 로컬 MySQL에서 돌려도 항상 같은 샘플이 나온다.
    profile = profile.sort_values("customer_unique_id").reset_index(drop=True)
    print(f"      전체 고객(customer_unique_id): {len(profile):,}명")

    print("[2/5] 층화 버킷 계산 중 (주문횟수 x 금액 x 리뷰 x 분기)...")
    profile = build_strata(profile)

    rep_pct = args.representative_ratio
    print(
        f"[3/5] {args.target_customers:,}명 샘플링 중 "
        f"(대표성 {rep_pct:.0%} / 희귀케이스 {1 - rep_pct:.0%})..."
    )
    sample = select_sample(profile, args.target_customers, rep_pct, args.seed)
    print(f"      실제 뽑힌 고객 수: {len(sample):,}명")

    print(f"[4/5] {args.target_db} 스키마 생성 중...")
    create_target_schema(args.target_db)

    print("[5/5] 계단식 복사 중 (customers -> orders -> items/payments/reviews, 나머지는 전체 복사)...")
    counts = cascade_copy(args.source_db, args.target_db, sample["customer_unique_id"].tolist())

    print("\n=== 완료 ===")
    for table, count in counts.items():
        print(f"  {table}: {count:,}행")
    print(f"\nseed={args.seed} (같은 seed로 재실행하면 동일한 샘플 재현 가능)")


if __name__ == "__main__":
    main()
