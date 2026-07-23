"""Olist 결정론적 query·mart SQL 본문."""

from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.shared.contracts import OlistTemplateId


def _draft(
    sql: str,
    *,
    source_tables: list[str],
    source_column_refs: list[str],
    output_columns: list[str],
    business_grain: str,
    reasoning: str,
    target_table: str | None = None,
) -> dict[str, Any]:
    return {
        "sql": sql,
        "sql_type": "create_table_as" if target_table else "select",
        "target_table": target_table,
        "source_tables": source_tables,
        "source_column_refs": source_column_refs,
        "derived_columns": output_columns,
        "output_columns": output_columns,
        "business_grain": business_grain,
        "postcheck_sql": (
            f"SELECT COUNT(*) AS row_count FROM {target_table}" if target_table else None
        ),
        "reasoning": reasoning,
    }


QUERY_DRAFTS: dict[OlistTemplateId, dict[str, Any]] = {
    OlistTemplateId.monthly_sales_orders: _draft(
        """WITH order_amounts AS (
    SELECT order_id, SUM(price) AS product_sales
    FROM order_items
    GROUP BY order_id
)
SELECT DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m-01') AS purchase_month,
       SUM(COALESCE(a.product_sales, 0)) AS product_sales,
       COUNT(DISTINCT o.order_id) AS order_count
FROM orders AS o
LEFT JOIN order_amounts AS a ON a.order_id = o.order_id
/*__FILTERS__*/
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m-01')
ORDER BY purchase_month
/*__LIMIT__*/;""",
        source_tables=["orders", "order_items"],
        source_column_refs=["orders.order_id", "orders.order_purchase_timestamp", "orders.order_status", "order_items.order_id", "order_items.price"],
        output_columns=["purchase_month", "product_sales", "order_count"],
        business_grain="구매 월",
        reasoning="주문별 상품금액을 먼저 집계해 조인 fan-out 없이 월별 매출과 주문 수를 계산합니다.",
    ),
    OlistTemplateId.daily_sales_orders: _draft(
        """WITH order_amounts AS (
    SELECT order_id, SUM(price) AS product_sales
    FROM order_items
    GROUP BY order_id
)
SELECT DATE(o.order_purchase_timestamp) AS purchase_date,
       SUM(COALESCE(a.product_sales, 0)) AS product_sales,
       COUNT(DISTINCT o.order_id) AS order_count
FROM orders AS o
LEFT JOIN order_amounts AS a ON a.order_id = o.order_id
/*__FILTERS__*/
GROUP BY DATE(o.order_purchase_timestamp)
ORDER BY purchase_date
/*__LIMIT__*/;""",
        source_tables=["orders", "order_items"],
        source_column_refs=["orders.order_id", "orders.order_purchase_timestamp", "orders.order_status", "order_items.order_id", "order_items.price"],
        output_columns=["purchase_date", "product_sales", "order_count"],
        business_grain="구매 일",
        reasoning="주문별 상품금액을 먼저 집계해 일별 매출과 서로 다른 주문 수를 계산합니다.",
    ),
    OlistTemplateId.order_status_distribution: _draft(
        """SELECT o.order_status,
       COUNT(DISTINCT o.order_id) AS order_count,
       ROUND(100.0 * COUNT(DISTINCT o.order_id) /
             NULLIF(SUM(COUNT(DISTINCT o.order_id)) OVER (), 0), 2) AS order_percentage
FROM orders AS o
/*__FILTERS__*/
GROUP BY o.order_status
ORDER BY order_count DESC, o.order_status
/*__LIMIT__*/;""",
        source_tables=["orders"],
        source_column_refs=["orders.order_id", "orders.order_status", "orders.order_purchase_timestamp"],
        output_columns=["order_status", "order_count", "order_percentage"],
        business_grain="주문 상태",
        reasoning="상태별 서로 다른 주문 수와 필터 적용 후 전체 대비 비율을 계산합니다.",
    ),
    OlistTemplateId.category_sales: _draft(
        """SELECT COALESCE(NULLIF(t.product_category_name_english, ''),
                NULLIF(p.product_category_name, ''), 'unknown') AS category_name,
       SUM(oi.price) AS product_sales,
       COUNT(*) AS item_count,
       COUNT(DISTINCT oi.order_id) AS order_count
FROM order_items AS oi
JOIN products AS p ON p.product_id = oi.product_id
LEFT JOIN product_category_name_translation AS t
  ON t.product_category_name = p.product_category_name
JOIN orders AS o ON o.order_id = oi.order_id
/*__FILTERS__*/
GROUP BY COALESCE(NULLIF(t.product_category_name_english, ''),
                  NULLIF(p.product_category_name, ''), 'unknown')
ORDER BY product_sales DESC, category_name
/*__LIMIT__*/;""",
        source_tables=["order_items", "products", "product_category_name_translation", "orders"],
        source_column_refs=["order_items.order_id", "order_items.product_id", "order_items.price", "products.product_id", "products.product_category_name", "product_category_name_translation.product_category_name", "product_category_name_translation.product_category_name_english", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_status"],
        output_columns=["category_name", "product_sales", "item_count", "order_count"],
        business_grain="상품 카테고리",
        reasoning="상품행 grain에서 매출을 합산하고 주문 수는 DISTINCT로 분리합니다.",
    ),
    OlistTemplateId.review_score_distribution: _draft(
        """WITH review_scores AS (
    SELECT 1 AS review_score UNION ALL SELECT 2 UNION ALL SELECT 3
    UNION ALL SELECT 4 UNION ALL SELECT 5
), filtered_reviews AS (
    SELECT DISTINCT r.review_id, r.review_score
    FROM order_reviews AS r
    JOIN orders AS o ON o.order_id = r.order_id
    /*__FILTERS__*/
)
SELECT s.review_score,
       COUNT(DISTINCT r.review_id) AS review_count,
       ROUND(100.0 * COUNT(DISTINCT r.review_id) /
             NULLIF((SELECT COUNT(DISTINCT review_id) FROM filtered_reviews), 0), 2) AS review_percentage
FROM review_scores AS s
LEFT JOIN filtered_reviews AS r ON r.review_score = s.review_score
GROUP BY s.review_score
ORDER BY s.review_score
/*__LIMIT__*/;""",
        source_tables=["order_reviews", "orders"],
        source_column_refs=["order_reviews.review_id", "order_reviews.order_id", "order_reviews.review_score", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_status"],
        output_columns=["review_score", "review_count", "review_percentage"],
        business_grain="리뷰 점수(1~5점)",
        reasoning="필터 적용 리뷰를 중복 제거하고 1~5점 축을 보존해 분포를 계산합니다.",
    ),
    OlistTemplateId.payment_method_summary: _draft(
        """SELECT p.payment_type,
       COUNT(DISTINCT p.order_id) AS order_count,
       COUNT(*) AS payment_count,
       SUM(p.payment_value) AS total_payment_value
FROM order_payments AS p
JOIN orders AS o ON o.order_id = p.order_id
/*__FILTERS__*/
GROUP BY p.payment_type
ORDER BY total_payment_value DESC, p.payment_type
/*__LIMIT__*/;""",
        source_tables=["order_payments", "orders"],
        source_column_refs=["order_payments.order_id", "order_payments.payment_type", "order_payments.payment_value", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_status"],
        output_columns=["payment_type", "order_count", "payment_count", "total_payment_value"],
        business_grain="결제 수단",
        reasoning="결제행 수와 서로 다른 주문 수를 분리해 결제 수단별 금액을 집계합니다.",
    ),
    OlistTemplateId.customer_state_sales: _draft(
        """WITH order_amounts AS (
    SELECT order_id, SUM(price) AS product_sales FROM order_items GROUP BY order_id
)
SELECT c.customer_state,
       COUNT(DISTINCT o.order_id) AS order_count,
       SUM(COALESCE(a.product_sales, 0)) AS product_sales
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
LEFT JOIN order_amounts AS a ON a.order_id = o.order_id
/*__FILTERS__*/
GROUP BY c.customer_state
ORDER BY product_sales DESC, c.customer_state
/*__LIMIT__*/;""",
        source_tables=["orders", "customers", "order_items"],
        source_column_refs=["orders.order_id", "orders.customer_id", "orders.order_purchase_timestamp", "orders.order_status", "customers.customer_id", "customers.customer_state", "order_items.order_id", "order_items.price"],
        output_columns=["customer_state", "order_count", "product_sales"],
        business_grain="고객 주",
        reasoning="주문금액을 선집계한 뒤 고객 주별 주문과 매출을 계산합니다.",
    ),
    OlistTemplateId.seller_state_sales: _draft(
        """SELECT s.seller_state,
       COUNT(DISTINCT oi.order_id) AS order_count,
       COUNT(*) AS item_count,
       SUM(oi.price) AS product_sales
FROM order_items AS oi
JOIN sellers AS s ON s.seller_id = oi.seller_id
JOIN orders AS o ON o.order_id = oi.order_id
/*__FILTERS__*/
GROUP BY s.seller_state
ORDER BY product_sales DESC, s.seller_state
/*__LIMIT__*/;""",
        source_tables=["order_items", "sellers", "orders"],
        source_column_refs=["order_items.order_id", "order_items.seller_id", "order_items.price", "sellers.seller_id", "sellers.seller_state", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_status"],
        output_columns=["seller_state", "order_count", "item_count", "product_sales"],
        business_grain="판매자 주",
        reasoning="상품행 매출과 서로 다른 주문 수를 판매자 주별로 분리 집계합니다.",
    ),
    OlistTemplateId.seller_performance: _draft(
        """SELECT oi.seller_id, s.seller_state,
       COUNT(DISTINCT oi.order_id) AS order_count,
       COUNT(*) AS item_count,
       SUM(oi.price) AS product_sales,
       SUM(oi.freight_value) AS freight_value
FROM order_items AS oi
JOIN sellers AS s ON s.seller_id = oi.seller_id
JOIN orders AS o ON o.order_id = oi.order_id
/*__FILTERS__*/
GROUP BY oi.seller_id, s.seller_state
ORDER BY product_sales DESC, oi.seller_id
/*__LIMIT__*/;""",
        source_tables=["order_items", "sellers", "orders"],
        source_column_refs=["order_items.order_id", "order_items.seller_id", "order_items.price", "order_items.freight_value", "sellers.seller_id", "sellers.seller_state", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_status"],
        output_columns=["seller_id", "seller_state", "order_count", "item_count", "product_sales", "freight_value"],
        business_grain="판매자",
        reasoning="판매자별 주문·상품·매출·배송비를 상품행과 주문 grain에 맞게 집계합니다.",
    ),
    OlistTemplateId.delivery_delay_summary: _draft(
        """SELECT CASE
           WHEN o.order_delivered_customer_date IS NULL THEN 'not_delivered'
           WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 'delayed'
           ELSE 'on_time'
       END AS delivery_status,
       COUNT(DISTINCT o.order_id) AS order_count,
       AVG(DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp)) AS avg_delivery_days,
       AVG(DATEDIFF(o.order_delivered_customer_date, o.order_estimated_delivery_date)) AS avg_delay_days
FROM orders AS o
/*__FILTERS__*/
GROUP BY CASE
           WHEN o.order_delivered_customer_date IS NULL THEN 'not_delivered'
           WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 'delayed'
           ELSE 'on_time'
         END
ORDER BY delivery_status
/*__LIMIT__*/;""",
        source_tables=["orders"],
        source_column_refs=["orders.order_id", "orders.order_status", "orders.order_purchase_timestamp", "orders.order_delivered_customer_date", "orders.order_estimated_delivery_date"],
        output_columns=["delivery_status", "order_count", "avg_delivery_days", "avg_delay_days"],
        business_grain="배송 상태",
        reasoning="주문 grain에서 실제·예상 배송일 차이와 구매 후 배송 소요일을 계산합니다.",
    ),
    OlistTemplateId.category_review_summary: _draft(
        """WITH order_reviews_one AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM order_reviews GROUP BY order_id
), order_categories AS (
    SELECT DISTINCT oi.order_id,
           COALESCE(NULLIF(t.product_category_name_english, ''),
                    NULLIF(p.product_category_name, ''), 'unknown') AS category_name
    FROM order_items AS oi
    JOIN products AS p ON p.product_id = oi.product_id
    LEFT JOIN product_category_name_translation AS t
      ON t.product_category_name = p.product_category_name
)
SELECT oc.category_name,
       COUNT(DISTINCT oc.order_id) AS reviewed_order_count,
       AVG(r.review_score) AS avg_review_score
FROM order_categories AS oc
JOIN order_reviews_one AS r ON r.order_id = oc.order_id
JOIN orders AS o ON o.order_id = oc.order_id
/*__FILTERS__*/
GROUP BY oc.category_name
ORDER BY avg_review_score DESC, oc.category_name
/*__LIMIT__*/;""",
        source_tables=["order_items", "products", "product_category_name_translation", "order_reviews", "orders"],
        source_column_refs=["order_items.order_id", "order_items.product_id", "products.product_id", "products.product_category_name", "product_category_name_translation.product_category_name", "product_category_name_translation.product_category_name_english", "order_reviews.order_id", "order_reviews.review_score", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_status"],
        output_columns=["category_name", "reviewed_order_count", "avg_review_score"],
        business_grain="상품 카테고리",
        reasoning="리뷰를 주문별로 먼저 집계하고 주문-카테고리를 중복 제거해 리뷰 fan-out을 방지합니다.",
    ),
    OlistTemplateId.payment_installment_summary: _draft(
        """SELECT p.payment_installments,
       COUNT(DISTINCT p.order_id) AS order_count,
       COUNT(*) AS payment_count,
       SUM(p.payment_value) AS total_payment_value,
       AVG(p.payment_value) AS avg_payment_value
FROM order_payments AS p
JOIN orders AS o ON o.order_id = p.order_id
/*__FILTERS__*/
GROUP BY p.payment_installments
ORDER BY p.payment_installments
/*__LIMIT__*/;""",
        source_tables=["order_payments", "orders"],
        source_column_refs=["order_payments.order_id", "order_payments.payment_installments", "order_payments.payment_value", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_status"],
        output_columns=["payment_installments", "order_count", "payment_count", "total_payment_value", "avg_payment_value"],
        business_grain="할부 개월",
        reasoning="할부 개월별 주문 수와 결제행 수를 구분하고 결제금액을 집계합니다.",
    ),
    OlistTemplateId.freight_cost_summary: _draft(
        """SELECT COALESCE(NULLIF(t.product_category_name_english, ''),
                NULLIF(p.product_category_name, ''), 'unknown') AS category_name,
       COUNT(DISTINCT oi.order_id) AS order_count,
       COUNT(*) AS item_count,
       SUM(oi.freight_value) AS total_freight_value,
       AVG(oi.freight_value) AS avg_item_freight_value
FROM order_items AS oi
JOIN products AS p ON p.product_id = oi.product_id
LEFT JOIN product_category_name_translation AS t
  ON t.product_category_name = p.product_category_name
JOIN orders AS o ON o.order_id = oi.order_id
/*__FILTERS__*/
GROUP BY COALESCE(NULLIF(t.product_category_name_english, ''),
                  NULLIF(p.product_category_name, ''), 'unknown')
ORDER BY total_freight_value DESC, category_name
/*__LIMIT__*/;""",
        source_tables=["order_items", "products", "product_category_name_translation", "orders"],
        source_column_refs=["order_items.order_id", "order_items.product_id", "order_items.freight_value", "products.product_id", "products.product_category_name", "product_category_name_translation.product_category_name", "product_category_name_translation.product_category_name_english", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_status"],
        output_columns=["category_name", "order_count", "item_count", "total_freight_value", "avg_item_freight_value"],
        business_grain="상품 카테고리",
        reasoning="배송비가 상품행에 기록된 점을 명시해 카테고리별 합계와 상품행 평균을 계산합니다.",
    ),
    OlistTemplateId.basket_size_summary: _draft(
        """WITH order_baskets AS (
    SELECT oi.order_id, COUNT(*) AS item_count,
           SUM(oi.price) AS order_product_value
    FROM order_items AS oi
    GROUP BY oi.order_id
)
SELECT b.item_count,
       COUNT(*) AS order_count,
       AVG(b.order_product_value) AS avg_order_product_value,
       MIN(b.order_product_value) AS min_order_product_value,
       MAX(b.order_product_value) AS max_order_product_value
FROM order_baskets AS b
JOIN orders AS o ON o.order_id = b.order_id
/*__FILTERS__*/
GROUP BY b.item_count
ORDER BY b.item_count
/*__LIMIT__*/;""",
        source_tables=["order_items", "orders"],
        source_column_refs=["order_items.order_id", "order_items.price", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_status"],
        output_columns=["item_count", "order_count", "avg_order_product_value", "min_order_product_value", "max_order_product_value"],
        business_grain="주문당 상품 수",
        reasoning="주문별 장바구니를 먼저 만든 뒤 상품 수별 주문금액 분포를 요약합니다.",
    ),
    OlistTemplateId.repeat_customer_summary: _draft(
        """WITH customer_orders AS (
    SELECT c.customer_unique_id,
           COUNT(DISTINCT o.order_id) AS purchase_count,
           MIN(o.order_purchase_timestamp) AS first_purchase_at,
           MAX(o.order_purchase_timestamp) AS last_purchase_at
    FROM customers AS c
    JOIN orders AS o ON o.customer_id = c.customer_id
    /*__FILTERS__*/
    GROUP BY c.customer_unique_id
)
SELECT CASE WHEN purchase_count = 1 THEN 'new' ELSE 'repeat' END AS customer_type,
       COUNT(*) AS customer_count,
       AVG(purchase_count) AS avg_purchase_count,
       AVG(DATEDIFF(last_purchase_at, first_purchase_at)) AS avg_customer_span_days
FROM customer_orders
GROUP BY CASE WHEN purchase_count = 1 THEN 'new' ELSE 'repeat' END
ORDER BY customer_type
/*__LIMIT__*/;""",
        source_tables=["customers", "orders"],
        source_column_refs=["customers.customer_id", "customers.customer_unique_id", "orders.order_id", "orders.customer_id", "orders.order_purchase_timestamp", "orders.order_status"],
        output_columns=["customer_type", "customer_count", "avg_purchase_count", "avg_customer_span_days"],
        business_grain="신규·재구매 고객 유형",
        reasoning="customer_unique_id 기준 구매 빈도를 먼저 계산해 신규·재구매 고객을 구분합니다.",
    ),
}


def _mart(
    template_id: OlistTemplateId,
    select_sql: str,
    *,
    source_tables: list[str],
    source_column_refs: list[str],
    output_columns: list[str],
    business_grain: str,
    reasoning: str,
) -> dict[str, Any]:
    target = f"analytics.olist_{template_id.value}"
    return _draft(
        f"CREATE TABLE {target} AS\n{select_sql.strip().rstrip(';')}\n/*__LIMIT__*/;",
        target_table=target,
        source_tables=source_tables,
        source_column_refs=source_column_refs,
        output_columns=output_columns,
        business_grain=business_grain,
        reasoning=reasoning,
    )


MART_DRAFTS: dict[OlistTemplateId, dict[str, Any]] = {
    OlistTemplateId.customer_rfm: _mart(
        OlistTemplateId.customer_rfm,
        """WITH order_amounts AS (
    SELECT order_id, SUM(price) AS product_sales FROM order_items GROUP BY order_id
), customer_orders AS (
    SELECT c.customer_unique_id, o.order_id, o.order_purchase_timestamp,
           COALESCE(a.product_sales, 0) AS product_sales
    FROM customers AS c
    JOIN orders AS o ON o.customer_id = c.customer_id
    LEFT JOIN order_amounts AS a ON a.order_id = o.order_id
    /*__FILTERS__*/
), anchor AS (SELECT MAX(order_purchase_timestamp) AS max_purchase_at FROM customer_orders)
SELECT co.customer_unique_id,
       DATEDIFF(a.max_purchase_at, MAX(co.order_purchase_timestamp)) AS recency_days,
       COUNT(DISTINCT co.order_id) AS frequency,
       SUM(co.product_sales) AS monetary_value,
       MIN(co.order_purchase_timestamp) AS first_purchase_at,
       MAX(co.order_purchase_timestamp) AS last_purchase_at
FROM customer_orders AS co CROSS JOIN anchor AS a
GROUP BY co.customer_unique_id, a.max_purchase_at""",
        source_tables=["customers", "orders", "order_items"],
        source_column_refs=["customers.customer_id", "customers.customer_unique_id", "orders.order_id", "orders.customer_id", "orders.order_purchase_timestamp", "orders.order_status", "order_items.order_id", "order_items.price"],
        output_columns=["customer_unique_id", "recency_days", "frequency", "monetary_value", "first_purchase_at", "last_purchase_at"],
        business_grain="고객 고유 ID",
        reasoning="주문금액을 선집계한 뒤 고객별 RFM을 계산해 결제·상품 조인 fan-out을 차단합니다.",
    ),
    OlistTemplateId.monthly_customer_cohort: _mart(
        OlistTemplateId.monthly_customer_cohort,
        """WITH order_amounts AS (
    SELECT order_id, SUM(price) AS product_sales FROM order_items GROUP BY order_id
), customer_orders AS (
    SELECT c.customer_unique_id, o.order_id, o.order_purchase_timestamp,
           COALESCE(a.product_sales, 0) AS product_sales
    FROM customers AS c JOIN orders AS o ON o.customer_id = c.customer_id
    LEFT JOIN order_amounts AS a ON a.order_id = o.order_id
    /*__FILTERS__*/
), first_purchase AS (
    SELECT customer_unique_id, MIN(order_purchase_timestamp) AS first_purchase_at
    FROM customer_orders GROUP BY customer_unique_id
)
SELECT DATE_FORMAT(f.first_purchase_at, '%Y-%m-01') AS cohort_month,
       DATE_FORMAT(co.order_purchase_timestamp, '%Y-%m-01') AS activity_month,
       TIMESTAMPDIFF(MONTH, DATE_FORMAT(f.first_purchase_at, '%Y-%m-01'),
                    DATE_FORMAT(co.order_purchase_timestamp, '%Y-%m-01')) AS cohort_index,
       COUNT(DISTINCT co.customer_unique_id) AS active_customers,
       COUNT(DISTINCT co.order_id) AS order_count,
       SUM(co.product_sales) AS product_sales
FROM customer_orders AS co JOIN first_purchase AS f USING (customer_unique_id)
GROUP BY cohort_month, activity_month, cohort_index""",
        source_tables=["customers", "orders", "order_items"],
        source_column_refs=["customers.customer_id", "customers.customer_unique_id", "orders.order_id", "orders.customer_id", "orders.order_purchase_timestamp", "orders.order_status", "order_items.order_id", "order_items.price"],
        output_columns=["cohort_month", "activity_month", "cohort_index", "active_customers", "order_count", "product_sales"],
        business_grain="첫 구매 월 × 활동 월",
        reasoning="고객 첫 구매 월과 활동 월을 분리해 코호트별 고객·주문·매출을 계산합니다.",
    ),
    OlistTemplateId.customer_repeat_behavior: _mart(
        OlistTemplateId.customer_repeat_behavior,
        """WITH order_amounts AS (
    SELECT order_id, SUM(price) AS product_sales FROM order_items GROUP BY order_id
), customer_orders AS (
    SELECT c.customer_unique_id, o.order_id, o.order_purchase_timestamp,
           COALESCE(a.product_sales, 0) AS product_sales,
           LAG(o.order_purchase_timestamp) OVER (
               PARTITION BY c.customer_unique_id ORDER BY o.order_purchase_timestamp, o.order_id
           ) AS previous_purchase_at
    FROM customers AS c JOIN orders AS o ON o.customer_id = c.customer_id
    LEFT JOIN order_amounts AS a ON a.order_id = o.order_id
    /*__FILTERS__*/
)
SELECT customer_unique_id, COUNT(DISTINCT order_id) AS purchase_count,
       GREATEST(COUNT(DISTINCT order_id) - 1, 0) AS repeat_purchase_count,
       AVG(DATEDIFF(order_purchase_timestamp, previous_purchase_at)) AS avg_days_between_purchases,
       SUM(product_sales) AS cumulative_product_sales,
       MIN(order_purchase_timestamp) AS first_purchase_at,
       MAX(order_purchase_timestamp) AS last_purchase_at
FROM customer_orders GROUP BY customer_unique_id""",
        source_tables=["customers", "orders", "order_items"],
        source_column_refs=["customers.customer_id", "customers.customer_unique_id", "orders.order_id", "orders.customer_id", "orders.order_purchase_timestamp", "orders.order_status", "order_items.order_id", "order_items.price"],
        output_columns=["customer_unique_id", "purchase_count", "repeat_purchase_count", "avg_days_between_purchases", "cumulative_product_sales", "first_purchase_at", "last_purchase_at"],
        business_grain="고객 고유 ID",
        reasoning="고객 주문 시퀀스에서 이전 구매일을 구한 뒤 재구매 횟수·주기·누적금액을 계산합니다.",
    ),
    OlistTemplateId.order_delivery_performance: _mart(
        OlistTemplateId.order_delivery_performance,
        """SELECT o.order_id, o.customer_id, o.order_status, o.order_purchase_timestamp,
       o.order_approved_at, o.order_delivered_carrier_date,
       o.order_delivered_customer_date, o.order_estimated_delivery_date,
       DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp) AS delivery_days,
       DATEDIFF(o.order_delivered_customer_date, o.order_estimated_delivery_date) AS delay_days,
       CASE WHEN o.order_delivered_customer_date IS NULL THEN NULL
            WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END AS is_delayed
FROM orders AS o
/*__FILTERS__*/""",
        source_tables=["orders"],
        source_column_refs=["orders.order_id", "orders.customer_id", "orders.order_status", "orders.order_purchase_timestamp", "orders.order_approved_at", "orders.order_delivered_carrier_date", "orders.order_delivered_customer_date", "orders.order_estimated_delivery_date"],
        output_columns=["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", "delivery_days", "delay_days", "is_delayed"],
        business_grain="주문",
        reasoning="주문 한 행을 보존하면서 예상·실제 배송일과 파생 배송 성과를 추가합니다.",
    ),
    OlistTemplateId.monthly_category_performance: _mart(
        OlistTemplateId.monthly_category_performance,
        """SELECT DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m-01') AS purchase_month,
       COALESCE(NULLIF(t.product_category_name_english, ''),
                NULLIF(p.product_category_name, ''), 'unknown') AS category_name,
       COUNT(DISTINCT oi.order_id) AS order_count, COUNT(*) AS item_count,
       SUM(oi.price) AS product_sales, SUM(oi.freight_value) AS freight_value
FROM order_items AS oi JOIN orders AS o ON o.order_id = oi.order_id
JOIN products AS p ON p.product_id = oi.product_id
LEFT JOIN product_category_name_translation AS t
  ON t.product_category_name = p.product_category_name
/*__FILTERS__*/
GROUP BY purchase_month, category_name""",
        source_tables=["order_items", "orders", "products", "product_category_name_translation"],
        source_column_refs=["order_items.order_id", "order_items.product_id", "order_items.price", "order_items.freight_value", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_status", "products.product_id", "products.product_category_name", "product_category_name_translation.product_category_name", "product_category_name_translation.product_category_name_english"],
        output_columns=["purchase_month", "category_name", "order_count", "item_count", "product_sales", "freight_value"],
        business_grain="구매 월 × 상품 카테고리",
        reasoning="상품행 grain에서 월×카테고리 매출·배송비를 집계하고 주문 수는 DISTINCT 처리합니다.",
    ),
    OlistTemplateId.monthly_seller_performance: _mart(
        OlistTemplateId.monthly_seller_performance,
        """WITH order_review AS (
    SELECT order_id, AVG(review_score) AS review_score FROM order_reviews GROUP BY order_id
), seller_orders AS (
    SELECT oi.order_id, oi.seller_id, SUM(oi.price) AS product_sales,
           SUM(oi.freight_value) AS freight_value, COUNT(*) AS item_count
    FROM order_items AS oi GROUP BY oi.order_id, oi.seller_id
)
SELECT DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m-01') AS purchase_month,
       so.seller_id, s.seller_state, COUNT(DISTINCT so.order_id) AS order_count,
       SUM(so.item_count) AS item_count, SUM(so.product_sales) AS product_sales,
       SUM(so.freight_value) AS freight_value,
       AVG(DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp)) AS avg_delivery_days,
       AVG(r.review_score) AS avg_review_score
FROM seller_orders AS so JOIN orders AS o ON o.order_id = so.order_id
JOIN sellers AS s ON s.seller_id = so.seller_id
LEFT JOIN order_review AS r ON r.order_id = so.order_id
/*__FILTERS__*/
GROUP BY purchase_month, so.seller_id, s.seller_state""",
        source_tables=["order_items", "orders", "sellers", "order_reviews"],
        source_column_refs=["order_items.order_id", "order_items.seller_id", "order_items.price", "order_items.freight_value", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_delivered_customer_date", "orders.order_status", "sellers.seller_id", "sellers.seller_state", "order_reviews.order_id", "order_reviews.review_score"],
        output_columns=["purchase_month", "seller_id", "seller_state", "order_count", "item_count", "product_sales", "freight_value", "avg_delivery_days", "avg_review_score"],
        business_grain="구매 월 × 판매자",
        reasoning="주문×판매자와 주문 리뷰를 각각 선집계해 월별 판매자 성과의 fan-out을 방지합니다.",
    ),
    OlistTemplateId.customer_seller_geo: _mart(
        OlistTemplateId.customer_seller_geo,
        """WITH order_seller AS (
    SELECT oi.order_id, oi.seller_id, SUM(oi.price) AS product_sales,
           SUM(oi.freight_value) AS freight_value
    FROM order_items AS oi GROUP BY oi.order_id, oi.seller_id
)
SELECT c.customer_state, s.seller_state,
       COUNT(DISTINCT os.order_id) AS order_count,
       SUM(os.product_sales) AS product_sales,
       SUM(os.freight_value) AS freight_value,
       AVG(DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp)) AS avg_delivery_days
FROM order_seller AS os JOIN orders AS o ON o.order_id = os.order_id
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN sellers AS s ON s.seller_id = os.seller_id
/*__FILTERS__*/
GROUP BY c.customer_state, s.seller_state""",
        source_tables=["order_items", "orders", "customers", "sellers"],
        source_column_refs=["order_items.order_id", "order_items.seller_id", "order_items.price", "order_items.freight_value", "orders.order_id", "orders.customer_id", "orders.order_purchase_timestamp", "orders.order_delivered_customer_date", "orders.order_status", "customers.customer_id", "customers.customer_state", "sellers.seller_id", "sellers.seller_state"],
        output_columns=["customer_state", "seller_state", "order_count", "product_sales", "freight_value", "avg_delivery_days"],
        business_grain="고객 주 × 판매자 주",
        reasoning="주문×판매자를 선집계한 뒤 고객·판매자 지역 쌍별 물류 성과를 계산합니다.",
    ),
    OlistTemplateId.category_review_delivery: _mart(
        OlistTemplateId.category_review_delivery,
        """WITH order_review AS (
    SELECT order_id, AVG(review_score) AS review_score FROM order_reviews GROUP BY order_id
), order_category AS (
    SELECT oi.order_id,
           COALESCE(NULLIF(t.product_category_name_english, ''),
                    NULLIF(p.product_category_name, ''), 'unknown') AS category_name,
           SUM(oi.price) AS product_sales, SUM(oi.freight_value) AS freight_value
    FROM order_items AS oi JOIN products AS p ON p.product_id = oi.product_id
    LEFT JOIN product_category_name_translation AS t
      ON t.product_category_name = p.product_category_name
    GROUP BY oi.order_id, category_name
)
SELECT oc.category_name, COUNT(DISTINCT oc.order_id) AS order_count,
       SUM(oc.product_sales) AS product_sales, SUM(oc.freight_value) AS freight_value,
       AVG(r.review_score) AS avg_review_score,
       AVG(DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp)) AS avg_delivery_days,
       AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END) AS delay_rate
FROM order_category AS oc JOIN orders AS o ON o.order_id = oc.order_id
LEFT JOIN order_review AS r ON r.order_id = oc.order_id
/*__FILTERS__*/
GROUP BY oc.category_name""",
        source_tables=["order_items", "products", "product_category_name_translation", "orders", "order_reviews"],
        source_column_refs=["order_items.order_id", "order_items.product_id", "order_items.price", "order_items.freight_value", "products.product_id", "products.product_category_name", "product_category_name_translation.product_category_name", "product_category_name_translation.product_category_name_english", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_delivered_customer_date", "orders.order_estimated_delivery_date", "orders.order_status", "order_reviews.order_id", "order_reviews.review_score"],
        output_columns=["category_name", "order_count", "product_sales", "freight_value", "avg_review_score", "avg_delivery_days", "delay_rate"],
        business_grain="상품 카테고리",
        reasoning="주문×카테고리와 주문 리뷰를 선집계해 카테고리 리뷰·배송 통합 지표를 계산합니다.",
    ),
    OlistTemplateId.payment_behavior: _mart(
        OlistTemplateId.payment_behavior,
        """WITH payment_order AS (
    SELECT order_id, COUNT(*) AS payment_count,
           COUNT(DISTINCT payment_type) AS payment_method_count,
           MAX(payment_installments) AS max_installments,
           SUM(payment_value) AS total_payment_value,
           GROUP_CONCAT(DISTINCT payment_type ORDER BY payment_type SEPARATOR ',') AS payment_types
    FROM order_payments GROUP BY order_id
)
SELECT o.order_id, c.customer_unique_id, o.order_purchase_timestamp,
       p.payment_count, p.payment_method_count, p.max_installments,
       p.total_payment_value, p.payment_types
FROM payment_order AS p JOIN orders AS o ON o.order_id = p.order_id
JOIN customers AS c ON c.customer_id = o.customer_id
/*__FILTERS__*/""",
        source_tables=["order_payments", "orders", "customers"],
        source_column_refs=["order_payments.order_id", "order_payments.payment_type", "order_payments.payment_installments", "order_payments.payment_value", "orders.order_id", "orders.customer_id", "orders.order_purchase_timestamp", "orders.order_status", "customers.customer_id", "customers.customer_unique_id"],
        output_columns=["order_id", "customer_unique_id", "order_purchase_timestamp", "payment_count", "payment_method_count", "max_installments", "total_payment_value", "payment_types"],
        business_grain="주문",
        reasoning="결제를 주문별로 먼저 집계해 결제 수단·할부·금액을 한 행에 보존합니다.",
    ),
    OlistTemplateId.product_logistics: _mart(
        OlistTemplateId.product_logistics,
        """WITH product_orders AS (
    SELECT oi.product_id, COUNT(DISTINCT oi.order_id) AS order_count,
           COUNT(*) AS item_count, SUM(oi.price) AS product_sales,
           SUM(oi.freight_value) AS freight_value,
           AVG(DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp)) AS avg_delivery_days,
           AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END) AS delay_rate
    FROM order_items AS oi JOIN orders AS o ON o.order_id = oi.order_id
    /*__FILTERS__*/
    GROUP BY oi.product_id
)
SELECT p.product_id,
       COALESCE(NULLIF(t.product_category_name_english, ''),
                NULLIF(p.product_category_name, ''), 'unknown') AS category_name,
       p.product_weight_g, p.product_length_cm, p.product_height_cm, p.product_width_cm,
       po.order_count, po.item_count, po.product_sales, po.freight_value,
       po.avg_delivery_days, po.delay_rate
FROM products AS p JOIN product_orders AS po ON po.product_id = p.product_id
LEFT JOIN product_category_name_translation AS t
  ON t.product_category_name = p.product_category_name""",
        source_tables=["products", "product_category_name_translation", "order_items", "orders"],
        source_column_refs=["products.product_id", "products.product_category_name", "products.product_weight_g", "products.product_length_cm", "products.product_height_cm", "products.product_width_cm", "product_category_name_translation.product_category_name", "product_category_name_translation.product_category_name_english", "order_items.product_id", "order_items.order_id", "order_items.price", "order_items.freight_value", "orders.order_id", "orders.order_purchase_timestamp", "orders.order_delivered_customer_date", "orders.order_estimated_delivery_date", "orders.order_status"],
        output_columns=["product_id", "category_name", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm", "order_count", "item_count", "product_sales", "freight_value", "avg_delivery_days", "delay_rate"],
        business_grain="상품",
        reasoning="상품별 주문 물류를 선집계하고 크기·무게·카테고리 속성을 결합합니다.",
    ),
}


OLIST_SQL_DRAFTS = {**QUERY_DRAFTS, **MART_DRAFTS}


__all__ = ["MART_DRAFTS", "OLIST_SQL_DRAFTS", "QUERY_DRAFTS"]
