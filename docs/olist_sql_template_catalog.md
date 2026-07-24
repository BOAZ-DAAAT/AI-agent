# Olist 결정론적 SQL 템플릿 카탈로그

이 문서는 Olist 고정 스키마에서 사용하는 결정론적 SQL 템플릿 25개의 계약을 정리합니다. 레지스트리의 기준 구현은 `agents/sql/olist_templates.py`, SQL 본문은 `agents/sql/olist_template_sql.py`입니다.

## 공통 라우팅 규칙

- `query`는 `route_kind=simple`, `sql_type=select`를 사용합니다.
- `mart`는 `route_kind=comprehensive`, `sql_type=create_table_as`를 사용하며 단일 `CREATE TABLE analytics.olist_<template_id> AS SELECT` 문만 생성합니다.
- 질문에서 구조화한 의도·지표·차원과 필수 스키마를 정확히 하나의 템플릿이 충족할 때만 결정론적 경로를 선택합니다.
- 템플릿을 특정할 수 없거나 둘 이상이 겹치는 경우, 필수 테이블·컬럼이 없거나 지원하지 않는 파라미터가 있는 경우 기존 semantic/LLM 계획 경로로 이동합니다.
- 결정론적 템플릿이 선택된 뒤 발생한 구성·사전 검증·실행 오류는 semantic SQL로 조용히 대체하지 않고 기존 오류 계약으로 반환합니다.
- 날짜는 명시적인 `YYYY-MM-DD` 또는 단일 연도만, 주문 상태는 Olist 상태 허용 목록만, 지역은 두 글자 주 코드만, Top N은 1~100만 허용합니다. 상대 기간과 임의 조건은 fallback합니다.

## Query 템플릿 15개

| ID | 질문 유형 / grain | 필수 스키마 | 허용 파라미터 | 출력 컬럼 |
|---|---|---|---|---|
| `monthly_sales_orders` | 월별 매출·주문 / 구매 월 | `orders(order_id, order_purchase_timestamp, order_status)`, `order_items(order_id, price)` | 기간, 주문 상태 | `purchase_month`, `product_sales`, `order_count` |
| `daily_sales_orders` | 일별 매출·주문 / 구매 일 | `orders`, `order_items`의 위 컬럼 | 기간, 주문 상태 | `purchase_date`, `product_sales`, `order_count` |
| `order_status_distribution` | 주문 상태 분포 / 주문 상태 | `orders(order_id, order_status, order_purchase_timestamp)` | 기간, Top N | `order_status`, `order_count`, `order_percentage` |
| `category_sales` | 카테고리 매출 / 상품 카테고리 | `order_items(order_id, product_id, price)`, `products(product_id, product_category_name)`, 번역 테이블, `orders` | 기간, 주문 상태, Top N | `category_name`, `product_sales`, `item_count`, `order_count` |
| `review_score_distribution` | 리뷰 평점 분포 / 1~5점 | `order_reviews(review_id, order_id, review_score)`, `orders` | 기간, 주문 상태 | `review_score`, `review_count`, `review_percentage` |
| `payment_method_summary` | 결제 수단 요약 / 결제 수단 | `order_payments(order_id, payment_type, payment_value)`, `orders` | 기간, 주문 상태, Top N | `payment_type`, `order_count`, `payment_count`, `total_payment_value` |
| `customer_state_sales` | 고객 주별 매출 / 고객 주 | `customers(customer_id, customer_state)`, `orders`, `order_items` | 기간, 주문 상태, 고객 주, Top N | `customer_state`, `order_count`, `product_sales` |
| `seller_state_sales` | 판매자 주별 매출 / 판매자 주 | `sellers(seller_id, seller_state)`, `order_items`, `orders` | 기간, 주문 상태, 판매자 주, Top N | `seller_state`, `order_count`, `item_count`, `product_sales` |
| `seller_performance` | 판매자 성과 / 판매자 | `sellers`, `order_items(order_id, seller_id, price, freight_value)`, `orders` | 기간, 주문 상태, 판매자 주, Top N | `seller_id`, `seller_state`, `order_count`, `item_count`, `product_sales`, `freight_value` |
| `delivery_delay_summary` | 배송 지연·소요일 / 배송 상태 | 배송 날짜 컬럼을 포함한 `orders` | 기간, 주문 상태 | `delivery_status`, `order_count`, `avg_delivery_days`, `avg_delay_days` |
| `category_review_summary` | 카테고리 리뷰 / 상품 카테고리 | `order_items`, `products`, 번역 테이블, `order_reviews`, `orders` | 기간, 주문 상태, Top N | `category_name`, `reviewed_order_count`, `avg_review_score` |
| `payment_installment_summary` | 할부 개월 요약 / 할부 개월 | `order_payments(order_id, payment_installments, payment_value)`, `orders` | 기간, 주문 상태, Top N | `payment_installments`, `order_count`, `payment_count`, `total_payment_value`, `avg_payment_value` |
| `freight_cost_summary` | 배송비 요약 / 상품 카테고리 | `order_items(order_id, product_id, freight_value)`, `products`, 번역 테이블, `orders` | 기간, 주문 상태, Top N | `category_name`, `order_count`, `item_count`, `total_freight_value`, `avg_item_freight_value` |
| `basket_size_summary` | 장바구니 크기 / 주문당 상품 수 | `order_items(order_id, price)`, `orders` | 기간, 주문 상태, Top N | `item_count`, `order_count`, 주문금액 평균·최솟값·최댓값 |
| `repeat_customer_summary` | 신규·재구매 요약 / 고객 유형 | `customers(customer_id, customer_unique_id)`, `orders` | 기간, 주문 상태 | `customer_type`, `customer_count`, `avg_purchase_count`, `avg_customer_span_days` |

## Mart 템플릿 10개

| ID | 질문 유형 / grain | 필수 스키마 | 허용 파라미터 | 핵심 출력 |
|---|---|---|---|---|
| `customer_rfm` | 고객 RFM / 고객 고유 ID | `customers`, `orders`, `order_items` | 기간, 주문 상태 | recency, frequency, monetary, 첫·마지막 구매일 |
| `monthly_customer_cohort` | 월별 코호트 / 첫 구매 월×활동 월 | `customers`, `orders`, `order_items` | 기간, 주문 상태 | 코호트 월, 활동 월, 경과 월, 활성 고객, 주문, 매출 |
| `customer_repeat_behavior` | 고객 재구매 행동 / 고객 고유 ID | `customers`, `orders`, `order_items` | 기간, 주문 상태 | 구매·재구매 횟수, 평균 구매 간격, 누적 매출 |
| `order_delivery_performance` | 주문 배송 성과 / 주문 | 배송 날짜 컬럼을 포함한 `orders` | 기간, 주문 상태 | 예상·실제 배송일, 배송·지연 일수, 지연 여부 |
| `monthly_category_performance` | 월별 카테고리 성과 / 구매 월×카테고리 | `order_items`, `orders`, `products`, 번역 테이블 | 기간, 주문 상태 | 주문, 상품, 매출, 배송비 |
| `monthly_seller_performance` | 월별 판매자 성과 / 구매 월×판매자 | `order_items`, `orders`, `sellers`, `order_reviews` | 기간, 주문 상태, 판매자 주 | 주문, 상품, 매출, 배송비, 배송일, 리뷰 |
| `customer_seller_geo` | 고객·판매자 지역 / 고객 주×판매자 주 | `order_items`, `orders`, `customers`, `sellers` | 기간, 주문 상태, 고객 주, 판매자 주 | 주문, 매출, 배송비, 배송일 |
| `category_review_delivery` | 카테고리 리뷰·배송 / 카테고리 | `order_items`, `products`, 번역 테이블, `orders`, `order_reviews` | 기간, 주문 상태 | 리뷰, 배송일, 지연율, 매출, 배송비 |
| `payment_behavior` | 결제 행동 / 주문 | `order_payments`, `orders`, `customers` | 기간, 주문 상태 | 결제 수단 수, 최대 할부, 결제금액, 결제수단 목록 |
| `product_logistics` | 상품 물류 / 상품 | `products`, 번역 테이블, `order_items`, `orders` | 기간, 주문 상태 | 크기·무게, 주문·상품 수, 매출·배송비, 배송일·지연율 |

## 중복 집계 방지 원칙

주문금액, 결제, 리뷰처럼 서로 다른 grain의 사실을 직접 다대다 조인하지 않습니다. 주문금액은 주문별, 리뷰는 주문별, 판매자 성과는 주문×판매자별로 먼저 집계한 뒤 상위 grain으로 올립니다. 주문 수는 상품행에서 `COUNT(DISTINCT order_id)`로 계산하며, 고객은 반복 주문 분석에서 `customer_unique_id`를 사용합니다.
