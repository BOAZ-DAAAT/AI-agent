---
document_id: metric-definitions
doc_type: analysis_foundation
query_type: metric_definitions
title: Olist 지표 정의
language: ko
version: "1.0"
source_tables: [customers, orders, order_items, order_payments, order_reviews]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---

# Olist 지표 정의

## 검색 별칭

- 지표 정의, 계산식, 주문 수, 매출, 결제금액, 객단가, 지연율
- metric definition, formula, order count, sales, revenue, payment value, AOV, late rate

## 주문 및 금액 지표

- [default] 주문 수는 `COUNT(DISTINCT orders.order_id)`다.
- [default] 상품 행 수는 `COUNT(*)` from `order_items`다.
- [default] 상품 매출은 `SUM(order_items.price)`다.
- [default] 배송비는 `SUM(order_items.freight_value)`다.
- [default] 배송비 포함 상품금액은 `SUM(order_items.price + order_items.freight_value)`다.
- [default] 결제금액은 `SUM(order_payments.payment_value)`다.
- [default] 상품 기준 객단가는 `SUM(order_items.price) / COUNT(DISTINCT orders.order_id)`다.
- [avoid] 상품 매출과 결제금액을 더하거나 같은 의미로 취급하지 않는다.

## 고객·만족도·배송 지표

- [default] 고객 생애 분석의 고객 수는 `COUNT(DISTINCT customers.customer_unique_id)`다.
- [default] 고객 구매 빈도는 고객별 `COUNT(DISTINCT orders.order_id)`다.
- [default] 재구매 고객은 별도 정의가 없으면 서로 다른 주문이 2건 이상인 고객이다.
- [default] 평균 만족도는 `AVG(order_reviews.review_score)`이며, 부정 리뷰는 `review_score <= 2`, 긍정 리뷰는 `review_score >= 4`다.
- [default] 고객 배송 지연은 `orders.order_delivered_customer_date > orders.order_estimated_delivery_date`다.
- [default] 지연일은 실제·예상 고객 배송일 차이, 배송 소요기간은 구매일·실제 배송일 차이다.
- [avoid] 리뷰가 없는 주문을 0점으로 보지 않는다.

## 조작적 정의

- [prefer] 사용자가 지표·임계값·기간·모집단을 명시하면 schema에서 계산 가능한 한 기본 정의보다 우선한다.
- [ask_if_missing] 매출, 고객가치, 지연, 성과처럼 해석이 결과를 크게 바꿀 때만 확인하거나 가정을 명시한다.
