---
document_id: payment-behavior
doc_type: analysis_query_rule
query_type: payment_behavior
title: Olist Payment Behavior Query Rules
language: ko
version: "1.0"
business_entities: [payments, orders, customers]
source_tables: [order_payments, orders, customers, order_items, products]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---
# 결제 행동 분석 규칙

## 정의

결제 수단, 할부, 결제 레코드 수, 결제 완료 주문 수, 결제금액을 분석한다. 결제 레코드 grain은 `order_payments.order_id`와 `order_payments.payment_sequential`의 조합이다.

## 검색 별칭

- 결제 행동, 결제 수단, 결제 방식, 할부, 결제 금액, 신용카드, boleto, 결제 유형
- payment behavior, payment method, installment, payment type, boleto, credit card, payment value

## 기본 지표

- [default] 결제 레코드 수는 `COUNT(*)` from `order_payments`다.
- [default] 결제 주문 수는 `COUNT(DISTINCT order_payments.order_id)`다.
- [default] 총 결제금액은 `SUM(order_payments.payment_value)`다.
- [default] 평균 할부 횟수는 `AVG(order_payments.payment_installments)`다.
- [prefer] 결제금액과 상품 매출은 서로 다른 금액 관점으로 유지한다.

## Grain과 조인

- [must] 결제는 `order_payments.order_id = orders.order_id`로 주문에 연결한다.
- [avoid] `payment_sequential`을 할부 횟수로 해석하지 않는다.
- [avoid] `order_payments`와 `order_items`를 직접 조인한 뒤 `payment_value`를 합산하지 않는다.
- [prefer] 카테고리·판매자·상품별 결제가 필요하면 먼저 결제 테이블을 주문 grain으로 집계한다.

## 시간·상태·기본 가정

- [default] `order_payments`에는 결제 시점이 없으므로 추세는 `orders.order_purchase_timestamp`를 사용한다.
- [default] 결제수단 분포는 별도 요청이 없으면 모든 상태를 포함한다. delivered/완료 주문 요청이면 해당 상태를 제한한다.
- [prefer] 결제수단 비율의 분모가 결제 레코드인지 서로 다른 주문인지 명시한다.

## 확인이 필요한 경우

- [ask_if_missing] 결제수단 점유율의 분모가 레코드 수인지 주문 수인지에 따라 해석이 달라지면 확인한다.
- [ask_if_missing] 주문 단위 결제금액을 판매자·카테고리에 배분하려면 배분 기준을 확인한다.
- 좋은 예: “결제수단별 서로 다른 주문 수와 총 결제금액을 보여줘.”
- 피해야 할 예: “상품 행에 결제금액을 붙여 카테고리별 결제금액을 합산해줘.”

## 관련 규칙

- `sales_orders`, `customer_value`, `regional_analysis`, `product_category`, `review_satisfaction`

## 사용하지 않는 경우

- [prefer] 상품 매출이 주제면 `sales_orders`, 고객 구매가치가 주제면 `customer_value`를 우선한다.

## 테이블 및 조인 가이드

- [must] `order_payments.order_id = orders.order_id`로 연결하고 상품·판매자 분석 전 결제를 주문 grain으로 선집계한다.

## 소프트 가이드

- [prefer] 결제수단 비율의 분모가 결제 레코드인지 서로 다른 주문인지 표시한다.

## 긍정 예시

- "결제수단별 서로 다른 주문 수와 총 결제금액을 보여줘."

## 부정 예시

- "상품 행에 결제금액을 붙여 카테고리별 결제금액을 합산해줘."
