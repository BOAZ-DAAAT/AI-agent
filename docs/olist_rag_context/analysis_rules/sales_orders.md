---
document_id: sales-orders
doc_type: analysis_query_rule
query_type: sales_orders
title: Olist Sales and Orders Query Rules
language: ko
version: "1.0"
business_entities: [orders, customers, payments, order_items]
source_tables: [customers, orders, order_items, order_payments]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---
# 매출 및 주문 분석 규칙

## 정의와 사용 시점

- 주문 수, 상품 매출, 배송비, 주문 상태, 객단가(AOV), 기간별 매출을 분석할 때 사용한다.
- 범용 매출 질문의 기반 규칙이며, 지역·카테고리·판매자·결제 분석이 필요하면 해당 규칙을 함께 사용한다.
- 결제수단이나 할부가 주제면 `payment_behavior`, 재구매나 고객가치가 주제면 각각 `purchase_frequency`, `customer_value`를 우선한다.

## 검색 별칭

- 매출, 주문, 주문 수, 판매액, 거래액, 상품금액, 객단가, 월별 매출, 주문 상태
- sales, revenue, order volume, GMV, merchandise sales, order count, item count, AOV

## 지표 정의

- [default] 주문 수는 `COUNT(DISTINCT orders.order_id)`이다.
- [default] 상품 매출은 `SUM(order_items.price)`이며 배송비는 포함하지 않는다.
- [default] 배송비 포함 상품 매출은 `SUM(order_items.price + order_items.freight_value)`이다.
- [default] 객단가는 `SUM(order_items.price) / COUNT(DISTINCT orders.order_id)`이다.
- [prefer] `SUM(order_payments.payment_value)`는 상품 매출과 다른 결제금액 관점이므로, 사용자가 결제·수납 금액을 요청한 경우에만 사용한다.

## Grain과 조인

- [must] 주문 단위 지표는 `orders.order_id` grain을 유지하고 다대일이 아닌 상세 테이블을 붙이기 전에 집계한다.
- [must] 상품 매출·판매자·상품·카테고리 분석은 `orders.order_id = order_items.order_id`로 조인한다.
- [avoid] `order_items`와 `order_payments`를 그대로 조인한 뒤 금액을 합산하지 않는다. 한 주문에 양쪽 모두 여러 행이 있을 수 있다.
- [avoid] `order_items.order_item_id`를 주문 수나 수량으로 해석하지 않는다. 상품 행 수가 필요할 때만 `COUNT(*)`를 사용한다.
- [prefer] 고객 지역은 `orders.customer_id = customers.customer_id`로, 상품 속성은 `order_items.product_id = products.product_id`로 연결한다.

## 시간·상태·기본 가정

- [default] 기간 기준은 `orders.order_purchase_timestamp`를 사용한다.
- [default] 일반 주문 추이는 모든 상태를 포함하며, delivered/completed/fulfilled 요청일 때만 `orders.order_status = 'delivered'` 필터를 사용한다.
- [avoid] 리뷰 생성일을 주문일로 사용하거나, 결측 배송일을 예상 배송일로 대체하지 않는다.
- [prefer] 결과에는 상품금액, 배송비 포함 금액, 결제금액 중 어떤 금액 관점인지 명시한다.

## 확인이 필요한 경우와 예시

- [ask_if_missing] “매출” 또는 “revenue”가 상품금액, 배송비 포함 금액, 결제금액 중 무엇인지 결과를 크게 바꾸면 질문하거나 가정을 명시한다.
- [ask_if_missing] “최근”의 기간이 정의되지 않아 추세 해석이 달라지면 기간 가정을 명시한다.
- 좋은 예: “2018년 월별 delivered 주문 수와 상품 매출을 보여줘.”
- 피해야 할 예: “주문별 이익을 계산해줘.” 비용·마진·환불·세금 데이터는 스키마에 없다.

## 관련 규칙

- `customer_value`, `product_category`, `seller_performance`, `payment_behavior`, `regional_analysis`, `delivery_delay`

## 사용하지 않는 경우

- [prefer] 결제수단·할부가 핵심이면 `payment_behavior`, 반복 구매·고객 생애가치가 핵심이면 고객 규칙을 우선한다.

## 테이블 및 조인 가이드

- [must] 상품 매출은 `orders.order_id = order_items.order_id`로 연결하고, 결제금액을 함께 쓸 때는 각 상세 테이블을 주문 grain으로 선집계한다.

## 소프트 가이드

- [prefer] 결과에 상품금액·배송비 포함 금액·결제금액 중 어느 금액 관점인지 표시한다.

## 긍정 예시

- "2018년 월별 delivered 주문 수와 상품 매출을 보여줘."

## 부정 예시

- "주문별 이익을 계산해줘." 비용·마진 데이터는 없다.
