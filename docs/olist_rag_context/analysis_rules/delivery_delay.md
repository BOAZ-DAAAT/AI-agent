---
document_id: delivery-delay
doc_type: analysis_query_rule
query_type: delivery_delay
title: Olist Delivery Delay Query Rules
language: ko
version: "1.0"
business_entities: [orders, order_items, customers, sellers]
source_tables: [orders, order_items, customers, sellers]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---
# 배송 지연 분석 규칙

## 정의와 사용 시점

- 고객 배송 지연, 배송 소요 기간, 약속일 미준수, 판매자 출고 마감일 미준수를 분석할 때 사용한다.
- 상태 수만 필요하면 `sales_orders`, 리뷰 점수만 필요하면 `review_satisfaction`을 사용한다.

## 검색 별칭

- 배송 지연, 늦은 배송, 배송 기간, 예상 배송일, 실제 배송일, 지연율, 배송 소요
- delivery delay, late delivery, delivery duration, promised date, transit days

## 지표 정의

- [must] 고객 배송 지연은 `orders.order_delivered_customer_date > orders.order_estimated_delivery_date`이다.
- [default] 지연일은 실제 고객 배송일과 예상 배송일의 날짜 차이다.
- [default] 배송 소요기간은 `orders.order_purchase_timestamp`부터 `orders.order_delivered_customer_date`까지의 차이다.
- [prefer] 판매자 출고 마감 미준수는 `order_items.shipping_limit_date`와 `orders.order_delivered_carrier_date`를 비교하며 고객 지연과 분리해 표시한다.

## Grain과 조인

- [must] 기본 고객 배송 지연의 grain은 `orders.order_id`다.
- [must] 기본 지연 지표는 `orders`만으로 계산하고, 판매자·상품·카테고리 분석일 때만 `order_items`를 조인한다.
- [avoid] 판매자·상품을 붙인 뒤 item row 수를 지연 주문 수로 세지 않는다.

## 기본 가정과 해석

- [default] 지연 계산은 실제·예상 배송일이 모두 있는 delivered 주문으로 제한한다.
- [default] 기간별 배송 cohort는 별도 요청이 없으면 구매일을 사용한다.
- [prefer] 지연율은 지연 주문의 분자와 eligible delivered 주문의 분모를 함께 설명한다.

## 확인이 필요한 경우와 예시

- [ask_if_missing] “지연”이 고객 약속일 초과, 구매-배송 소요일, 판매자 출고 마감 미준수 중 무엇인지 모호하면 확인한다.
- [ask_if_missing] 미배송 주문을 진행 중으로 볼지 지연으로 볼지가 결과에 중요하면 확인한다.
- 좋은 예: “주별 delivered 주문의 평균 지연일과 지연율을 보여줘.”
- 피해야 할 예: “배송일이 없는 주문을 예상 배송일로 채워 지연일을 계산해줘.”

## 관련 규칙

- `sales_orders`, `seller_performance`, `regional_analysis`, `product_category`, `review_satisfaction`

## 사용하지 않는 경우

- [prefer] 주문 상태 수만 필요하면 `sales_orders`, 리뷰 점수만 필요하면 `review_satisfaction`을 우선한다.

## 테이블 및 조인 가이드

- [must] 기본 고객 배송 지연은 `orders`에서 계산하고 판매자·상품 분석이 필요할 때만 `order_items`를 연결한다.

## 소프트 가이드

- [prefer] 지연율에는 지연 주문 분자와 eligible delivered 주문 분모를 함께 설명한다.

## 긍정 예시

- "주별 delivered 주문의 평균 지연일과 지연율을 보여줘."

## 부정 예시

- "배송일이 없는 주문을 예상 배송일로 채워 지연일을 계산해줘."
