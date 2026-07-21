---
document_id: regional-analysis
doc_type: analysis_query_rule
query_type: regional_analysis
title: Olist Regional Analysis Query Rules
language: ko
version: "1.0"
business_entities: [customers, sellers, geolocation, orders]
source_tables: [customers, sellers, geolocation, orders, order_items, order_payments, order_reviews, products]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---
# 지역 분석 규칙

## 정의

고객 또는 판매자 위치를 기준으로 주문·매출·고객·판매자·리뷰·배송 지표를 지역별로 비교한다. 고객 지역과 판매자 지역은 다른 질문에 답하므로 구분한다.

## 검색 별칭

- 지역 분석, 지역별, 고객 지역, 판매자 지역, 도시별, 주별, 배송 지역, 우편번호, 출발지, 도착지
- regional analysis, customer state, seller state, city, geography, geolocation, zip prefix, origin, destination

## 기본 지표

- [default] 지역별 주문 수는 요청한 고객/판매자 지역별 `COUNT(DISTINCT orders.order_id)`다.
- [default] 지역별 고객 수는 `COUNT(DISTINCT customers.customer_unique_id)`다.
- [default] 활동 판매자 수는 판매자 지역별 `COUNT(DISTINCT order_items.seller_id)`다.

## Grain과 조인

- [must] 고객·수요·도착지 지역은 `orders.customer_id = customers.customer_id`로 연결한다.
- [must] 판매자·공급·출발지 지역은 `order_items.seller_id = sellers.seller_id`로 연결한다.
- [must] 고객 지역과 판매자 지역은 서로 다른 grain이며 같은 의미로 대체하지 않는다.
- [avoid] 판매자 지역을 구매자·고객 지역으로 사용하지 않는다.
- [avoid] 도시명만으로 geolocation을 조인해 좌표를 만들지 않는다.
- [prefer] zip prefix 좌표가 필요하면 중복 geolocation 행을 zip prefix 단위로 먼저 집계한다.

## 기본 가정과 표현

- [default] buyer/destination/demand 질문은 고객 지역을 사용한다.
- [default] seller/origin/supply 질문은 판매자 지역을 사용한다.
- [prefer] 결과 라벨에 고객(도착·수요) 지역인지 판매자(출발·공급) 지역인지, city/state/zip-prefix 중 어떤 수준인지 명시한다.

## 확인이 필요한 경우

- [ask_if_missing] “지역”이 고객 지역인지 판매자 지역인지 모호하고 결과가 달라지면 확인하거나 가정한다.
- [ask_if_missing] 거리·지도·좌표 분석은 geolocation 조인과 집계 방식을 별도로 확인한다.
- 좋은 예: “고객 state별 delivered 주문 수와 평균 배송 지연일을 보여줘.”
- 피해야 할 예: “판매자 state를 고객의 배송 지역으로 해석해줘.”

## 관련 규칙

- `sales_orders`, `delivery_delay`, `seller_performance`, `customer_value`, `payment_behavior`, `review_satisfaction`

## 사용하지 않는 경우

- [prefer] 지리적 분해가 없으면 해당 지표의 분석 주제 규칙을 우선한다.

## 테이블 및 조인 가이드

- [must] 고객 지역은 `orders.customer_id = customers.customer_id`, 판매자 지역은 `order_items.seller_id = sellers.seller_id`를 사용한다.

## 소프트 가이드

- [prefer] 고객(도착·수요) 지역인지 판매자(출발·공급) 지역인지와 city/state/zip-prefix 수준을 표시한다.

## 긍정 예시

- "고객 state별 delivered 주문 수와 평균 배송 지연일을 보여줘."

## 부정 예시

- "판매자 state를 고객의 배송 지역으로 해석해줘."
