---
document_id: customer-value
doc_type: analysis_query_rule
query_type: customer_value
title: Olist Customer Value Query Rules
language: ko
version: "1.0"
business_entities: [customers, orders, order_items, payments]
source_tables: [customers, orders, order_items, order_payments]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---
# 고객 가치 분석 규칙

## 정의와 사용 시점

- 고객별 누적 구매금액, RFM, 고가치 고객 구간, 고객별 평균 주문금액을 분석할 때 사용한다.
- 구매 횟수만 중심이면 `purchase_frequency`, 결제 방식 중심이면 `payment_behavior`를 사용한다.

## 검색 별칭

- 고객 가치, 고가치 고객, 고객별 매출, 고객별 구매금액, RFM, LTV, CLV
- customer value, high-value customer, customer spend, monetary value, RFM, LTV, CLV

## 지표 정의

- [must] 고객 금액은 별도 정의가 없으면 `customers.customer_unique_id`별 `SUM(order_items.price)`이다.
- [default] 고객 빈도는 고객별 `COUNT(DISTINCT orders.order_id)`이다.
- [default] 최근성(recency)은 기준일과 고객의 `MAX(orders.order_purchase_timestamp)` 차이로 계산한다.
- [prefer] 고객별 평균 상품금액은 누적 상품금액을 서로 다른 주문 수로 나눈다.

## Grain과 조인

- [must] 고객 가치·RFM·생애가치의 customer grain은 `customers.customer_unique_id`이다.
- [must] `orders.customer_id = customers.customer_id`, `orders.order_id = order_items.order_id`를 사용한다.
- [avoid] 상품 행과 결제 행을 직접 조인한 뒤 고객 금액을 합산하지 않는다.
- [avoid] `customers.customer_id`로 고객 생애가치를 집계하지 않는다.

## 기본 가정과 해석

- [default] “고객 가치”는 결제 관련 표현이 없다면 상품금액 기준의 관측 누적가치다.
- [default] 기준일이 없으면 최대 관측 구매일을 사용하되 가정임을 밝힌다.
- [prefer] 이는 예측 CLV가 아니라 관측된 과거 구매가치임을 명시한다.
- [prefer] 카테고리·판매자별로 분해하면 한 고객이 여러 그룹에 나타날 수 있음을 알린다.

## 확인이 필요한 경우와 예시

- [ask_if_missing] “가치”가 상품금액, 결제금액, 주문 횟수 중 무엇인지 모호하면 확인하거나 가정을 명시한다.
- [ask_if_missing] RFM의 최근성 기준일이 비즈니스 기준일이어야 하면 확인한다.
- 좋은 예: “고객별 누적 상품금액과 주문 횟수로 상위 고객을 보여줘.”
- 피해야 할 예: “고객별 미래 CLV를 예측해줘.” 예측 모델 입력·목표가 제공되지 않았다.

## 관련 규칙

- `purchase_frequency`, `sales_orders`, `payment_behavior`, `product_category`, `regional_analysis`

## 사용하지 않는 경우

- [prefer] 결제 방식이 주제면 `payment_behavior`, 주문 횟수만 필요하면 `purchase_frequency`를 우선한다.

## 테이블 및 조인 가이드

- [must] 고객 가치에는 `orders.customer_id = customers.customer_id`, `orders.order_id = order_items.order_id`를 사용한다.

## 소프트 가이드

- [prefer] 관측된 과거 구매가치인지 예측 CLV인지 구분해 표시한다.

## 긍정 예시

- "고객별 누적 상품금액과 주문 횟수로 상위 고객을 보여줘."

## 부정 예시

- "고객별 미래 CLV를 예측해줘." 예측 목표와 입력이 제공되지 않았다.
