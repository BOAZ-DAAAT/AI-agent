---
document_id: purchase-frequency
doc_type: analysis_query_rule
query_type: purchase_frequency
title: Olist Purchase Frequency Query Rules
language: ko
version: "1.0"
business_entities: [customers, orders]
source_tables: [customers, orders]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---
# 구매 빈도 및 재구매 분석 규칙

## 정의와 사용 시점

- 같은 고객이 얼마나 자주 주문하는지, 재구매 고객 비중·주문 횟수 구간·재주문 간격을 분석할 때 사용한다.
- 전체 주문량만 필요하면 `sales_orders`, 금액 중심 고객 세분화면 `customer_value`를 사용한다.

## 검색 별칭

- 구매 빈도, 재구매, 반복 구매, 단골, 충성 고객, 재주문, 1회 구매 고객, 다회 구매 고객
- purchase frequency, repeat customer, repurchase, reorder, loyal customer, one-time buyer

## 지표 정의

- [must] 고객별 구매 횟수는 `customers.customer_unique_id`별 `COUNT(DISTINCT orders.order_id)`이다.
- [default] 재구매 고객은 서로 다른 주문이 2건 이상인 고객이다.
- [default] 1회 구매 고객은 주문이 정확히 1건인 고객이다.
- [prefer] 재주문 간격은 같은 `customer_unique_id` 안에서 연속한 `orders.order_purchase_timestamp`의 날짜 차이로 계산한다.

## Grain과 조인

- [must] 고객 생애·반복 행동의 고객 grain은 `customers.customer_unique_id`이다.
- [must] `orders.customer_id = customers.customer_id`로 연결한다.
- [avoid] `customers.customer_id`는 주문 연결 키이므로 반복 고객 식별자로 사용하지 않는다.
- [avoid] 결제·리뷰·상품 행을 구매 이벤트로 세지 않는다.

## 시간·상태·기본 가정

- [default] 별도 요청이 없으면 관측된 전체 주문 기간을 사용한다.
- [default] 사용자가 delivered/완료 구매를 요청하지 않으면 모든 관측 주문을 포함한다.
- [prefer] 반복 고객 비율을 제시할 때 분모와 반복 기준(2회 이상)을 함께 적는다.

## 확인이 필요한 경우와 예시

- [ask_if_missing] “충성 고객”의 정의가 단순 재구매를 넘어 금액·기간·횟수 기준을 요구하면 기준을 확인하거나 가정을 명시한다.
- [ask_if_missing] 유지율·이탈률은 기준 기간에 따라 의미가 달라지므로 기간을 확인한다.
- 좋은 예: “고객별 주문 횟수 분포와 2회 이상 재구매 고객 비율을 보여줘.”
- 피해야 할 예: “customer_id 기준으로 재구매 고객을 세어줘.”

## 관련 규칙

- `customer_value`, `sales_orders`, `regional_analysis`, `product_category`, `seller_performance`

## 사용하지 않는 경우

- [prefer] 금액 중심 고객 세분화는 `customer_value`, 전체 주문 추이는 `sales_orders`를 우선한다.

## 테이블 및 조인 가이드

- [must] `orders.customer_id = customers.customer_id`로 연결한 뒤 `customer_unique_id`별로 주문을 집계한다.

## 소프트 가이드

- [prefer] 재구매 기준과 관측 기간을 결과에 함께 표시한다.

## 긍정 예시

- "고객별 주문 횟수 분포와 2회 이상 재구매 고객 비율을 보여줘."

## 부정 예시

- "customer_id 기준으로 재구매 고객을 세어줘."
