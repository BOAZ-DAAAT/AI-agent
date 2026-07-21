---
document_id: time-status-population-rules
doc_type: analysis_foundation
query_type: time_status_population_rules
title: Olist 시간 상태 및 모집단 규칙
language: ko
version: "1.0"
source_tables: [orders, order_items, order_reviews]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---

# Olist 시간·상태·모집단 규칙

## 검색 별칭

- 기간 기준, 주문일, 배송일, 리뷰일, 주문 상태, 모집단, delivered 주문
- time basis, purchase date, delivery date, review date, order status, population, delivered

## 시간 기준

- [default] 주문·매출 추이는 `orders.order_purchase_timestamp`를 사용한다.
- [default] 승인 질문은 `orders.order_approved_at`, 실제 배송 질문은 `orders.order_delivered_customer_date`를 사용한다.
- [default] 리뷰 시점은 `order_reviews.review_creation_date`, 판매자 출고 마감 질문은 `order_items.shipping_limit_date`를 사용한다.
- [avoid] 리뷰 생성일을 주문일로, 예상 배송일을 실제 배송일로 사용하지 않는다.

## 상태와 모집단

- [default] 일반 주문 생성·유입 분석은 별도 요청이 없으면 모든 `orders.order_status`를 포함한다.
- [default] 완료·fulfilled·배송 완료 분석은 `orders.order_status = 'delivered'`를 사용한다.
- [default] 배송 지연은 실제·예상 배송일이 모두 있는 delivered 주문을 모집단으로 한다.
- [avoid] 시간 기반 지표에서 null 날짜를 다른 날짜로 임의 대체하지 않는다.

## 조작적 정의

- [prefer] 사용자가 cohort·기간·상태 집합을 명시하면 기본 모집단보다 우선한다.
- [ask_if_missing] 최근·성공 주문·유효 주문의 기간 또는 상태 범위가 결론을 바꿀 때만 확인하거나 가정을 명시한다.
