---
document_id: table-orders
doc_type: analysis_foundation
query_type: table_orders
title: Olist 주문 테이블 계약서
language: ko
version: "1.0"
source_tables: [orders, customers, order_items, order_payments, order_reviews]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---
# Olist 주문 테이블 계약서

## 검색 별칭

- table orders, schema contract, grain, 안전한 사용

## 정의

- [must] order_id는 주문 기본 키이며 order_status와 시점 컬럼이 주문 라이프사이클 분석을 정의한다.
- [must] 참조 source table에 있는 컬럼만 사용한다.
- [avoid] 스키마에 없는 비즈니스 필드를 추론하거나 grain을 묵시적으로 바꾸지 않는다.
- [prefer] 결과가 의존하는 지표, grain, 조인 경로를 명시한다.

## 컬럼 계약

- [must] `order_id`가 주문 grain, `customer_id`가 고객 연결 키, `order_status`가 주문 상태다.
- [default] 주문 추이는 `order_purchase_timestamp`, 승인 분석은 `order_approved_at`를 사용한다.
- [prefer] `order_delivered_carrier_date`는 판매자에서 운송사로 전달된 시점, `order_delivered_customer_date`는 고객 실제 배송 시점이다.
- [prefer] `order_estimated_delivery_date`는 약속된 예상 고객 배송일이다.

## 조인 및 주의사항

- [must] orders는 customers·items·payments·reviews의 중심 조인 테이블이다.
- [avoid] delivered 상태와 날짜 존재 여부를 같은 의미로 임의 대체하지 않는다.
- [avoid] null 배송일을 예상 배송일로 채워 배송 지표를 계산하지 않는다.
