---
document_id: table-order-payments
doc_type: analysis_foundation
query_type: table_order_payments
title: Olist 주문 결제 테이블 계약서
language: ko
version: "1.0"
source_tables: [order_payments, orders]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---
# Olist 주문 결제 테이블 계약서

## 검색 별칭

- table order payments, schema contract, grain, 안전한 사용

## 정의

- [must] 한 주문에는 여러 결제 행이 있을 수 있으며 payment_sequential은 할부 횟수가 아니다.
- [must] 참조 source table에 있는 컬럼만 사용한다.
- [avoid] 스키마에 없는 비즈니스 필드를 추론하거나 grain을 묵시적으로 바꾸지 않는다.
- [prefer] 결과가 의존하는 지표, grain, 조인 경로를 명시한다.

## 컬럼 계약

- [must] `order_id`와 `payment_sequential`의 조합이 결제 레코드 grain이다.
- [prefer] `payment_type`은 결제 수단, `payment_installments`는 할부 횟수, `payment_value`는 결제 레코드 금액이다.
- [avoid] `payment_sequential`을 할부 횟수로 해석하지 않는다.

## 조인 및 주의사항

- [must] `order_payments.order_id = orders.order_id`로 연결한다.
- [avoid] 결제금액을 상품·카테고리·판매자에 배분하려면 배분 규칙 없이 전액을 중복 배정하지 않는다.
- [prefer] 상품 행과 함께 사용할 때는 결제를 주문 grain으로 먼저 집계한다.
