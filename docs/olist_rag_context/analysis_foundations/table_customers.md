---
document_id: table-customers
doc_type: analysis_foundation
query_type: table_customers
title: Olist 고객 테이블 계약서
language: ko
version: "1.0"
source_tables: [customers, orders, geolocation]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---
# Olist 고객 테이블 계약서

## 검색 별칭

- table customers, schema contract, grain, 안전한 사용

## 정의

- [must] customer_id는 주문 조인 키이고 customer_unique_id는 고객 생애 식별자다.
- [must] 참조 source table에 있는 컬럼만 사용한다.
- [avoid] 스키마에 없는 비즈니스 필드를 추론하거나 grain을 묵시적으로 바꾸지 않는다.
- [prefer] 결과가 의존하는 지표, grain, 조인 경로를 명시한다.

## 컬럼 계약

- [must] `customers.customer_id`는 `orders.customer_id`와 연결하는 주문 단위 고객 레코드 키다.
- [must] `customers.customer_unique_id`는 반복 구매·RFM·고객가치에 쓰는 실제 고객 식별자다.
- [prefer] `customer_zip_code_prefix`, `customer_city`, `customer_state`는 고객 도착지·수요 지역 분석에 사용한다.

## 조인 및 주의사항

- [must] 주문은 `orders.customer_id = customers.customer_id`로 연결한다.
- [prefer] 좌표가 필요하면 `customer_zip_code_prefix = geolocation.geolocation_zip_code_prefix`로 연결하기 전에 geolocation을 prefix grain으로 집계한다.
- [avoid] `customer_id`로 재구매 고객을 세거나, 판매자 지역을 고객 지역으로 대체하지 않는다.
