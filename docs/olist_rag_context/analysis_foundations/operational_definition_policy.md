---
document_id: operational-definition-policy
doc_type: analysis_foundation
query_type: operational_definition_policy
title: Olist 조작적 정의 정책
language: ko
version: "1.0"
source_tables: [customers, orders, order_items, order_payments, order_reviews]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---

# 조작적 정의 적용 정책

## 검색 별칭

- 조작적 정의, 사용자 정의 지표, 임계값, cohort, override, 커스텀 KPI
- operational definition, custom metric, threshold, cohort, override, custom KPI

## 우선순위

- [must] 사용자가 명시한 지표·grain·시간 기준·상태 집합·임계값·cohort 정의는 schema로 계산 가능하면 기본 정의보다 우선한다.
- [must] 사용자 정의는 SQL 계획과 최종 결과에 명시한다.
- [avoid] 기본 정의가 있다는 이유로 명시적 사용자 조작적 정의를 덮어쓰지 않는다.

## 계산 가능성 점검

- [prefer] 필요한 컬럼과 조인 경로가 존재하고, 다중행 조인이 요청한 지표를 중복 집계하지 않는지 점검한다.
- [prefer] 금액 관점·분모·기간·상태가 달라지면 계산식과 모집단을 함께 기록한다.
- [ask_if_missing] 높은 가치·지연·활성 고객·성공 주문처럼 임계값/모집단이 없고 분석 결과를 크게 바꾸는 경우에만 확인한다.

## 예시

- [prefer] “배송 지연은 구매 후 실제 배송까지 7일 초과”라면 `order_purchase_timestamp`와 `order_delivered_customer_date` 차이에 7일 임계값을 적용한다.
- [prefer] “충성 고객은 90일 내 3회 이상 주문”이라면 `customer_unique_id`, 90일 창, 3회 임계값을 사용한다.
- [avoid] 존재하지 않는 할인·환불·비용·마진 컬럼을 요구하는 조작적 정의를 계산 가능한 사실처럼 처리하지 않는다.
