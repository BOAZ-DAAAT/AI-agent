---
document_id: review-metrics
doc_type: analysis_foundation
query_type: review_metrics
title: Olist 리뷰 지표
language: ko
version: "1.0"
source_tables: [orders, order_reviews]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---
# Olist 리뷰 지표

## 검색 별칭

- review metrics, schema contract, grain, 안전한 사용

## 정의

- [must] 평균 점수, 리뷰 수, 점수 구간 비율은 리뷰 grain 지표다.
- [must] 참조 source table에 있는 컬럼만 사용한다.
- [avoid] 스키마에 없는 비즈니스 필드를 추론하거나 grain을 묵시적으로 바꾸지 않는다.
- [prefer] 결과가 의존하는 지표, grain, 조인 경로를 명시한다.
