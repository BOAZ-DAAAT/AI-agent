---
document_id: table-order-reviews
doc_type: analysis_foundation
query_type: table_order_reviews
title: Olist 주문 리뷰 테이블 계약서
language: ko
version: "1.0"
source_tables: [order_reviews, orders]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---
# Olist 주문 리뷰 테이블 계약서

## 검색 별칭

- table order reviews, schema contract, grain, 안전한 사용

## 정의

- [must] review_id가 리뷰 grain이며 주문 단위 리뷰를 모든 상품이나 판매자에게 고유하게 귀속할 수 없다.
- [must] 참조 source table에 있는 컬럼만 사용한다.
- [avoid] 스키마에 없는 비즈니스 필드를 추론하거나 grain을 묵시적으로 바꾸지 않는다.
- [prefer] 결과가 의존하는 지표, grain, 조인 경로를 명시한다.

## 컬럼 계약

- [must] `review_id`가 리뷰 grain이고 `order_id`가 주문 연결 키다.
- [default] `review_score`는 구조화된 만족도 점수다.
- [prefer] `review_comment_title`, `review_comment_message`는 사용자가 텍스트 분석을 명시했을 때만 사용한다.
- [prefer] `review_creation_date`는 리뷰 생성 시점, `review_answer_timestamp`는 응답 시점 분석에 사용한다.

## 조인 및 주의사항

- [must] `order_reviews.order_id = orders.order_id`로 연결한다.
- [avoid] 리뷰가 없는 주문을 0점으로 처리하지 않는다.
- [avoid] 주문 단위 리뷰를 다중 상품·판매자 주문에서 특정 단일 상품·판매자의 사실로 단정하지 않는다.
