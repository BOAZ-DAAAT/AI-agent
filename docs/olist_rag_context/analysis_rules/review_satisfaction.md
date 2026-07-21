---
document_id: review-satisfaction
doc_type: analysis_query_rule
query_type: review_satisfaction
title: Olist Review Satisfaction Query Rules
language: ko
version: "1.0"
business_entities: [reviews, orders, customers, sellers, products]
source_tables: [order_reviews, orders, customers, order_items, sellers, products]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---
# 리뷰 만족도 분석 규칙

## 정의

`order_reviews.review_score`를 사용해 평균 만족도, 긍정·부정 리뷰 비율, 리뷰 수, 리뷰 생성·응답 시점을 분석한다. 기본 grain은 `order_reviews.review_id`다.

## 검색 별칭

- 리뷰 만족도, 리뷰 점수, 고객 만족, 부정 리뷰, 긍정 리뷰, 리뷰 코멘트, 평점
- review satisfaction, review score, negative review, positive review, comment rate

## 기본 지표

- [default] 평균 만족도는 `AVG(order_reviews.review_score)`이다.
- [default] 부정 리뷰는 `review_score <= 2`, 긍정 리뷰는 `review_score >= 4`로 정의한다.
- [default] 리뷰 수는 `COUNT(DISTINCT order_reviews.review_id)`다.
- [prefer] 긍정·중립·부정 비율을 제시할 때 사용한 점수 구간을 함께 적는다.

## Grain과 조인

- [must] 리뷰 grain은 `order_reviews.review_id`이며, 주문 정보는 `order_reviews.order_id = orders.order_id`로 연결한다.
- [prefer] 판매자·카테고리·상품 분해가 요청됐을 때만 item 테이블을 추가로 조인한다.
- [avoid] 리뷰가 없는 주문을 0점 리뷰로 처리하지 않는다.
- [avoid] 여러 판매자·카테고리가 있는 주문의 주문 단위 리뷰를 특정 하나의 판매자나 카테고리에 단정하지 않는다.

## 시간·결측·기본 가정

- [default] 리뷰 시점 분석은 `order_reviews.review_creation_date`를 사용한다.
- [default] 구매기간별 만족도 비교일 때만 `orders.order_purchase_timestamp`를 사용한다.
- [prefer] 판매자·카테고리별 리뷰 결과는 주문 단위 리뷰의 귀속 한계를 함께 표시한다.

## 확인이 필요한 경우

- [ask_if_missing] “나쁜 리뷰”의 점수 기준이 정의되지 않았고 기준 변경이 결론을 바꾸면 확인하거나 가정한다.
- [ask_if_missing] 자유 텍스트 코멘트를 불만 유형으로 분류하려면 별도 분류 기준을 확인한다.
- 좋은 예: “월별 평균 리뷰 점수와 1~2점 리뷰 비율을 보여줘.”
- 피해야 할 예: “리뷰가 없는 주문을 불만족으로 간주해줘.”

## 관련 규칙

- `delivery_delay`, `seller_performance`, `product_category`, `sales_orders`, `regional_analysis`

## 사용하지 않는 경우

- [prefer] 리뷰 지표 없이 주문·매출만 필요하면 `sales_orders`를 우선한다.

## 테이블 및 조인 가이드

- [must] `order_reviews.order_id = orders.order_id`로 연결하고 다중 상품 주문의 리뷰 귀속 한계를 유지한다.

## 소프트 가이드

- [prefer] 긍정·부정 비율을 낼 때 사용한 리뷰 점수 임계값을 표시한다.

## 긍정 예시

- "월별 평균 리뷰 점수와 1~2점 리뷰 비율을 보여줘."

## 부정 예시

- "리뷰가 없는 주문을 불만족으로 간주해줘."
