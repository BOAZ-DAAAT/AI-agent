---
document_id: seller-performance
doc_type: analysis_query_rule
query_type: seller_performance
title: Olist Seller Performance Query Rules
language: ko
version: "1.0"
business_entities: [sellers, orders, order_items, reviews]
source_tables: [sellers, order_items, orders, order_reviews]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---
# 판매자 성과 분석 규칙

## 정의

판매자별 주문 수, 상품 매출, 상품 행 수, 배송 적시성, 리뷰 점수를 비교한다. 판매자 식별 grain은 `order_items.seller_id`이며, 판매자 마스터만으로 활동 판매자를 판단하지 않는다.

## 검색 별칭

- 판매자 성과, 셀러 성과, 판매자별 매출, 판매자 순위, 셀러 배송, 판매자 리뷰
- seller performance, seller sales, seller ranking, seller delay, seller review

## 기본 지표

- [default] 판매자별 주문 수는 `order_items.seller_id`별 `COUNT(DISTINCT orders.order_id)`이다.
- [default] 판매자 상품 매출은 `SUM(order_items.price)`이고, 상품 행 수는 `COUNT(*)`다.
- [prefer] 만족도는 요청할 때만 `AVG(order_reviews.review_score)`를 사용한다.
- [prefer] 고객 배송 지연과 판매자 출고 마감 미준수는 서로 다른 지표로 표시한다.

## Grain과 조인

- [must] 판매자 기준은 `order_items.seller_id`다.
- [must] 위치 정보는 `order_items.seller_id = sellers.seller_id`로, 주문 상태·시간·배송일은 `order_items.order_id = orders.order_id`로 연결한다.
- [avoid] `sellers` 테이블 행만으로 실제 판매·활동 판매자를 추론하지 않는다.
- [avoid] 주문 전체 결제금액을 배분 규칙 없이 판매자별로 전액 배정하지 않는다.
- [default] 한 주문에 여러 판매자가 있으면 주문은 각 판매자에 한 번씩 집계한다.

## 시간·상태·기본 가정

- [default] 기간별 지표는 `orders.order_purchase_timestamp`를 사용한다.
- [default] 판매자 금액은 별도 요청이 없으면 item price 기준이며 배송비·결제금액과 구분한다.
- [prefer] 판매자 비교에는 서로 다른 주문 수와 상품 행 수를 같이 제시한다.

## 확인이 필요한 경우

- [ask_if_missing] “판매자 성과”가 매출, 배송, 리뷰, 주문 수, 복합 점수 중 무엇인지 확인하거나 가정한다.
- [ask_if_missing] 사용자 정의 종합 점수를 만들기 전 지표·가중치·정규화 방식을 확인한다.
- 좋은 예: “판매자별 상품 매출, 주문 수, 평균 리뷰 점수를 따로 비교해줘.”
- 피해야 할 예: “결제금액 전체를 각 판매자 매출로 더해줘.”

## 관련 규칙

- `sales_orders`, `delivery_delay`, `review_satisfaction`, `regional_analysis`, `product_category`

## 사용하지 않는 경우

- [prefer] 고객·판매자 지역 자체가 주제면 `regional_analysis`, 카테고리 자체가 주제면 `product_category`를 우선한다.

## 테이블 및 조인 가이드

- [must] 판매자 기준은 `order_items.seller_id`이고, 위치는 `sellers.seller_id`로 연결한다.

## 소프트 가이드

- [prefer] 판매자 비교에는 서로 다른 주문 수와 상품 행 수를 분리해 표시한다.

## 긍정 예시

- "판매자별 상품 매출, 주문 수, 평균 리뷰 점수를 따로 비교해줘."

## 부정 예시

- "결제금액 전체를 각 판매자 매출로 더해줘."
