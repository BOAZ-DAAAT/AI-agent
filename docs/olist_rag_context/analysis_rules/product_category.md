---
document_id: product-category
doc_type: analysis_query_rule
query_type: product_category
title: Olist Product Category Query Rules
language: ko
version: "1.0"
business_entities: [products, categories, orders, order_items]
source_tables: [products, product_category_name_translation, order_items, orders, order_reviews]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---
# 상품 카테고리 분석 규칙

## 정의

상품 카테고리 분석은 `products.product_category_name` 또는 `product_category_name_translation`의 영문명을 기준으로 주문·상품행·매출·배송·리뷰 지표를 카테고리별로 비교한다. 기본 출력 grain은 카테고리다.

## 지원하는 질문

- 카테고리별 주문 수, 상품 행 수, 상품 매출, 배송비, 평균 가격을 비교한다.
- 리뷰 점수·배송 지연·기간별 추이를 카테고리로 나눈다.
- 포르투갈어 카테고리명과 영문 번역명을 선택해 보고한다.

## 검색 별칭

- 상품 카테고리, 카테고리별 매출, 카테고리 성과, 상품군, 카테고리 믹스, 영문 카테고리
- product category, category sales, category performance, category mix, product mix

## 기본 지표

- [default] 카테고리별 주문 수는 `COUNT(DISTINCT orders.order_id)`이다.
- [default] 카테고리별 상품 행 수는 `order_items`의 `COUNT(*)`이다.
- [default] 카테고리별 상품 매출은 `SUM(order_items.price)`이고 평균 상품 가격은 `AVG(order_items.price)`다.
- [prefer] 만족도는 요청이 있을 때만 `AVG(order_reviews.review_score)`를 사용하며, 배송 지표는 `delivery_delay` 규칙을 함께 적용한다.

## Grain과 조인

- [must] 카테고리는 상품/상품행 grain에서 정해지므로 `order_items.product_id = products.product_id`로 연결한다.
- [must] 영문 라벨은 `products.product_category_name = product_category_name_translation.product_category_name`으로 연결한다.
- [must] 주문 수·시간·상태가 필요하면 `order_items.order_id = orders.order_id`로 연결한다.
- [avoid] 영문 카테고리명을 `products`에 직접 조인 키로 사용하지 않는다.
- [avoid] 상품 행을 조인한 뒤 `COUNT(*)`를 카테고리 주문 수로 사용하지 않는다.
- [prefer] 한 주문에 여러 카테고리가 있으면 해당 주문은 각 구매 카테고리에 한 번씩 나타난다고 결과에 밝힌다.

## 상태·결측·기본 가정

- [default] 완료·배송 완료 성과 요청일 때만 delivered 주문으로 제한한다.
- [default] `products.product_category_name`이 null이면 unknown 카테고리로 남기고, 번역이 없으면 원문명을 보존한다.
- [avoid] 리뷰가 없는 주문을 0점 리뷰로 보거나, 상품 마스터 행 수를 판매 상품 수로 보지 않는다.

## 확인이 필요한 경우

- [ask_if_missing] “상위 카테고리” 또는 “카테고리 성과”의 기준이 매출·주문·리뷰·배송 중 무엇인지 확인하거나 가정한다.
- [ask_if_missing] 카테고리 점유율에서 다중 카테고리 주문을 어떻게 해석할지가 결과를 바꾸면 명시한다.
- 좋은 예: “영문 카테고리별 상품 매출과 서로 다른 주문 수를 비교해줘.”
- 피해야 할 예: “브랜드별·3단계 카테고리 체계를 만들어줘.” 해당 스키마에는 단일 카테고리만 있다.

## 관련 규칙

- `sales_orders`, `delivery_delay`, `review_satisfaction`, `seller_performance`, `regional_analysis`

## 사용하지 않는 경우

- [prefer] 카테고리 분해가 없으면 `sales_orders`, 판매자 자체 비교가 주제면 `seller_performance`를 우선한다.

## 테이블 및 조인 가이드

- [must] `order_items.product_id = products.product_id`로 연결하고 영문 라벨은 category translation 테이블을 사용한다.

## 소프트 가이드

- [prefer] 주문 수가 상품행 기준인지 서로 다른 주문 기준인지, 다중 카테고리 주문 처리 방식을 표시한다.

## 긍정 예시

- "영문 카테고리별 상품 매출과 서로 다른 주문 수를 비교해줘."

## 부정 예시

- "브랜드별 3단계 카테고리 체계를 만들어줘." 해당 스키마에는 없다.
