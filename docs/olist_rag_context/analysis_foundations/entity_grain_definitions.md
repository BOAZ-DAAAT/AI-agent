---
document_id: entity-grain-definitions
doc_type: analysis_foundation
query_type: entity_grain_definitions
title: Olist 엔터티 및 Grain 정의
language: ko
version: "1.0"
source_tables: [customers, orders, order_items, order_payments, order_reviews, products, sellers]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---

# Olist 엔터티 및 Grain 정의

## 검색 별칭

- 분석 단위, grain, 엔터티, 고객 키, 주문 키, 상품 행, 결제 행, 리뷰 행
- entity grain, analysis unit, customer key, order key, item row, payment row, review row

## 핵심 엔터티

- [must] 주문 grain은 `orders.order_id`다.
- [must] 고객 생애 grain은 `customers.customer_unique_id`다.
- [avoid] `customers.customer_id`는 주문-고객 조인 키이지 반복 구매·RFM·생애가치 식별자가 아니다.
- [must] 상품 행 grain은 `order_items.order_id`와 `order_items.order_item_id`의 조합이다.
- [must] 판매자 grain은 `order_items.seller_id`다.
- [must] 결제 행 grain은 `order_payments.order_id`와 `order_payments.payment_sequential`의 조합이다.
- [must] 리뷰 grain은 `order_reviews.review_id`다.
- [must] 상품·판매자 마스터 grain은 각각 `products.product_id`, `sellers.seller_id`다.

## Grain 전환

- [prefer] 상품·결제·리뷰처럼 주문당 여러 행이 가능한 테이블을 조인한 뒤 주문 지표는 `COUNT(DISTINCT orders.order_id)`를 사용하거나 주문 grain으로 선집계한다.
- [prefer] 고객 단위 분석은 주문을 `customer_unique_id`에 연결한 후 고객 grain으로 집계한다.
- [avoid] 판매자·카테고리별 주문 수의 합이 전체 주문 수와 같다고 가정하지 않는다.

## 사용자 정의 단위

- [prefer] 사용자가 상품 행 수를 판매 수량으로 정의하면 해당 grain을 사용하되 결과에 item-line 기준임을 표시한다.
- [ask_if_missing] 고객·구매·판매가 고객/주문/상품 행 중 무엇인지에 따라 결과가 달라질 때만 확인한다.
