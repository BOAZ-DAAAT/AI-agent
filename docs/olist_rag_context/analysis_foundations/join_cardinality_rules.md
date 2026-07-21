---
document_id: join-cardinality-rules
doc_type: analysis_foundation
query_type: join_cardinality_rules
title: Olist 조인 및 카디널리티 규칙
language: ko
version: "1.0"
source_tables: [customers, geolocation, order_items, order_payments, order_reviews, orders, product_category_name_translation, products, sellers]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---

# Olist 조인 및 카디널리티 규칙

## 검색 별칭

- 조인, 카디널리티, 중복 집계, fanout, 선집계, 다대다 조인
- join, cardinality, duplicate aggregation, fanout, pre-aggregation, many-to-many

## 표준 조인 경로

- [must] `orders.customer_id = customers.customer_id`로 주문과 고객을 연결한다.
- [must] `orders.order_id`를 `order_items.order_id`, `order_payments.order_id`, `order_reviews.order_id`에 연결한다.
- [must] `order_items.product_id = products.product_id`, `order_items.seller_id = sellers.seller_id`를 사용한다.
- [must] 카테고리 번역은 `products.product_category_name = product_category_name_translation.product_category_name`로 연결한다.
- [prefer] 고객·판매자 zip prefix를 geolocation에 연결하기 전 zip prefix별 중복 좌표를 집계한다.

## Fanout 방지

- [must] 상품 행과 결제 행은 모두 주문당 여러 행일 수 있으므로 각각 주문 grain으로 선집계한 뒤 결합한다.
- [avoid] `order_items`와 `order_payments`를 직접 조인하고 `price`, `freight_value`, `payment_value`를 합산하지 않는다.
- [avoid] 주문 단위 리뷰를 여러 상품·판매자 행에 붙인 뒤 리뷰 수나 점수를 그대로 집계하지 않는다.
- [prefer] 다중행 테이블 조인 후 주문 수는 `COUNT(DISTINCT orders.order_id)`로 계산한다.

## 확인 조건

- [ask_if_missing] 주문 단위 결제금액 또는 리뷰 점수를 판매자·카테고리에 배분하려면 배분 정책을 확인하거나 가정을 명시한다.
