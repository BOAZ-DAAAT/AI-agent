---
document_id: table-contracts
doc_type: analysis_foundation
query_type: table_contracts
title: Olist 테이블 계약서
language: ko
version: "1.0"
source_tables: [customers, geolocation, order_items, order_payments, order_reviews, orders, product_category_name_translation, products, sellers]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---

# Olist 테이블 계약서

## Customers

- [must] `customers.customer_id`는 주문 조인 키이고, `customers.customer_unique_id`는 고객 생애 행동 식별자다.
- [prefer] 고객 지역은 `customer_city`, `customer_state`, `customer_zip_code_prefix`를 사용한다.

## Orders

- [must] `orders.order_id`는 주문 분석의 기본 키다.
- [prefer] 상태는 `order_status`, 주문 시점은 `order_purchase_timestamp`, 배송 지표는 실제·예상 고객 배송일을 사용한다.

## Order Items, Payments, Reviews

- [must] `order_items`와 `order_payments`는 주문당 여러 행을 가질 수 있다.
- [avoid] `payment_sequential`을 `payment_installments`와 같은 의미로 사용하지 않는다.
- [must] `order_reviews.review_id`가 리뷰 grain이고 `review_score`가 구조화된 만족도 지표다.

## Products, Sellers, Geography

- [must] 상품·판매자 마스터 키는 `products.product_id`, `sellers.seller_id`다.
- [prefer] 카테고리 번역 테이블로 영문 라벨을 보완하되 번역되지 않은 원문 카테고리는 보존한다.
- [avoid] geolocation은 zip prefix당 중복 행이 가능하므로 도시명 조인이나 fanout 무시는 피한다.
