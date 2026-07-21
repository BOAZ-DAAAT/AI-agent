---
document_id: table-order-items
doc_type: analysis_foundation
query_type: table_order_items
title: Olist 주문 상품행 테이블 계약서
language: ko
version: "1.0"
source_tables: [order_items, orders, products, sellers]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---
# Olist 주문 상품행 테이블 계약서

## 검색 별칭

- table order items, schema contract, grain, 안전한 사용

## 정의

- [must] 한 주문에는 여러 상품 행이 있을 수 있으며 price와 freight_value는 상품행 grain 지표다.
- [must] 참조 source table에 있는 컬럼만 사용한다.
- [avoid] 스키마에 없는 비즈니스 필드를 추론하거나 grain을 묵시적으로 바꾸지 않는다.
- [prefer] 결과가 의존하는 지표, grain, 조인 경로를 명시한다.

## 컬럼 계약

- [must] `order_id`와 `order_item_id`의 조합이 상품 행 grain이다.
- [prefer] `product_id`, `seller_id`는 각각 상품·판매자 마스터 연결 키다.
- [default] `price`는 상품금액, `freight_value`는 상품 행에 부과된 배송비다.
- [prefer] `shipping_limit_date`는 판매자 출고 마감 질문에 사용하며 고객 예상 배송일과 혼동하지 않는다.

## 조인 및 주의사항

- [must] `order_items.order_id = orders.order_id`, `order_items.product_id = products.product_id`, `order_items.seller_id = sellers.seller_id`를 사용한다.
- [avoid] `order_item_id`를 주문 수 또는 명시적 수량 컬럼으로 해석하지 않는다.
- [avoid] 결제 행과 직접 조인한 뒤 `price`·`freight_value`·`payment_value`를 합산하지 않는다.
