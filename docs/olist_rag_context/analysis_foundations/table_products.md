---
document_id: table-products
doc_type: analysis_foundation
query_type: table_products
title: Olist 상품 테이블 계약서
language: ko
version: "1.0"
source_tables: [products, order_items, product_category_name_translation]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---
# Olist 상품 테이블 계약서

## 검색 별칭

- table products, schema contract, grain, 안전한 사용

## 정의

- [must] product_id가 상품 grain이며 product_category_name이 사용 가능한 카테고리 속성이다.
- [must] 참조 source table에 있는 컬럼만 사용한다.
- [avoid] 스키마에 없는 비즈니스 필드를 추론하거나 grain을 묵시적으로 바꾸지 않는다.
- [prefer] 결과가 의존하는 지표, grain, 조인 경로를 명시한다.

## 컬럼 계약

- [must] `product_id`가 상품 마스터 grain이고 `product_category_name`이 사용 가능한 카테고리 속성이다.
- [prefer] `product_name_lenght`, `product_description_lenght`, `product_photos_qty`는 상품 정보의 길이·수량 특성이지 실제 이름·설명 본문이 아니다.
- [prefer] `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm`는 물류·크기 분석에 사용한다.

## 조인 및 주의사항

- [must] 판매 실적은 `order_items.product_id = products.product_id`로 연결한 상품 행을 기준으로 계산한다.
- [prefer] 영문 카테고리 라벨은 translation 테이블로 보완한다.
- [avoid] 상품 마스터 행 수를 판매 수량이나 주문 수로 해석하지 않는다.
