---
document_id: table-product-category-name-translation
doc_type: analysis_foundation
query_type: table_product_category_name_translation
title: Olist 카테고리 번역 테이블 계약서
language: ko
version: "1.0"
source_tables: [product_category_name_translation, products]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---
# Olist 카테고리 번역 테이블 계약서

## 검색 별칭

- table product category name translation, schema contract, grain, 안전한 사용

## 정의

- [must] 포르투갈어 카테고리명을 조인 키로 사용하고 번역이 없으면 원문 라벨을 유지한다.
- [must] 참조 source table에 있는 컬럼만 사용한다.
- [avoid] 스키마에 없는 비즈니스 필드를 추론하거나 grain을 묵시적으로 바꾸지 않는다.
- [prefer] 결과가 의존하는 지표, grain, 조인 경로를 명시한다.

## 컬럼 계약

- [must] `product_category_name`은 products의 포르투갈어 카테고리명과 연결하는 키다.
- [prefer] `product_category_name_english`는 보고용 영문 라벨이다.

## 조인 및 주의사항

- [must] `products.product_category_name = product_category_name_translation.product_category_name`로 연결한다.
- [avoid] 영문 라벨을 products에 대한 직접 조인 키로 사용하지 않는다.
- [prefer] 번역이 없는 카테고리는 누락으로 제거하지 말고 원문 카테고리명으로 유지한다.
