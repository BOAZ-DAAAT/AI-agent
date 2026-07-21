---
document_id: table-sellers
doc_type: analysis_foundation
query_type: table_sellers
title: Olist 판매자 테이블 계약서
language: ko
version: "1.0"
source_tables: [sellers, order_items, geolocation]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---
# Olist 판매자 테이블 계약서

## 검색 별칭

- table sellers, schema contract, grain, 안전한 사용

## 정의

- [must] seller_id가 판매자 마스터 키이며 실제 판매 활동은 order_items로 측정한다.
- [must] 참조 source table에 있는 컬럼만 사용한다.
- [avoid] 스키마에 없는 비즈니스 필드를 추론하거나 grain을 묵시적으로 바꾸지 않는다.
- [prefer] 결과가 의존하는 지표, grain, 조인 경로를 명시한다.

## 컬럼 계약

- [must] `seller_id`가 판매자 마스터 grain이다.
- [prefer] `seller_zip_code_prefix`, `seller_city`, `seller_state`는 판매자 출발지·공급 지역 분석에 사용한다.

## 조인 및 주의사항

- [must] 관측된 판매 활동은 `order_items.seller_id = sellers.seller_id`로 연결한 상품 행에서 측정한다.
- [prefer] 좌표가 필요하면 seller zip prefix와 geolocation zip prefix를 prefix grain으로 연결한다.
- [avoid] sellers 마스터의 모든 행을 기간 내 활동 판매자로 해석하지 않는다.
