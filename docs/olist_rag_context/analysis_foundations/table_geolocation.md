---
document_id: table-geolocation
doc_type: analysis_foundation
query_type: table_geolocation
title: Olist 지리 좌표 테이블 계약서
language: ko
version: "1.0"
source_tables: [geolocation, customers, sellers]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
---
# Olist 지리 좌표 테이블 계약서

## 검색 별칭

- table geolocation, schema contract, grain, 안전한 사용

## 정의

- [must] 우편번호 prefix에는 중복 좌표 행이 있을 수 있으므로 조인 전에 prefix별 집계한다.
- [must] 참조 source table에 있는 컬럼만 사용한다.
- [avoid] 스키마에 없는 비즈니스 필드를 추론하거나 grain을 묵시적으로 바꾸지 않는다.
- [prefer] 결과가 의존하는 지표, grain, 조인 경로를 명시한다.

## 컬럼 계약

- [must] `geolocation.geolocation_zip_code_prefix`는 고객·판매자 zip prefix와 연결하는 지역 코드다.
- [prefer] `geolocation_lat`, `geolocation_lng`는 지도·거리·좌표 요청에만 사용한다.
- [prefer] `geolocation_city`, `geolocation_state`는 zip prefix에 매핑된 보조 지역 라벨이다.

## 조인 및 주의사항

- [must] 고객은 `customers.customer_zip_code_prefix`, 판매자는 `sellers.seller_zip_code_prefix`를 geolocation zip prefix와 연결한다.
- [avoid] city 이름만으로 geolocation을 조인하지 않는다.
- [avoid] 같은 zip prefix에 여러 좌표 행이 있을 수 있으므로 원본 geolocation을 그대로 조인해 주문 수·금액을 집계하지 않는다.
