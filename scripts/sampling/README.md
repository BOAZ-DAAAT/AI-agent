# Olist 샘플 DB 만들기

원본 `olist` DB(약 10만 행)가 개발 반복에 너무 느려서, `customer_unique_id` 기준 층화 샘플링으로
`olist_sampling`이라는 축소 DB를 별도로 만든다. 원본은 건드리지 않는다.

## 왜 customer_unique_id 기준인가

`customer_id`는 PK지만 주문마다 새로 발급되는 계정이라, 이 기준으로 뽑으면 같은 사람의 주문
이력이 쪼개져서 RFM의 frequency/monetary가 왜곡된다. `customer_unique_id`가 진짜 사람이다.

## 실행 순서

```bash
# 1. 샘플 DB 생성 (기본 1만 명, 대표성 70% + 희귀케이스 30%)
python scripts/sampling/sample_olist.py

# 2. 원본과 분포 비교 (결측 비율, 결제액 분위수, 리뷰 점수 분포)
python scripts/sampling/compare_distributions.py

# 3. GE는 별도로 확인 (.venv-ge 안에서)
python -c "
from DATA_Analyst_Assistant_Agent.agents.sql.db.integrity_runner import run_dataset_integrity_checks
run_dataset_integrity_checks(dataset_name='olist_sampling')
"
```

## 옵션

```bash
python scripts/sampling/sample_olist.py \
  --source-db olist \
  --target-db olist_sampling \
  --target-customers 10000 \
  --representative-ratio 0.7 \
  --seed 42
```

같은 `--seed`면 같은 샘플이 재현된다(팀원끼리 공유 가능).

## 샘플링 로직 요약

1. `customers` + `orders` + `order_payments` + `order_reviews`를 조인해서 고객별 집계(주문횟수,
   총결제액, 평균리뷰점수, 첫 주문일)를 뽑는다.
2. 4축으로 층화한다: 주문횟수(1건/2~3건/4건+) x 결제액 5분위 x 리뷰(없음/1~2/3/4~5점) x 분기.
3. 목표 인원의 70%는 층화 비율대로, 30%는 고액상위5%·다회구매·리뷰없음·저평점 같은 희귀케이스를
   의도적으로 오버샘플해서 뽑는다(랜덤이면 씹혀나가기 쉬운 케이스라서).
4. `customers` -> `orders` -> `order_items`/`order_payments`/`order_reviews` 순서로, 부모가
   뽑힌 것만 자식도 따라가는 계단식 복사를 한다(독립적으로 다시 샘플링하지 않음 — 그러면 조인이
   깨진다).
5. `products`/`sellers`/`product_category_name_translation`/`geolocation`은 카디널리티가 낮아
   원본을 그대로 복사한다.

## 개발 중 전환

`.env`의 `MYSQL_DATABASE`를 `olist_sampling`으로 바꾸면 SQL 에이전트가 샘플 DB를 본다. 최종
확인할 땐 다시 `olist`로 돌려놓는다.
