# syntax=docker/dockerfile:1

# 파이썬 3.12 경량판. 분석 라이브러리가 없으므로 컴파일러 등 무거운 준비물 불필요
FROM python:3.12-slim

# curl: 컨테이너 헬스체크용
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 → 코드만 바뀌면 이 레이어는 캐시 재사용 (재빌드 몇 초)
COPY requirements-backend.txt ./
RUN pip install --no-cache-dir -r requirements-backend.txt

# backend 패키지만 복사 — 에이전트/구백엔드/테스트는 이 이미지에 불필요
COPY backend/ ./backend/

EXPOSE 8000

# 0.0.0.0: 컨테이너 밖에서 접근 가능하게
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
