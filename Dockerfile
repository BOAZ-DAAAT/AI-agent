# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────
# 백엔드(FastAPI) 이미지 설계도
# 위에서 아래로 한 줄씩 실행하며 컨테이너 이미지를 쌓아 올린다.
# ─────────────────────────────────────────────────────────────

# 1) 베이스 이미지: 파이썬 3.12가 미리 깔린 가벼운 Debian(slim) 리눅스.
#    이 프로젝트가 요구하는 파이썬(>=3.12)에 맞춘다. slim = 불필요한 것 뺀 경량판.
FROM python:3.12-slim

# 2) 시스템 라이브러리 설치.
#    일부 과학/분석 패키지는 순수 파이썬이 아니라, OS 레벨 라이브러리가 있어야 동작한다.
#    - build-essential : 맞는 미리빌드(wheel)가 없을 때 소스에서 컴파일하기 위한 도구(gcc 등)
#    - libgomp1        : ortools / pymc / scikit-learn 등이 쓰는 병렬처리(OpenMP) 런타임
#    - curl            : 헬스체크·디버깅용
#    설치 후 apt 캐시를 지워(rm) 이미지 용량을 줄인다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 3) uv 설치. 이 레포는 uv.lock을 쓰므로 pip 대신 uv로 의존성을 재현한다.
#    공식 uv 이미지에서 실행 파일만 복사해 온다(가장 빠르고 깔끔한 방법).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 4) 작업 디렉터리. 이후 모든 명령과 앱 실행이 /app 기준으로 돈다.
#    앱의 저장 폴더(.data_agent)도 CWD 기준이라 /app/.data_agent 가 된다.
WORKDIR /app

# uv가 캐시를 하드링크가 아닌 복사 방식으로 쓰게 함(컨테이너 파일시스템에서 안전).
ENV UV_LINK_MODE=copy

# 5) 의존성 먼저 설치 → 코드 복사보다 앞에 둔다.
#    이유: Docker는 층 단위로 캐시한다. 코드만 바꿨을 때 의존성 설치 층을
#    다시 안 돌리고 캐시를 재사용 → 재빌드가 훨씬 빠르다.
#    --frozen: uv.lock을 그대로 사용(버전 고정 재현).
#    --no-install-project: 루트 프로젝트 자체는 설치 안 함(코드는 /app에서 바로 임포트).
#    --no-dev: 개발용 의존성 제외.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 6) 앱 코드 전체 복사(.dockerignore에 적힌 것은 제외됨).
COPY . .

# 7) uvicorn 설치.
#    FastAPI를 실제로 띄우는 실행기(ASGI 서버)인데, 이 레포 pyproject 의존성에는
#    빠져 있고 코드에서 지연 import만 한다. 그래서 이미지에 명시적으로 추가한다.
#    [standard]는 성능 옵션(websocket, uvloop 등)을 포함한 구성.
RUN uv pip install "uvicorn[standard]"

# 8) 설치된 가상환경(.venv)의 실행 파일을 PATH 앞에 둬서 uvicorn/python을 바로 부를 수 있게 함.
ENV PATH="/app/.venv/bin:$PATH"

# 9) 이 컨테이너가 8000번 포트를 쓴다는 것을 문서화(실제 공개는 compose에서 함).
EXPOSE 8000

# 10) 컨테이너가 시작될 때 실행할 명령.
#     핵심: --host 0.0.0.0  → 컨테이너 밖에서 접근 가능하게(코드 기본값 127.0.0.1이면 외부 불가).
#     create_app 은 앱을 만들어 주는 팩토리 함수라 --factory 를 붙인다.
CMD ["uvicorn", "data_agent_backend.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
