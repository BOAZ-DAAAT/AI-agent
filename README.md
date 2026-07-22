# BOAZ Data Analysis Pipeline

이 저장소는 다음 흐름으로 동작하는 데이터 분석 파이프라인을 포함합니다.

```text
사용자 질문
→ Supervisor
→ SQL Agent
→ EDA Agent
→ Analysis Agent
→ Report 생성
```

추가로 프론트엔드에서 사용할 수 있도록, 로컬 MySQL 저장소와 경량 FastAPI backend도 함께 제공합니다.

---

## 1. 구성 요약

### 핵심 구성요소

- `DATA_Analyst_Assistant_Agent/`
  - LangGraph 기반 분석 오케스트레이션 파이프라인
  - SQL / EDA / Analysis / Report 에이전트 포함
- `backend/`
  - MySQL 탐색/적재용 경량 FastAPI 서버
- `docker-compose.yml`
  - MySQL + backend + frontend 기동용 compose 설정
- `seed/`
  - MySQL 초기 스키마/시드 스크립트
- `scripts/run_orchestration_demo.py`
  - 오케스트레이션 데모 실행 스크립트

### 언제 무엇을 실행하면 되나

- **분석 파이프라인 자체를 돌리고 싶다**
  - `python -m DATA_Analyst_Assistant_Agent.run ...`
- **MySQL 저장소 + API 서버를 함께 띄우고 싶다**
  - `docker compose up -d mysql backend`
- **프론트엔드까지 포함한 전체 스택을 띄우고 싶다**
  - 형제 디렉터리에 `../Frontend` 레포가 있어야 하며 `docker compose up -d` 실행

---

## 2. 요구사항

### 로컬 실행

- Python **3.12+**
- MySQL 접근 가능 환경
- LLM API Key 1개 이상
  - `OPENROUTER_API_KEY` 또는
  - `OPENAI_API_KEY` 또는
  - `GOOGLE_API_KEY` / `GEMINI_API_KEY`

### Docker 실행

- Docker
- Docker Compose

---

## 3. 환경변수 설정

루트에서 `.env` 파일을 준비합니다.

```bash
cp .env.example .env
```

최소한 아래 값들은 확인해서 채워주세요.

```env
# 분석 대상 MySQL(datasource)
DB_HOST=localhost
DB_PORT=3306
DB_NAME=sql_agent
DB_USER=root
DB_PASSWORD=1234
MYSQL_DATASOURCE_NAME=default_mysql

# 로컬 datamart / compose MySQL
MYSQL_ROOT_PASSWORD=change_me_root_pw
DATAMART_DB_USER=your_datamart_user
DATAMART_DB_PASSWORD=your_datamart_password
DATAMART_DB_HOST=127.0.0.1
DATAMART_DB_PORT=3306
DATAMART_DB_NAME=analytics

# backend가 ingest / integrity queue 실행 시 사용할 storage mysql
STORAGE_MYSQL_HOST=127.0.0.1
STORAGE_MYSQL_PORT=3306
STORAGE_MYSQL_USER=root
STORAGE_MYSQL_PASSWORD=change_me_root_pw

# backend api / cors
CORS_ALLOW_ORIGINS=*
DATA_AGENT_BACKEND_URL=http://localhost:8000
DATA_AGENT_BASE_DIR=.data_agent

# LLM
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=http://localhost
OPENROUTER_APP_TITLE=BOAZ DATA Analyst Assistant Agent
LLM_MODEL=~openai/gpt-latest
```

### 환경변수 주의사항

- 파이프라인의 SQL Agent는 내부적으로 `DB_*` 값을 기준으로 MySQL에 접속합니다.
- ingest / integrity backend는 내부적으로 `STORAGE_MYSQL_*` 값을 기준으로 로컬 storage MySQL에 접속합니다.
- `.env.example`에는 `MYSQL_USER`, `MYSQL_PASSWORD`도 같이 보이는데, 실제 분석 실행 시에는 **`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_PASSWORD`가 올바른지** 먼저 확인하는 것이 안전합니다.
- 로컬에서 `uvicorn backend.main:app ...` 으로 backend를 직접 띄우면 `STORAGE_MYSQL_HOST`, `STORAGE_MYSQL_PORT`, `STORAGE_MYSQL_USER`, `STORAGE_MYSQL_PASSWORD` 도 같이 맞춰야 합니다.
- `docker compose`로 띄운 저장소 MySQL을 분석 대상으로 바로 쓰려면 보통 다음처럼 맞추면 됩니다.

```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=analytics
DB_USER=root
DB_PASSWORD=<MYSQL_ROOT_PASSWORD와 동일한 값>
```

---

## 4. 설치

### 방법 A. `uv` 사용 권장

```bash
uv sync
```

### 방법 B. `venv + pip`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-backend.txt
```

Runtime note: the analysis code runner is pinned to `pandas==2.3.3`
with `numpy==2.4.6`. Do not upgrade the analysis environment to
`pandas>=3`; `pandas 3.0.4` on Windows was observed to crash the
subprocess runner in daily datetime range operations.

---

## 5. 실행 방법

### 5-1. 가장 빠른 로컬 파이프라인 실행

Supervisor 기반 분석 파이프라인을 직접 실행합니다.

```bash
python -m DATA_Analyst_Assistant_Agent.run "최근 6개월 월별 매출 추세를 분석해줘"
```

유용한 옵션:

```bash
python -m DATA_Analyst_Assistant_Agent.run \
  "카테고리별 매출 차이를 비교해줘" \
  --show-sql \
  --no-open
```

CLI 도움말:

```bash
python -m DATA_Analyst_Assistant_Agent.run --help
```

### 실행 결과물

기본적으로 아래 폴더에 결과가 저장됩니다.

```text
daaa_outputs/latest/
```

대표 출력물:

- `run_summary.json`
- `generated_sql.sql`
- `final_report.html`
- `final_report.md`
- `sql_result.csv`
- `artifacts/`

---

### 5-2. 질의 중간에 추가 입력이 필요한 경우

이 파이프라인은 실행 중 clarification 또는 approval 단계에서 멈출 수 있습니다.

### 같은 터미널에서 바로 이어서 진행

```bash
python -m DATA_Analyst_Assistant_Agent.run \
  "매출 분석해줘" \
  --interactive
```

### 비대화식으로 실행 후 재개 명령만 출력

```bash
python -m DATA_Analyst_Assistant_Agent.run \
  "매출 분석해줘" \
  --no-interactive
```

이 경우 출력된 `thread_id`와 `resume command`를 사용해서 이어서 실행합니다.

#### clarification 재개

```bash
python -m DATA_Analyst_Assistant_Agent.run \
  --thread-id <중단된_thread_id> \
  --resume-answer "최근 6개월 월별 매출 기준으로 분석해줘"
```

#### approval 재개

```bash
python -m DATA_Analyst_Assistant_Agent.run \
  --thread-id <중단된_thread_id> \
  --approve
```

---

### 5-3. 데모 스크립트 실행

HTML 결과를 포함한 데모 실행이 필요하면:

```bash
python scripts/run_orchestration_demo.py --help
python scripts/run_orchestration_demo.py "최근 3개월 지역별 매출과 이상치를 요약해줘"
```

---

### 5-4. MySQL + backend만 Docker로 실행

프론트 없이 저장소와 API 서버만 띄우려면:

```bash
docker compose up -d mysql backend
```

확인:

```bash
curl http://localhost:8000/health
```

이 backend는 다음 역할을 담당합니다.

- 원격 MySQL의 DB 목록 조회
- 테이블 목록 조회
- 테이블 preview 조회
- 원격 DB를 로컬 저장소 MySQL로 ingest

주요 엔드포인트:

- `GET /health`
- `POST /mysql/databases`
- `POST /mysql/tables`
- `POST /mysql/preview`
- `POST /storage/ingest`
- `GET /storage/databases`
- `GET /storage/{database}/tables`
- `GET /storage/{database}/tables/{table}/preview`

---

### 5-5. 프론트엔드 포함 전체 스택 실행

`docker-compose.yml`의 frontend 서비스는 **형제 디렉터리의 `../Frontend` 레포**를 빌드하도록 되어 있습니다.

즉 현재 구조가 아래처럼 준비되어 있어야 합니다.

```text
<parent>/
├── git_base/
└── Frontend/
```

그 다음 전체 실행:

```bash
docker compose up -d
```

접속 주소:

- frontend: `http://localhost`
- backend: `http://localhost:8000`
- mysql: `127.0.0.1:3306`

> `../Frontend`가 없으면 `frontend` 빌드가 실패합니다. 이 경우 `mysql backend`만 선택해서 올리면 됩니다.

---

## 6. 검증 방법

### 1) CLI 스모크 체크

```bash
python -m DATA_Analyst_Assistant_Agent.run --help
```

### 2) backend 헬스체크

```bash
curl http://localhost:8000/health
```

### 3) 테스트 실행

```bash
pytest tests DATA_Analyst_Assistant_Agent/tests data_agent_backend/tests -q
```

필요 시 특정 범위만 먼저 확인:

```bash
pytest tests/test_sql_agent.py -q
pytest tests/test_sql_agent_orchestration.py -q
```

---

## 7. 자주 겪는 문제

### 1) DB 연결 실패

증상:

- `DB engine creation failed`
- `DB 연결 실패`

확인할 것:

- `.env`의 `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- MySQL이 실제로 떠 있는지
- compose MySQL을 쓰는 경우 `MYSQL_ROOT_PASSWORD`와 `DB_PASSWORD`가 일치하는지

### 2) LLM 호출 실패

확인할 것:

- `OPENROUTER_API_KEY` 또는 `OPENAI_API_KEY` 또는 `GOOGLE_API_KEY` 설정 여부
- `LLM_MODEL` 이름이 현재 사용 가능한 모델인지

### 3) frontend compose 빌드 실패

원인:

- `../Frontend` 디렉터리가 없음

대응:

```bash
docker compose up -d mysql backend
```

### 4) 결과 HTML이 자동으로 안 열림

일부 환경에서는 자동 open이 동작하지 않을 수 있습니다.

그 경우 결과 폴더의 아래 파일을 직접 열면 됩니다.

```text
daaa_outputs/latest/index.html
```

---

## 8. 추천 실행 순서

### A. 분석 파이프라인만 확인하고 싶을 때

1. `.env` 작성
2. Python 의존성 설치
3. 분석 대상 MySQL 준비
4. `python -m DATA_Analyst_Assistant_Agent.run "질문"`

### B. 저장소 MySQL + backend API까지 함께 확인하고 싶을 때

1. `.env` 작성
2. `docker compose up -d mysql backend`
3. `curl http://localhost:8000/health`
4. 필요 시 `python -m DATA_Analyst_Assistant_Agent.run "질문"`

### C. 프론트까지 포함해 전체 데모를 띄우고 싶을 때

1. `../Frontend` 레포 준비
2. `.env` 작성
3. `docker compose up -d`
4. `http://localhost` 접속

---

## 9. 참고 문서

- `docs/analysis_agent_team_guide.md`
- `docs/data_analysis_agent.md`
- `docs/sql_agent_architecture/architecture.ko.md`
- `docs/sql_agent_architecture/prd-sql-agent-architecture.ko.md`
