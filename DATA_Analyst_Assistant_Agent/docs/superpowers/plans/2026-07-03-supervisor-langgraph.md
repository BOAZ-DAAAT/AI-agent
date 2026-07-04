# Supervisor LangGraph Implementation Plan

## 절대 구현 제약

- `agents/sql/`, `agents/eda/`, `agents/analysis/`, `agents/report/` 하위에 현재 구현되어 있는 Sub-Agent 코드는 절대 변경하지 않는다.
- Sub-Agent 실행 중 문제가 발견되더라도 먼저 새 `supervisor/` 계층의 adapter, state 변환, validation, 테스트 더블 범위에서 해결한다.
- Sub-Agent 내부 수정이 없이는 진행할 수 없는 결함이 확인되면 구현을 중단하고, 어떤 파일의 어떤 계약이 맞지 않는지 기록한 뒤 사용자 승인을 받기 전까지 변경하지 않는다.
- 실패를 가정한 fallback logic은 최소화한다. fallback은 런타임 장애, LLM 응답 파싱 실패, Sub-Agent 입력 계약 보호, 안전한 종료 상태 기록처럼 꼭 필요한 경우에만 구현한다.
- 하드 코딩은 기존 계약을 보호하는 상수, 테스트 더블, 명시적 enum/action 이름에 한정한다. 분석 순서, 결과 판단, 성공 응답, Sub-Agent 산출물을 흉내 내는 하드 코딩은 구현하지 않는다.
- Supervisor가 고정된 `SQL -> EDA -> Analysis -> Report` 순서를 주 실행 전략으로 강제하지 않는다. 필요한 최소 guardrail만 적용하고, 다음 행동 판단은 state, validation 결과, artifact evidence, plan에 근거해야 한다.
- fallback은 성공처럼 위장하면 안 된다. 충분한 근거가 없으면 `needs_clarification`, `failed_with_recoverable_context`, `failed_terminal` 중 하나로 명시적으로 종료한다.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 SQL, EDA, Analysis, Report Sub-Agent 내부 구현은 유지하면서 새 `supervisor/` 패키지에 LangGraph 기반 상위 Supervisor를 추가한다.

**Architecture:** Supervisor는 LangGraph workflow로 전체 상태, 계획, 실행, 검증, 요약, 종료를 관리한다. 기존 Sub-Agent는 `SubAgentAdapter`를 통해 `Agent.run(state, runtime)` 계약 그대로 호출하고, checkpoint에는 재개 판단에 필요한 compact state와 artifact 참조만 저장한다.

**Tech Stack:** Python 3.12, Pydantic v2, LangGraph, `langgraph-checkpoint-sqlite`, LangChain Core, 기존 `BackendAdapter`, 기존 `AgentEnvelope`/`OrchestrationState` 계약

---

## 읽은 요구사항 문서

- `Docs/supervisor-langgraph-design-summary.md`
- `Docs/supervisor-checkpoint-memory-decisions.md`

두 파일은 `docs/`에도 동일한 내용으로 중복 존재한다. 구현 계획 문서는 skill 기본 경로와 기존 소문자 `docs/` 폴더에 맞춰 저장한다.

## 구현 범위

- 새 `supervisor/` 패키지를 만든다.
- 기존 `agents/sql/`, `agents/eda/`, `agents/analysis/`, `agents/report/` 내부 workflow와 현재 구현 코드는 절대 수정하지 않는다.
- 삭제된 validation/visualization agent export 흔적은 공개 API 정리 범위로만 수정한다.
- 기존 `SQLAgentSupervisor` 이름은 하위 호환 alias로 유지하되 새 구현은 `SupervisorAgent`가 담당한다.
- checkpoint는 `.data_agent/checkpoints/supervisor.sqlite`에 저장한다.
- 이번 단계에서는 LangGraph Store, 장기 메모리, vector memory, checkpoint cleanup, time travel/fork, 특정 checkpoint id resume을 구현하지 않는다.

## 파일 구조

- Modify: `../pyproject.toml`
  - `langgraph-checkpoint-sqlite` 의존성을 추가한다.
- Modify: `../requirements.txt`
  - 로컬 설치 환경용 동일 의존성을 추가한다.
- Create: `supervisor/__init__.py`
  - `SupervisorAgent`, `SQLAgentSupervisor`, `build_graph`를 공개한다.
- Create: `supervisor/state.py`
  - checkpoint 저장 가능한 `SupervisorState`와 compact result 모델을 정의한다.
- Create: `supervisor/checkpoint.py`
  - SQLite checkpointer 경로와 context manager를 제공한다.
- Create: `supervisor/tools.py`
  - 기존 4개 Sub-Agent를 adapter tool 형태로 호출한다.
- Create: `supervisor/validation.py`
  - 실행 전 guardrail과 실행 후 validation decision을 담당한다.
- Create: `supervisor/summarizer.py`
  - 각 단계별 step summary를 compact하게 생성한다.
- Create: `supervisor/prompts.py`
  - clarification, planning, next action prompt를 분리한다.
- Create: `supervisor/decision.py`
  - LLM JSON 응답 파싱과 최소 안전 fallback decision을 제공한다.
- Create: `supervisor/graph.py`
  - Supervisor LangGraph node와 edge를 구성한다.
- Create: `supervisor/agent.py`
  - 외부 실행 진입점, backend run 생성, checkpoint invoke, resume API를 담당한다.
- Modify: `__init__.py`
  - 새 Supervisor export로 전환하고 삭제된 agent export를 제거한다.
- Modify: `agents/__init__.py`
  - 삭제된 validation/visualization lazy export를 제거한다.
- Modify: `run.py`
  - CLI 설명과 import를 새 Supervisor 기준으로 정리한다.
- Create: `tests/supervisor/test_state.py`
  - state 생성, compact result 병합, OrchestrationState 변환을 검증한다.
- Create: `tests/supervisor/test_checkpoint.py`
  - checkpoint DB 경로 생성과 saver context를 검증한다.
- Create: `tests/supervisor/test_tools.py`
  - adapter가 기존 Sub-Agent 계약을 유지하고 artifact를 compact하게 누적하는지 검증한다.
- Create: `tests/supervisor/test_validation.py`
  - Sub-Agent 실행 전 guardrail과 envelope validation 판단을 검증한다.
- Create: `tests/supervisor/test_decision.py`
  - LLM decision JSON 파싱과 fallback의 안전 종료 동작을 검증한다.
- Create: `tests/supervisor/test_graph.py`
  - fake adapter와 fake decision model로 graph가 state 기반 next action을 따르는지 검증한다.
- Create: `tests/test_public_api.py`
  - package import와 하위 호환 export를 검증한다.

### Task 1: Supervisor 의존성 추가

**Files:**
- Modify: `../pyproject.toml`
- Modify: `../requirements.txt`

- [ ] **Step 1: 의존성 변경 전 import 실패 확인**

Run:

```bash
python3 - <<'PY'
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    print("unexpected ok", SqliteSaver)
except Exception as exc:
    print(type(exc).__name__, exc)
PY
```

Expected:

```text
ModuleNotFoundError No module named 'langgraph'
```

현재 로컬 환경에는 LangGraph가 설치되어 있지 않다. 의존성 파일 수정 후 실제 설치는 실행자가 프로젝트 환경에서 수행한다.

- [ ] **Step 2: `../pyproject.toml`에 SQLite checkpointer 의존성 추가**

Find this dependency block and add `langgraph-checkpoint-sqlite` next to `langgraph`.

```toml
dependencies = [
    "dowhy>=0.14",
    "duckdb>=1.5.4",
    "esda>=2.10.0",
    "fastapi>=0.138.1",
    "geopandas>=1.1.4",
    "langchain-core>=1.4.8",
    "langchain-google-genai>=4.2.6",
    "langchain-openai>=1.3.3",
    "langgraph>=1.2.6",
    "langgraph-checkpoint-sqlite",
    "libpysal>=4.14.1",
    "matplotlib>=3.11.0",
    "numpy>=2.4.6",
    "ortools>=9.15.6755",
    "pandas>=3.0.4",
    "pydantic>=2.13.4",
    "pymc-marketing==0.19.3",
    "pymysql>=1.2.0",
    "pytest>=9.1.1",
    "python-dotenv>=1.2.2",
    "scikit-learn>=1.9.0",
    "seaborn>=0.13.2",
    "sqlalchemy>=2.0.51",
    "sqlglot>=30.12.0",
    "statsmodels>=0.14.6",
]
```

- [ ] **Step 3: `../requirements.txt`에 동일 의존성 추가**

Add this line immediately after `langgraph`.

```text
langgraph-checkpoint-sqlite
```

- [ ] **Step 4: 설치된 환경에서 import 확인**

Run after dependency installation:

```bash
python3 - <<'PY'
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

print(SqliteSaver.__name__)
print(StateGraph.__name__)
PY
```

Expected:

```text
SqliteSaver
StateGraph
```

- [ ] **Step 5: Commit**

```bash
git add ../pyproject.toml ../requirements.txt
git commit -m "feat: add supervisor checkpoint dependency"
```

### Task 2: Supervisor state 계약 정의

**Files:**
- Create: `supervisor/state.py`
- Test: `tests/supervisor/test_state.py`

- [ ] **Step 1: 실패하는 state 테스트 작성**

Create `tests/supervisor/test_state.py`.

```python
from __future__ import annotations

from DATA_Analyst_Assistant_Agent.shared.contracts import SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    ArtifactSummary,
    empty_supervisor_state,
    merge_agent_result,
    to_orchestration_state,
)


def test_empty_supervisor_state_uses_compact_defaults() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )

    assert state["thread_id"] == "thread_sales_001"
    assert state["current_run_id"] == "run_001"
    assert state["run_ids"] == ["run_001"]
    assert state["latest_user_query"] == "월별 매출 추이를 분석해줘"
    assert state["user_turns"] == [{"run_id": "run_001", "query": "월별 매출 추이를 분석해줘"}]
    assert state["agent_results"] == []
    assert state["artifacts"] == {}
    assert state["terminal_state"] == "running"


def test_merge_agent_result_adds_artifacts_and_completion() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 실행 완료",
        artifact_ids=["artifact_sql_result"],
        artifacts=[
            ArtifactSummary(
                artifact_id="artifact_sql_result",
                type="sql_result",
                kind="sql_result",
                summary="10 rows",
            )
        ],
    )

    merged = merge_agent_result(state, result)

    assert merged["completed_agents"] == ["sql_agent"]
    assert merged["failed_agents"] == []
    assert merged["agent_results"][0]["summary"] == "SQL 실행 완료"
    assert merged["artifacts"]["sql_agent"][0]["artifact_id"] == "artifact_sql_result"


def test_to_orchestration_state_preserves_existing_agent_artifacts() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id="ds_001",
    )
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 실행 완료",
        artifact_ids=["artifact_sql_result"],
    )
    state = merge_agent_result(state, result)
    state["generated_sql"] = "SELECT 1 AS sample_value"
    state["terminal_state"] = SupervisorTerminalState.completed.value

    orchestration = to_orchestration_state(state)

    assert orchestration.run_id == "run_001"
    assert orchestration.thread_id == "thread_sales_001"
    assert orchestration.datasource_id == "ds_001"
    assert orchestration.user_query == "월별 매출 추이를 분석해줘"
    assert orchestration.artifact_ids == {"sql_agent": ["artifact_sql_result"]}
    assert orchestration.generated_sql == "SELECT 1 AS sample_value"
    assert orchestration.terminal_state == SupervisorTerminalState.completed
```

- [ ] **Step 2: 테스트 실패 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_state.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'DATA_Analyst_Assistant_Agent.supervisor'
```

- [ ] **Step 3: `supervisor/state.py` 구현**

Create `supervisor/state.py`.

```python
from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.shared.contracts import (
    AnalysisPlan,
    OrchestrationState,
    SupervisorTerminalState,
)


AgentName = Literal["sql_agent", "eda_agent", "analysis_agent", "report_agent"]
AgentStatusValue = Literal["success", "warning", "failed", "approval_required"]
NextAction = Literal[
    "clarify",
    "create_plan",
    "call_sql_agent",
    "call_eda_agent",
    "call_analysis_agent",
    "call_report_agent",
    "finalize",
    "fail",
]


class ArtifactSummary(BaseModel):
    artifact_id: str
    type: str = ""
    kind: str = ""
    summary: str = ""
    uri: str | None = None


class AgentCompactResult(BaseModel):
    agent: AgentName
    status: AgentStatusValue
    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    retryable: bool = False
    error: str = ""


class StepSummary(BaseModel):
    step: str
    agent: AgentName | None = None
    action: str
    summary: str
    artifact_ids: list[str] = Field(default_factory=list)
    next_action: str = ""


class PendingApproval(BaseModel):
    approval_id: str
    agent: AgentName
    reason: str
    approval_type: str


class SupervisorState(TypedDict, total=False):
    thread_id: str
    current_run_id: str
    run_ids: list[str]
    project_id: str | None
    datasource_id: str | None
    catalog_summary: dict[str, Any] | None
    latest_user_query: str
    user_turns: list[dict[str, str]]
    clarified_query: str
    needs_clarification: bool
    clarification_question: str
    analysis_plan: dict[str, Any]
    current_step: str
    next_action: NextAction
    terminal_state: str
    final_answer: str
    agent_results: list[dict[str, Any]]
    artifacts: dict[str, list[dict[str, Any]]]
    validation_results: list[dict[str, Any]]
    step_summaries: list[dict[str, Any]]
    completed_agents: list[str]
    failed_agents: list[str]
    pending_approval: dict[str, Any] | None
    generated_sql: str
    error_state: dict[str, Any]
    retry_counts: dict[str, int]
    max_retry_per_agent: int


def empty_supervisor_state(
    *,
    thread_id: str,
    run_id: str,
    user_query: str,
    datasource_id: str | None,
    project_id: str | None = None,
    catalog_summary: dict[str, Any] | None = None,
) -> SupervisorState:
    return {
        "thread_id": thread_id,
        "current_run_id": run_id,
        "run_ids": [run_id],
        "project_id": project_id,
        "datasource_id": datasource_id,
        "catalog_summary": catalog_summary,
        "latest_user_query": user_query,
        "user_turns": [{"run_id": run_id, "query": user_query}],
        "clarified_query": "",
        "needs_clarification": False,
        "clarification_question": "",
        "analysis_plan": {},
        "current_step": "created",
        "next_action": "create_plan",
        "terminal_state": "running",
        "final_answer": "",
        "agent_results": [],
        "artifacts": {},
        "validation_results": [],
        "step_summaries": [],
        "completed_agents": [],
        "failed_agents": [],
        "pending_approval": None,
        "generated_sql": "",
        "error_state": {},
        "retry_counts": {},
        "max_retry_per_agent": 1,
    }


def merge_agent_result(state: SupervisorState, result: AgentCompactResult) -> SupervisorState:
    agent_results = list(state.get("agent_results", []))
    result_payload = result.model_dump(mode="json")
    agent_results.append(result_payload)

    artifacts = {agent: list(items) for agent, items in state.get("artifacts", {}).items()}
    artifacts.setdefault(result.agent, [])
    artifacts[result.agent].extend(artifact.model_dump(mode="json") for artifact in result.artifacts)

    completed_agents = list(state.get("completed_agents", []))
    failed_agents = list(state.get("failed_agents", []))
    if result.status in {"success", "warning", "approval_required"} and result.agent not in completed_agents:
        completed_agents.append(result.agent)
    if result.status == "failed" and result.agent not in failed_agents:
        failed_agents.append(result.agent)

    merged: SupervisorState = {
        **state,
        "agent_results": agent_results,
        "artifacts": artifacts,
        "completed_agents": completed_agents,
        "failed_agents": failed_agents,
    }
    if result.error:
        merged["error_state"] = {
            "agent": result.agent,
            "message": result.error,
            "retryable": result.retryable,
        }
    return merged


def artifact_ids_by_agent(state: SupervisorState) -> dict[str, list[str]]:
    ids: dict[str, list[str]] = {}
    for result in state.get("agent_results", []):
        agent = str(result.get("agent", ""))
        if not agent:
            continue
        ids.setdefault(agent, [])
        for artifact_id in result.get("artifact_ids", []):
            if artifact_id not in ids[agent]:
                ids[agent].append(str(artifact_id))
    return ids


def to_orchestration_state(state: SupervisorState) -> OrchestrationState:
    terminal_value = state.get("terminal_state")
    terminal_state = None
    if terminal_value and terminal_value != "running":
        terminal_state = SupervisorTerminalState(str(terminal_value))

    plan_payload = state.get("analysis_plan") or {}
    plan = AnalysisPlan(
        goal=str(plan_payload.get("goal") or state.get("latest_user_query") or ""),
        datasource_id=state.get("datasource_id"),
        catalog_summary=state.get("catalog_summary"),
        route_kind=str(plan_payload.get("route_kind") or "comprehensive"),
        generated_sql=state.get("generated_sql") or "SELECT 1 AS sample_value",
        source_sql=state.get("generated_sql") or "SELECT 1 AS sample_value",
        planner_mode="llm" if plan_payload.get("planner_mode") == "llm" else "deterministic",
    )

    return OrchestrationState(
        run_id=state["current_run_id"],
        thread_id=state.get("thread_id"),
        datasource_id=state.get("datasource_id"),
        catalog_summary=state.get("catalog_summary"),
        user_query=state.get("clarified_query") or state.get("latest_user_query", ""),
        goal=plan.goal,
        plan=plan,
        current_step=state.get("current_step", "created"),
        artifact_ids=artifact_ids_by_agent(state),
        terminal_state=terminal_state,
        remaining_agents=[],
        completed_agents=list(state.get("completed_agents", [])),
        route_kind=plan.route_kind,
        planner_mode=plan.planner_mode,
        generated_sql=state.get("generated_sql", ""),
        retry_counts=dict(state.get("retry_counts", {})),
        max_retry_per_agent=int(state.get("max_retry_per_agent", 1)),
        error_state=dict(state.get("error_state", {})),
    )
```

- [ ] **Step 4: state 테스트 통과 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_state.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add supervisor/state.py tests/supervisor/test_state.py
git commit -m "feat: add supervisor state contracts"
```

### Task 3: SQLite checkpoint factory 구현

**Files:**
- Create: `supervisor/checkpoint.py`
- Test: `tests/supervisor/test_checkpoint.py`

- [ ] **Step 1: 실패하는 checkpoint 테스트 작성**

Create `tests/supervisor/test_checkpoint.py`.

```python
from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.checkpoint import open_sqlite_checkpointer


def test_open_sqlite_checkpointer_creates_parent_directory(tmp_path) -> None:
    db_path = tmp_path / "checkpoints" / "supervisor.sqlite"

    with open_sqlite_checkpointer(db_path) as saver:
        assert saver is not None

    assert db_path.exists()
```

- [ ] **Step 2: 테스트 실패 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_checkpoint.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'DATA_Analyst_Assistant_Agent.supervisor.checkpoint'
```

- [ ] **Step 3: `supervisor/checkpoint.py` 구현**

Create `supervisor/checkpoint.py`.

```python
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


DEFAULT_CHECKPOINT_PATH = Path(".data_agent") / "checkpoints" / "supervisor.sqlite"


@contextmanager
def open_sqlite_checkpointer(path: str | Path | None = None) -> Iterator[SqliteSaver]:
    checkpoint_path = Path(path) if path else DEFAULT_CHECKPOINT_PATH
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        yield saver
```

- [ ] **Step 4: checkpoint 테스트 통과 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_checkpoint.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add supervisor/checkpoint.py tests/supervisor/test_checkpoint.py
git commit -m "feat: add supervisor sqlite checkpoint factory"
```

### Task 4: Sub-Agent Adapter Tool 구현

**Files:**
- Create: `supervisor/tools.py`
- Test: `tests/supervisor/test_tools.py`

- [ ] **Step 1: 실패하는 adapter 테스트 작성**

Create `tests/supervisor/test_tools.py`.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, AgentStatus, OrchestrationState
from DATA_Analyst_Assistant_Agent.supervisor.state import empty_supervisor_state
from DATA_Analyst_Assistant_Agent.supervisor.tools import SubAgentAdapter


@dataclass
class FakeArtifactRef:
    artifact_id: str


@dataclass
class FakeArtifactRecord:
    artifact_id: str
    type: str
    uri: str | None = None
    metadata: dict[str, Any] | None = None
    preview: dict[str, Any] | None = None


class FakeAdapter:
    base_data_dir = ".data_agent"

    def get_artifact(self, artifact_id: str) -> FakeArtifactRecord:
        return FakeArtifactRecord(
            artifact_id=artifact_id,
            type="sql_result",
            metadata={"kind": "sql_result"},
            preview={"row_count": 3},
        )


class StubAgent:
    name = "sql_agent"

    def run(self, state: OrchestrationState, runtime) -> AgentEnvelope:
        state.generated_sql = "SELECT 1 AS sample_value"
        return AgentEnvelope(
            status=AgentStatus.success,
            agent_name="sql_agent",
            summary="SQL 완료",
            artifact_refs=[FakeArtifactRef("artifact_sql_result")],
            next_handoff="validation_agent",
        )


def test_subagent_adapter_calls_existing_agent_contract_and_compacts_result() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    adapter = SubAgentAdapter(backend_adapter=FakeAdapter(), agents={"sql_agent": StubAgent()})

    result = adapter.call("sql_agent", state)

    assert result.agent_result.agent == "sql_agent"
    assert result.agent_result.status == "success"
    assert result.agent_result.artifact_ids == ["artifact_sql_result"]
    assert result.state_updates["generated_sql"] == "SELECT 1 AS sample_value"
```

- [ ] **Step 2: 테스트 실패 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_tools.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'DATA_Analyst_Assistant_Agent.supervisor.tools'
```

- [ ] **Step 3: `supervisor/tools.py` 구현**

Create `supervisor/tools.py`.

```python
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.agents.analysis.agent import AnalysisAgent
from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.agents.eda.agent import EDAAgent
from DATA_Analyst_Assistant_Agent.agents.report.agent import ReportAgent
from DATA_Analyst_Assistant_Agent.agents.sql.agent import SQLAgent
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import AgentEnvelope, AgentStatus
from DATA_Analyst_Assistant_Agent.supervisor.state import (
    AgentCompactResult,
    AgentName,
    ArtifactSummary,
    SupervisorState,
    to_orchestration_state,
)


class RunnableAgent(Protocol):
    name: str

    def run(self, state, runtime: AgentRuntime) -> AgentEnvelope:
        ...


class AgentToolResult(BaseModel):
    agent_result: AgentCompactResult
    state_updates: dict[str, Any] = Field(default_factory=dict)


def default_agents() -> dict[AgentName, RunnableAgent]:
    return {
        "sql_agent": SQLAgent(),
        "eda_agent": EDAAgent(),
        "analysis_agent": AnalysisAgent(),
        "report_agent": ReportAgent(),
    }


class SubAgentAdapter:
    def __init__(
        self,
        *,
        backend_adapter: BackendAdapter,
        agents: dict[str, RunnableAgent] | None = None,
    ) -> None:
        self.backend_adapter = backend_adapter
        self.runtime = AgentRuntime(adapter=backend_adapter)
        self.agents = agents or default_agents()

    def call(self, agent_name: AgentName, state: SupervisorState) -> AgentToolResult:
        if agent_name not in self.agents:
            raise ValueError(f"Unknown supervisor sub-agent: {agent_name}")

        orchestration_state = to_orchestration_state(state)
        envelope = self.agents[agent_name].run(orchestration_state, self.runtime)
        compact = self._compact_envelope(envelope)
        return AgentToolResult(
            agent_result=compact,
            state_updates={
                "generated_sql": orchestration_state.generated_sql,
                "error_state": orchestration_state.error_state,
            },
        )

    def _compact_envelope(self, envelope: AgentEnvelope) -> AgentCompactResult:
        artifact_ids = envelope.artifact_ids()
        return AgentCompactResult(
            agent=envelope.agent_name,
            status=self._status_value(envelope.status),
            summary=envelope.summary,
            artifact_ids=artifact_ids,
            artifacts=[self._artifact_summary(artifact_id) for artifact_id in artifact_ids],
            validation_errors=[
                check.detail
                for check in envelope.validation.local_checks
                if check.severity == "error" and not check.passed
            ],
            validation_warnings=[
                check.detail
                for check in envelope.validation.local_checks
                if check.severity == "warning" and not check.passed
            ],
            retryable=envelope.retry_hint.retryable,
            error="" if envelope.status != AgentStatus.failed else envelope.summary,
        )

    def _artifact_summary(self, artifact_id: str) -> ArtifactSummary:
        try:
            artifact = self.backend_adapter.get_artifact(artifact_id)
        except Exception:
            return ArtifactSummary(artifact_id=artifact_id)
        metadata = getattr(artifact, "metadata", None) or {}
        preview = getattr(artifact, "preview", None) or {}
        summary = ""
        if isinstance(preview, dict):
            summary = ", ".join(f"{key}={value}" for key, value in list(preview.items())[:3])
        return ArtifactSummary(
            artifact_id=artifact_id,
            type=str(getattr(artifact, "type", "")),
            kind=str(metadata.get("kind", "")),
            summary=summary,
            uri=getattr(artifact, "uri", None),
        )

    @staticmethod
    def _status_value(status: AgentStatus | str) -> str:
        return status.value if isinstance(status, AgentStatus) else str(status)
```

- [ ] **Step 4: adapter 테스트 통과 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_tools.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add supervisor/tools.py tests/supervisor/test_tools.py
git commit -m "feat: add supervisor subagent adapter"
```

### Task 5: Guardrail과 validation 구현

**Files:**
- Create: `supervisor/validation.py`
- Test: `tests/supervisor/test_validation.py`

- [ ] **Step 1: 실패하는 validation 테스트 작성**

Create `tests/supervisor/test_validation.py`.

```python
from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult, empty_supervisor_state, merge_agent_result
from DATA_Analyst_Assistant_Agent.supervisor.validation import (
    guard_agent_preconditions,
    validate_subagent_result,
)


def test_guard_blocks_eda_without_sql_artifact() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    decision = guard_agent_preconditions("eda_agent", state)

    assert decision.allowed is False
    assert decision.next_action == "call_sql_agent"


def test_guard_allows_report_after_evidence_artifact() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    state = merge_agent_result(
        state,
        AgentCompactResult(
            agent="analysis_agent",
            status="success",
            summary="분석 완료",
            artifact_ids=["artifact_analysis"],
        ),
    )

    decision = guard_agent_preconditions("report_agent", state)

    assert decision.allowed is True
    assert decision.next_action == "call_report_agent"


def test_validate_failed_retryable_result_routes_to_same_agent() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    result = AgentCompactResult(
        agent="sql_agent",
        status="failed",
        summary="SQL 실패",
        retryable=True,
        error="SQL validation failed",
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is False
    assert decision.next_action == "call_sql_agent"
```

- [ ] **Step 2: 테스트 실패 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_validation.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'DATA_Analyst_Assistant_Agent.supervisor.validation'
```

- [ ] **Step 3: `supervisor/validation.py` 구현**

Create `supervisor/validation.py`.

```python
from __future__ import annotations

from pydantic import BaseModel

from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult, AgentName, NextAction, SupervisorState


class GuardDecision(BaseModel):
    allowed: bool
    next_action: NextAction
    reason: str = ""


class ResultValidationDecision(BaseModel):
    valid: bool
    next_action: NextAction
    reason: str = ""


def guard_agent_preconditions(agent: AgentName, state: SupervisorState) -> GuardDecision:
    completed = set(state.get("completed_agents", []))
    artifacts = state.get("artifacts", {})

    if agent == "sql_agent":
        return GuardDecision(allowed=True, next_action="call_sql_agent")

    if agent == "eda_agent" and not artifacts.get("sql_agent"):
        return GuardDecision(
            allowed=False,
            next_action="call_sql_agent",
            reason="EDA Agent는 SQL 결과 artifact가 필요합니다.",
        )

    if agent == "analysis_agent" and not (artifacts.get("sql_agent") or artifacts.get("eda_agent")):
        return GuardDecision(
            allowed=False,
            next_action="call_sql_agent",
            reason="Analysis Agent는 SQL 결과 또는 EDA profile이 필요합니다.",
        )

    if agent == "report_agent" and not (completed & {"sql_agent", "eda_agent", "analysis_agent"}):
        return GuardDecision(
            allowed=False,
            next_action="call_sql_agent",
            reason="Report Agent는 하나 이상의 evidence artifact가 필요합니다.",
        )

    return GuardDecision(allowed=True, next_action=f"call_{agent}")


def validate_subagent_result(state: SupervisorState, result: AgentCompactResult) -> ResultValidationDecision:
    if result.status in {"success", "warning", "approval_required"} and not result.validation_errors:
        if result.agent == "report_agent":
            return ResultValidationDecision(valid=True, next_action="finalize", reason="최종 리포트가 생성되었습니다.")
        return ResultValidationDecision(valid=True, next_action="create_plan", reason="다음 행동 결정을 계속합니다.")

    retry_count = state.get("retry_counts", {}).get(result.agent, 0)
    max_retry = int(state.get("max_retry_per_agent", 1))
    if result.retryable and retry_count < max_retry:
        return ResultValidationDecision(
            valid=False,
            next_action=f"call_{result.agent}",
            reason=f"{result.agent} 결과가 재시도 가능 상태입니다.",
        )

    return ResultValidationDecision(
        valid=False,
        next_action="fail",
        reason=result.error or f"{result.agent} 결과 검증에 실패했습니다.",
    )
```

- [ ] **Step 4: validation 테스트 통과 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_validation.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add supervisor/validation.py tests/supervisor/test_validation.py
git commit -m "feat: add supervisor validation guardrails"
```

### Task 6: Step summary와 prompt/decision 구현

**Files:**
- Create: `supervisor/summarizer.py`
- Create: `supervisor/prompts.py`
- Create: `supervisor/decision.py`
- Test: `tests/supervisor/test_decision.py`

- [ ] **Step 1: 실패하는 decision 테스트 작성**

Create `tests/supervisor/test_decision.py`.

```python
from __future__ import annotations

from dataclasses import dataclass

from DATA_Analyst_Assistant_Agent.supervisor.decision import decide_next_action, parse_decision_json
from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult, empty_supervisor_state, merge_agent_result
from DATA_Analyst_Assistant_Agent.supervisor.summarizer import summarize_agent_step


@dataclass
class FakeMessage:
    content: str


class FakeModel:
    def invoke(self, messages):
        return FakeMessage('{"next_action":"call_eda_agent","reason":"SQL 결과를 탐색합니다."}')


def test_parse_decision_json_extracts_next_action() -> None:
    decision = parse_decision_json('{"next_action":"call_analysis_agent","reason":"EDA 완료"}')

    assert decision.next_action == "call_analysis_agent"
    assert decision.reason == "EDA 완료"


def test_decide_next_action_uses_model_json_when_available() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    decision = decide_next_action(state, model=FakeModel())

    assert decision.next_action == "call_eda_agent"


def test_decide_next_action_fallback_runs_sql_only_without_evidence() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    decision = decide_next_action(state, model=None)

    assert decision.next_action == "call_sql_agent"


def test_decide_next_action_fallback_fails_after_evidence_exists() -> None:
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )
    state = merge_agent_result(
        state,
        AgentCompactResult(
            agent="sql_agent",
            status="success",
            summary="SQL 완료",
            artifact_ids=["artifact_sql_result"],
        ),
    )

    decision = decide_next_action(state, model=None)

    assert decision.next_action == "fail"
    assert "fallback" in decision.reason


def test_summarize_agent_step_is_compact() -> None:
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 완료",
        artifact_ids=["artifact_sql_result"],
    )
    summary = summarize_agent_step("execute_subagent", result, next_action="call_eda_agent")

    assert summary.step == "execute_subagent"
    assert summary.agent == "sql_agent"
    assert summary.artifact_ids == ["artifact_sql_result"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_decision.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'DATA_Analyst_Assistant_Agent.supervisor.decision'
```

- [ ] **Step 3: `supervisor/summarizer.py` 구현**

Create `supervisor/summarizer.py`.

```python
from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult, StepSummary


def summarize_agent_step(step: str, result: AgentCompactResult, *, next_action: str) -> StepSummary:
    return StepSummary(
        step=step,
        agent=result.agent,
        action=f"call_{result.agent}",
        summary=result.summary[:1000],
        artifact_ids=list(result.artifact_ids),
        next_action=next_action,
    )
```

- [ ] **Step 4: `supervisor/prompts.py` 구현**

Create `supervisor/prompts.py`.

```python
from __future__ import annotations


CLARIFY_QUERY_PROMPT = """사용자의 데이터 분석 요청이 실행 가능한지 판단하세요.
필요한 데이터 소스, 지표, 기간, 차원이 명확하지 않으면 한국어로 하나의 clarification 질문을 만드세요.
JSON 형식으로만 답하세요: {"needs_clarification": true|false, "question": "..."}"""


CREATE_ANALYSIS_PLAN_PROMPT = """사용자의 데이터 분석 요청을 기존 SQL, EDA, Analysis, Report Sub-Agent로 해결할 계획을 세우세요.
기존 Sub-Agent 내부 구현은 변경할 수 없습니다.
JSON 형식으로만 답하세요: {"goal":"...", "steps":[{"agent":"sql_agent","purpose":"..."}], "route_kind":"comprehensive"}"""


DECIDE_NEXT_ACTION_PROMPT = """SupervisorState 요약을 보고 다음 행동 하나를 고르세요.
허용 값: clarify, call_sql_agent, call_eda_agent, call_analysis_agent, call_report_agent, finalize, fail.
SQL 결과가 없으면 EDA보다 SQL을 먼저 호출합니다.
최종 리포트가 없고 evidence가 충분하면 report_agent를 호출합니다.
JSON 형식으로만 답하세요: {"next_action":"...", "reason":"..."}"""
```

- [ ] **Step 5: `supervisor/decision.py` 구현**

Create `supervisor/decision.py`.

```python
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from DATA_Analyst_Assistant_Agent.supervisor.prompts import DECIDE_NEXT_ACTION_PROMPT
from DATA_Analyst_Assistant_Agent.supervisor.state import NextAction, SupervisorState


class SupervisorDecision(BaseModel):
    next_action: NextAction
    reason: str = ""


def parse_decision_json(text: str) -> SupervisorDecision:
    payload_text = text.strip()
    if not payload_text.startswith("{"):
        match = re.search(r"\{.*\}", payload_text, flags=re.DOTALL)
        payload_text = match.group(0) if match else "{}"
    payload = json.loads(payload_text)
    return SupervisorDecision(
        next_action=payload["next_action"],
        reason=str(payload.get("reason", "")),
    )


def decide_next_action(state: SupervisorState, *, model: Any | None = None) -> SupervisorDecision:
    if model is not None:
        try:
            message = model.invoke(
                [
                    ("system", DECIDE_NEXT_ACTION_PROMPT),
                    ("user", json.dumps(_decision_snapshot(state), ensure_ascii=False, default=str)),
                ]
            )
            return parse_decision_json(str(getattr(message, "content", message)))
        except Exception:
            pass

    return _fallback_decision(state)


def _decision_snapshot(state: SupervisorState) -> dict[str, Any]:
    return {
        "latest_user_query": state.get("latest_user_query", ""),
        "analysis_plan": state.get("analysis_plan", {}),
        "completed_agents": state.get("completed_agents", []),
        "failed_agents": state.get("failed_agents", []),
        "artifacts": state.get("artifacts", {}),
        "validation_results": state.get("validation_results", [])[-3:],
        "step_summaries": state.get("step_summaries", [])[-5:],
        "terminal_state": state.get("terminal_state", "running"),
    }


def _fallback_decision(state: SupervisorState) -> SupervisorDecision:
    completed = set(state.get("completed_agents", []))
    artifacts = state.get("artifacts", {})
    if not completed and not artifacts.get("sql_agent"):
        return SupervisorDecision(
            next_action="call_sql_agent",
            reason="fallback: 첫 evidence 생성을 위한 최소 SQL 실행만 허용합니다.",
        )
    if "report_agent" in completed:
        return SupervisorDecision(next_action="finalize", reason="final report artifact가 이미 생성되었습니다.")
    return SupervisorDecision(
        next_action="fail",
        reason="fallback: 다음 행동을 근거 있게 결정할 수 없어 안전 종료합니다.",
    )
```

- [ ] **Step 6: decision 테스트 통과 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_decision.py -v
```

Expected:

```text
5 passed
```

- [ ] **Step 7: Commit**

```bash
git add supervisor/summarizer.py supervisor/prompts.py supervisor/decision.py tests/supervisor/test_decision.py
git commit -m "feat: add supervisor decision helpers"
```

### Task 7: Supervisor LangGraph 구현

**Files:**
- Create: `supervisor/graph.py`
- Test: `tests/supervisor/test_graph.py`

- [ ] **Step 1: 실패하는 graph 테스트 작성**

Create `tests/supervisor/test_graph.py`.

```python
from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.graph import build_graph
from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult, empty_supervisor_state
from DATA_Analyst_Assistant_Agent.supervisor.tools import AgentToolResult


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class SequencedDecisionModel:
    def __init__(self) -> None:
        self.actions = iter(
            [
                "call_sql_agent",
                "call_eda_agent",
                "call_analysis_agent",
                "call_report_agent",
                "finalize",
            ]
        )

    def invoke(self, messages):
        action = next(self.actions)
        return FakeMessage(f'{{"next_action":"{action}","reason":"테스트 모델 결정"}}')


class FakeSubAgentAdapter:
    def call(self, agent_name, state):
        return AgentToolResult(
            agent_result=AgentCompactResult(
                agent=agent_name,
                status="success",
                summary=f"{agent_name} 완료",
                artifact_ids=[f"artifact_{agent_name}"],
            ),
            state_updates={"generated_sql": "SELECT 1 AS sample_value" if agent_name == "sql_agent" else ""},
        )


def test_supervisor_graph_runs_all_subagents_and_finalizes() -> None:
    graph = build_graph(subagent_adapter=FakeSubAgentAdapter(), model=SequencedDecisionModel())
    state = empty_supervisor_state(
        thread_id="thread_sales_001",
        run_id="run_001",
        user_query="월별 매출 추이를 분석해줘",
        datasource_id=None,
    )

    result = graph.invoke(state, {"configurable": {"thread_id": "thread_sales_001"}})

    assert result["terminal_state"] == "completed"
    assert result["completed_agents"] == ["sql_agent", "eda_agent", "analysis_agent", "report_agent"]
    assert result["final_answer"] == "최종 리포트 생성이 완료되었습니다."
```

- [ ] **Step 2: 테스트 실패 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_graph.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'DATA_Analyst_Assistant_Agent.supervisor.graph'
```

- [ ] **Step 3: `supervisor/graph.py` 구현**

Create `supervisor/graph.py`.

```python
from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from DATA_Analyst_Assistant_Agent.supervisor.decision import decide_next_action
from DATA_Analyst_Assistant_Agent.supervisor.state import AgentName, SupervisorState, merge_agent_result
from DATA_Analyst_Assistant_Agent.supervisor.summarizer import summarize_agent_step
from DATA_Analyst_Assistant_Agent.supervisor.tools import SubAgentAdapter
from DATA_Analyst_Assistant_Agent.supervisor.validation import guard_agent_preconditions, validate_subagent_result


ACTION_TO_AGENT: dict[str, AgentName] = {
    "call_sql_agent": "sql_agent",
    "call_eda_agent": "eda_agent",
    "call_analysis_agent": "analysis_agent",
    "call_report_agent": "report_agent",
}


def clarify_query_node(state: SupervisorState) -> dict[str, Any]:
    query = state.get("latest_user_query", "").strip()
    if len(query) < 2:
        return {
            "needs_clarification": True,
            "clarification_question": "분석하려는 질문을 조금 더 구체적으로 입력해 주세요.",
            "terminal_state": "needs_clarification",
            "current_step": "clarify_query",
        }
    return {
        "needs_clarification": False,
        "clarified_query": query,
        "current_step": "clarify_query",
    }


def create_analysis_plan_node(state: SupervisorState) -> dict[str, Any]:
    if state.get("analysis_plan"):
        return {"current_step": "create_analysis_plan"}
    return {
        "analysis_plan": {
            "goal": state.get("clarified_query") or state.get("latest_user_query", ""),
            "candidate_agents": [
                "sql_agent",
                "eda_agent",
                "analysis_agent",
                "report_agent",
            ],
            "constraints": [
                "Sub-Agent 내부 코드는 변경하지 않는다.",
                "다음 행동은 state, validation, artifact evidence를 근거로 결정한다.",
                "근거가 부족하면 성공으로 위장하지 않고 clarification 또는 failure 상태로 종료한다.",
            ],
            "route_kind": "comprehensive",
            "planner_mode": "state_based",
        },
        "current_step": "create_analysis_plan",
    }


def make_decide_next_action_node(model: Any | None):
    def decide_next_action_node(state: SupervisorState) -> dict[str, Any]:
        if state.get("terminal_state") == "needs_clarification":
            return {"next_action": "clarify", "current_step": "decide_next_action"}
        decision = decide_next_action(state, model=model)
        return {
            "next_action": decision.next_action,
            "current_step": "decide_next_action",
            "validation_results": [
                *state.get("validation_results", []),
                {"step": "decide_next_action", "decision": decision.model_dump(mode="json")},
            ],
        }

    return decide_next_action_node


def make_execute_subagent_node(subagent_adapter: SubAgentAdapter):
    def execute_subagent_node(state: SupervisorState) -> dict[str, Any]:
        next_action = state.get("next_action", "fail")
        agent = ACTION_TO_AGENT.get(next_action)
        if not agent:
            return {
                "terminal_state": "failed_terminal",
                "error_state": {"message": f"실행할 수 없는 next_action입니다: {next_action}"},
                "current_step": "execute_subagent",
            }

        guard = guard_agent_preconditions(agent, state)
        if not guard.allowed:
            return {
                "next_action": guard.next_action,
                "validation_results": [
                    *state.get("validation_results", []),
                    {"step": "guard", "agent": agent, "reason": guard.reason},
                ],
                "current_step": "execute_subagent",
            }

        tool_result = subagent_adapter.call(agent, state)
        merged = merge_agent_result(state, tool_result.agent_result)
        updates: dict[str, Any] = {
            "agent_results": merged["agent_results"],
            "artifacts": merged["artifacts"],
            "completed_agents": merged["completed_agents"],
            "failed_agents": merged["failed_agents"],
            "error_state": tool_result.state_updates.get("error_state") or merged.get("error_state", {}),
            "generated_sql": tool_result.state_updates.get("generated_sql") or state.get("generated_sql", ""),
            "current_step": "execute_subagent",
        }
        updates["_last_agent_result"] = tool_result.agent_result.model_dump(mode="json")
        return updates

    return execute_subagent_node


def validate_subagent_result_node(state: SupervisorState) -> dict[str, Any]:
    result_payload = state.get("_last_agent_result") or {}
    if not result_payload:
        return {"current_step": "validate_subagent_result"}

    from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult

    result = AgentCompactResult.model_validate(result_payload)
    decision = validate_subagent_result(state, result)
    retry_counts = dict(state.get("retry_counts", {}))
    if not decision.valid and decision.next_action == f"call_{result.agent}":
        retry_counts[result.agent] = retry_counts.get(result.agent, 0) + 1

    return {
        "next_action": decision.next_action,
        "retry_counts": retry_counts,
        "validation_results": [
            *state.get("validation_results", []),
            {
                "step": "validate_subagent_result",
                "agent": result.agent,
                "decision": decision.model_dump(mode="json"),
            },
        ],
        "current_step": "validate_subagent_result",
    }


def summarize_step_node(state: SupervisorState) -> dict[str, Any]:
    result_payload = state.get("_last_agent_result") or {}
    if not result_payload:
        return {"current_step": "summarize_step"}

    from DATA_Analyst_Assistant_Agent.supervisor.state import AgentCompactResult

    result = AgentCompactResult.model_validate(result_payload)
    summary = summarize_agent_step(
        "execute_subagent",
        result,
        next_action=state.get("next_action", ""),
    )
    return {
        "step_summaries": [*state.get("step_summaries", []), summary.model_dump(mode="json")],
        "current_step": "summarize_step",
    }


def finalize_node(state: SupervisorState) -> dict[str, Any]:
    if state.get("next_action") == "fail":
        return {
            "terminal_state": "failed_terminal",
            "final_answer": json.dumps(state.get("error_state", {}), ensure_ascii=False),
            "current_step": "finalize",
        }
    if state.get("terminal_state") == "needs_clarification":
        return {
            "final_answer": state.get("clarification_question", ""),
            "current_step": "finalize",
        }
    return {
        "terminal_state": "completed",
        "final_answer": "최종 리포트 생성이 완료되었습니다.",
        "current_step": "finalize",
    }


def route_after_clarify(state: SupervisorState) -> str:
    return "finalize" if state.get("terminal_state") == "needs_clarification" else "create_analysis_plan"


def route_after_decision(state: SupervisorState) -> str:
    action = state.get("next_action")
    if action in {"clarify", "finalize", "fail"}:
        return "finalize"
    return "execute_subagent"


def route_after_execute(state: SupervisorState) -> str:
    if state.get("next_action") in ACTION_TO_AGENT and not state.get("_last_agent_result"):
        return "decide_next_action"
    return "validate_subagent_result"


def route_after_summary(state: SupervisorState) -> str:
    return "finalize" if state.get("next_action") in {"finalize", "fail"} else "decide_next_action"


def build_graph(
    *,
    subagent_adapter: SubAgentAdapter,
    model: Any | None = None,
    checkpointer: Any | None = None,
):
    builder = StateGraph(SupervisorState)
    builder.add_node("clarify_query", clarify_query_node)
    builder.add_node("create_analysis_plan", create_analysis_plan_node)
    builder.add_node("decide_next_action", make_decide_next_action_node(model))
    builder.add_node("execute_subagent", make_execute_subagent_node(subagent_adapter))
    builder.add_node("validate_subagent_result", validate_subagent_result_node)
    builder.add_node("summarize_step", summarize_step_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "clarify_query")
    builder.add_conditional_edges(
        "clarify_query",
        route_after_clarify,
        {"create_analysis_plan": "create_analysis_plan", "finalize": "finalize"},
    )
    builder.add_edge("create_analysis_plan", "decide_next_action")
    builder.add_conditional_edges(
        "decide_next_action",
        route_after_decision,
        {"execute_subagent": "execute_subagent", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "execute_subagent",
        route_after_execute,
        {"validate_subagent_result": "validate_subagent_result", "decide_next_action": "decide_next_action"},
    )
    builder.add_edge("validate_subagent_result", "summarize_step")
    builder.add_conditional_edges(
        "summarize_step",
        route_after_summary,
        {"decide_next_action": "decide_next_action", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: graph 테스트 통과 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_graph.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: `_last_agent_result` 저장 범위 점검**

`_last_agent_result`는 checkpoint에 들어갈 수 있는 작은 dict다. 대용량 row, DataFrame, LLM raw response, runtime 객체가 포함되지 않는지 확인한다.

Run:

```bash
python3 -m pytest tests/supervisor/test_state.py tests/supervisor/test_graph.py -v
```

Expected:

```text
4 passed
```

- [ ] **Step 6: Commit**

```bash
git add supervisor/graph.py tests/supervisor/test_graph.py
git commit -m "feat: add langgraph supervisor workflow"
```

### Task 8: Supervisor 실행 진입점 구현

**Files:**
- Create: `supervisor/agent.py`
- Create: `supervisor/__init__.py`
- Test: `tests/supervisor/test_agent.py`

- [ ] **Step 1: 실패하는 agent 테스트 작성**

Create `tests/supervisor/test_agent.py`.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from DATA_Analyst_Assistant_Agent.supervisor.agent import SQLAgentSupervisor, SupervisorAgent


@dataclass
class FakeRun:
    run_id: str


class FakeBackendAdapter:
    base_data_dir = ".data_agent"

    def create_run(self, *, thread_id=None, project_id=None, metadata=None):
        return FakeRun(run_id="run_001")

    def update_run_status(self, run_id, status, *, metadata=None, context=None):
        return FakeRun(run_id=run_id)


class FakeGraph:
    def invoke(self, state, config):
        return {
            **state,
            "completed_agents": ["sql_agent", "eda_agent", "analysis_agent", "report_agent"],
            "terminal_state": "completed",
            "final_answer": "최종 리포트 생성이 완료되었습니다.",
        }


def test_supervisor_agent_run_returns_orchestration_state(monkeypatch) -> None:
    agent = SupervisorAgent(FakeBackendAdapter(), checkpoint_path=":memory:")

    monkeypatch.setattr(agent, "_invoke_graph", lambda initial_state, thread_id: FakeGraph().invoke(initial_state, {}))

    state = agent.run("월별 매출 추이를 분석해줘", thread_id="thread_sales_001", datasource_id=None)

    assert state.run_id == "run_001"
    assert state.thread_id == "thread_sales_001"
    assert state.terminal_state.value == "completed"


def test_sql_agent_supervisor_alias_points_to_new_supervisor() -> None:
    assert SQLAgentSupervisor is SupervisorAgent
```

- [ ] **Step 2: 테스트 실패 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_agent.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'DATA_Analyst_Assistant_Agent.supervisor.agent'
```

- [ ] **Step 3: `supervisor/agent.py` 구현**

Create `supervisor/agent.py`.

```python
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from data_agent_backend.models.runs import RunStatus
from langgraph.types import Command

from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.shared.contracts import OrchestrationState, SupervisorTerminalState
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
from DATA_Analyst_Assistant_Agent.supervisor.checkpoint import open_sqlite_checkpointer
from DATA_Analyst_Assistant_Agent.supervisor.graph import build_graph
from DATA_Analyst_Assistant_Agent.supervisor.state import empty_supervisor_state, to_orchestration_state
from DATA_Analyst_Assistant_Agent.supervisor.tools import SubAgentAdapter


class SupervisorAgent:
    def __init__(
        self,
        adapter: BackendAdapter | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        model: Any | None = None,
        use_llm_decision: bool = True,
    ) -> None:
        self.adapter = adapter or BackendAdapter()
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.use_llm_decision = use_llm_decision

    def run(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        datasource_id: str | None = None,
        project_id: str | None = None,
    ) -> OrchestrationState:
        thread_id = thread_id or f"thread_{uuid.uuid4().hex}"
        if datasource_id is None:
            datasource_id = self.adapter.get_default_datasource_id()
        catalog_summary = self.adapter.get_catalog_summary(datasource_id) if datasource_id else None
        run = self.adapter.create_run(
            thread_id=thread_id,
            project_id=project_id,
            metadata={"query": query, "supervisor": "langgraph"},
        )
        initial_state = empty_supervisor_state(
            thread_id=thread_id,
            run_id=run.run_id,
            user_query=query,
            datasource_id=datasource_id,
            project_id=project_id,
            catalog_summary=catalog_summary,
        )

        try:
            output = self._invoke_graph(initial_state, thread_id)
            orchestration_state = to_orchestration_state(output)
            status = RunStatus.completed if orchestration_state.terminal_state == SupervisorTerminalState.completed else RunStatus.failed
            self.adapter.update_run_status(run.run_id, status, metadata={"terminal_state": output.get("terminal_state")})
            return orchestration_state
        except Exception as exc:
            self.adapter.update_run_status(run.run_id, RunStatus.failed, metadata={"error": str(exc)})
            raise

    def resume(self, *, thread_id: str, resume_payload: dict[str, Any]) -> dict[str, Any]:
        with open_sqlite_checkpointer(self.checkpoint_path) as checkpointer:
            graph = build_graph(
                subagent_adapter=SubAgentAdapter(backend_adapter=self.adapter),
                model=self._decision_model(),
                checkpointer=checkpointer,
            )
            return graph.invoke(Command(resume=resume_payload), {"configurable": {"thread_id": thread_id}})

    def _invoke_graph(self, initial_state: dict[str, Any], thread_id: str) -> dict[str, Any]:
        with open_sqlite_checkpointer(self.checkpoint_path) as checkpointer:
            graph = build_graph(
                subagent_adapter=SubAgentAdapter(backend_adapter=self.adapter),
                model=self._decision_model(),
                checkpointer=checkpointer,
            )
            return graph.invoke(initial_state, {"configurable": {"thread_id": thread_id}})

    def _decision_model(self) -> Any | None:
        if self.model is not None:
            return self.model
        if not self.use_llm_decision:
            return None
        return get_chat_model(temperature=0)


SQLAgentSupervisor = SupervisorAgent
```

- [ ] **Step 4: `supervisor/__init__.py` 구현**

Create `supervisor/__init__.py`.

```python
from __future__ import annotations

from DATA_Analyst_Assistant_Agent.supervisor.agent import SQLAgentSupervisor, SupervisorAgent
from DATA_Analyst_Assistant_Agent.supervisor.graph import build_graph

__all__ = ["SQLAgentSupervisor", "SupervisorAgent", "build_graph"]
```

- [ ] **Step 5: agent 테스트 통과 확인**

Run:

```bash
python3 -m pytest tests/supervisor/test_agent.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

```bash
git add supervisor/agent.py supervisor/__init__.py tests/supervisor/test_agent.py
git commit -m "feat: add supervisor agent entrypoint"
```

### Task 9: 공개 API와 CLI 전환

**Files:**
- Modify: `__init__.py`
- Modify: `agents/__init__.py`
- Modify: `run.py`
- Test: `tests/test_public_api.py`

- [ ] **Step 1: 실패하는 public API 테스트 작성**

Create `tests/test_public_api.py`.

```python
from __future__ import annotations


def test_package_exports_new_supervisor_aliases() -> None:
    import DATA_Analyst_Assistant_Agent as daaa

    assert daaa.SupervisorAgent is daaa.SQLAgentSupervisor
    assert callable(daaa.build_graph)


def test_agents_package_no_longer_exports_deleted_agents() -> None:
    import DATA_Analyst_Assistant_Agent.agents as agents

    assert "CentralValidationAgent" not in agents.__all__
    assert "VisualizationAgent" not in agents.__all__
```

- [ ] **Step 2: 테스트 실패 확인**

Run:

```bash
python3 -m pytest tests/test_public_api.py -v
```

Expected:

```text
AssertionError
```

현재 root `__init__.py`와 `agents/__init__.py`에는 삭제된 validation/visualization export가 남아 있다.

- [ ] **Step 3: root `__init__.py` export 수정**

Replace the root `__all__` and `_EXPORT_MAP` with this content while keeping the existing `__getattr__` implementation.

```python
__all__ = [
    "AgentEnvelope",
    "AgentStatus",
    "AnalysisAgent",
    "BackendAdapter",
    "EDAAgent",
    "OrchestrationState",
    "ReportAgent",
    "SQLAgent",
    "SQLAgentSupervisor",
    "SupervisorAgent",
    "SupervisorTerminalState",
    "build_graph",
]

_EXPORT_MAP = {
    "AgentEnvelope": ("DATA_Analyst_Assistant_Agent.shared.contracts", "AgentEnvelope"),
    "AgentStatus": ("DATA_Analyst_Assistant_Agent.shared.contracts", "AgentStatus"),
    "AnalysisAgent": ("DATA_Analyst_Assistant_Agent.agents", "AnalysisAgent"),
    "BackendAdapter": ("DATA_Analyst_Assistant_Agent.shared.backend_adapter", "BackendAdapter"),
    "EDAAgent": ("DATA_Analyst_Assistant_Agent.agents", "EDAAgent"),
    "OrchestrationState": ("DATA_Analyst_Assistant_Agent.shared.contracts", "OrchestrationState"),
    "ReportAgent": ("DATA_Analyst_Assistant_Agent.agents", "ReportAgent"),
    "SQLAgent": ("DATA_Analyst_Assistant_Agent.agents", "SQLAgent"),
    "SQLAgentSupervisor": ("DATA_Analyst_Assistant_Agent.supervisor", "SQLAgentSupervisor"),
    "SupervisorAgent": ("DATA_Analyst_Assistant_Agent.supervisor", "SupervisorAgent"),
    "SupervisorTerminalState": ("DATA_Analyst_Assistant_Agent.shared.contracts", "SupervisorTerminalState"),
    "build_graph": ("DATA_Analyst_Assistant_Agent.supervisor.graph", "build_graph"),
}
```

- [ ] **Step 4: `agents/__init__.py` export 수정**

Replace the `__all__` and `_EXPORT_MAP` with this content while keeping the existing `__getattr__` implementation.

```python
__all__ = [
    "AgentRuntime",
    "AnalysisAgent",
    "EDAAgent",
    "ReportAgent",
    "SQLAgent",
]

_EXPORT_MAP = {
    "AgentRuntime": ("DATA_Analyst_Assistant_Agent.agents.common", "AgentRuntime"),
    "AnalysisAgent": ("DATA_Analyst_Assistant_Agent.agents.analysis.agent", "AnalysisAgent"),
    "EDAAgent": ("DATA_Analyst_Assistant_Agent.agents.eda.agent", "EDAAgent"),
    "ReportAgent": ("DATA_Analyst_Assistant_Agent.agents.report.agent", "ReportAgent"),
    "SQLAgent": ("DATA_Analyst_Assistant_Agent.agents.sql.agent", "SQLAgent"),
}
```

- [ ] **Step 5: `run.py` import와 설명 수정**

Change the import:

```python
from DATA_Analyst_Assistant_Agent import BackendAdapter, SupervisorAgent
```

Change the parser description:

```python
description="Run DATA_Analyst_Assistant_Agent through the LangGraph Supervisor.",
```

Change supervisor construction:

```python
supervisor = SupervisorAgent(adapter)
```

- [ ] **Step 6: public API 테스트 통과 확인**

Run:

```bash
python3 -m pytest tests/test_public_api.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 7: Commit**

```bash
git add __init__.py agents/__init__.py run.py tests/test_public_api.py
git commit -m "refactor: expose langgraph supervisor api"
```

### Task 10: 통합 검증과 회귀 테스트

**Files:**
- No new production files
- Verify all changed files

- [ ] **Step 1: Supervisor 단위 테스트 전체 실행**

Run:

```bash
python3 -m pytest tests/supervisor -v
```

Expected:

```text
all supervisor tests passed
```

- [ ] **Step 2: 전체 테스트 실행**

Run:

```bash
python3 -m pytest -v
```

Expected:

```text
all tests passed
```

- [ ] **Step 3: import smoke test 실행**

Run from the parent directory where the package name is importable:

```bash
cd ..
python3 - <<'PY'
import DATA_Analyst_Assistant_Agent as daaa

print(daaa.SupervisorAgent.__name__)
print(daaa.SQLAgentSupervisor.__name__)
print(daaa.build_graph)
PY
```

Expected:

```text
SupervisorAgent
SupervisorAgent
<function build_graph
```

- [ ] **Step 4: CLI help smoke test 실행**

Run:

```bash
cd ..
python3 -m DATA_Analyst_Assistant_Agent.run --help
```

Expected:

```text
Run DATA_Analyst_Assistant_Agent through the LangGraph Supervisor.
```

- [ ] **Step 5: 실제 실행 smoke test**

`.env`와 DB 접속 정보가 있는 환경에서만 실행한다.

Run:

```bash
cd ..
python3 -m DATA_Analyst_Assistant_Agent.run "월별 매출 추이를 분석해줘" --json --no-open --output-dir daaa_outputs/supervisor_smoke
```

Expected:

```text
{
  "terminal_state": "completed",
  ...
}
```

그리고 다음 파일이 생성되어야 한다.

```text
DATA_Analyst_Assistant_Agent/.data_agent/checkpoints/supervisor.sqlite
daaa_outputs/supervisor_smoke/run_summary.json
daaa_outputs/supervisor_smoke/final_report.md
```

- [ ] **Step 6: checkpoint DB에 checkpoint가 저장됐는지 확인**

Run:

```bash
sqlite3 DATA_Analyst_Assistant_Agent/.data_agent/checkpoints/supervisor.sqlite \
  'select thread_id, count(*) from checkpoints group by thread_id;'
```

Expected:

```text
daaa-manual-run|...
```

- [ ] **Step 7: Commit**

```bash
git status --short
git commit --allow-empty -m "test: verify langgraph supervisor integration"
```

## Self-Review

1. **Spec coverage:** `supervisor/` 폴더, `SupervisorState`, node 입출력, adapter tool 계약, `run.py`/공개 API 전환, 기존 `SQLAgentSupervisor` 대체, checkpoint 단기 메모리, 테스트 전략을 모두 포함했다.
2. **Placeholder scan:** 미완성 표식이나 구현 공백을 남기는 표현은 사용하지 않았다.
3. **Type consistency:** `AgentName`, `NextAction`, `SupervisorState`, `AgentCompactResult`, `AgentToolResult` 이름은 모든 task에서 동일하게 사용했다.
4. **Scope check:** 장기 메모리, checkpoint cleanup, time travel/fork, deleted validation/visualization agent 재구현은 범위에서 제외했다.

## 참고 자료

- LangGraph persistence 공식 문서: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph SQLite checkpointer reference: https://reference.langchain.com/python/langgraph.checkpoint.sqlite
- LangGraph SQLite saver source: https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/__init__.py

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-03-supervisor-langgraph.md`. Two execution options:

1. **Subagent-Driven (recommended)** - 각 task마다 fresh subagent를 dispatch하고 task 사이에 리뷰한다.
2. **Inline Execution** - 이 세션에서 `superpowers:executing-plans`로 checkpoint를 두고 순차 실행한다.

Which approach?
