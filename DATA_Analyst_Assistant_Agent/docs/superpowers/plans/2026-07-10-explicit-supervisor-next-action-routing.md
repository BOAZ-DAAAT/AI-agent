# Supervisor Next Action 명시적 라우팅 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hard Validation 성공 이후의 의미를 `create_plan`이 아닌 `decide_next_action`으로 표현하고, `summarize_step` 이후의 모든 허용 경로와 잘못된 action 처리를 명시적으로 만든다.

**Architecture:** 실제 분석 계획 생성을 뜻하는 `create_plan`은 clarification 경로에 유지하고, 검증 후 Supervisor 재판단을 뜻하는 내부 전이 action으로 `decide_next_action`을 추가한다. Hard Validation이 새 action을 반환하면 Semantic Validation과 Step Summary가 이를 보존하고, `_route_after_summarize()`가 허용된 action을 폐쇄적으로 매핑하며 그 외 값은 명시적 오류로 거부한다.

**Tech Stack:** Python 3.12, LangGraph `StateGraph`, Pydantic, `TypedDict`/`Literal`, pytest

## Global Constraints

- 사용자 응답, 코드 주석, Markdown 문서는 한국어로 작성한다.
- `create_plan`은 clarification 이후 실제 `create_analysis_plan` 실행 의미로 유지하며 전역 치환하거나 삭제하지 않는다.
- `decide_next_action`은 Hard Validation 이후 Supervisor가 다음 행동을 재판단하는 내부 전이 의미로만 사용한다.
- `build_next_action_context().available_next_actions`와 `DECIDE_NEXT_ACTION_PROMPT`에는 `decide_next_action`을 노출하지 않는다. `decide_next_action` 노드가 자기 자신을 실행 action으로 선택하는 루프를 만들지 않는다.
- Semantic Validation은 advisory 역할만 수행하고 state의 `next_action`을 덮어쓰지 않는다.
- Step Summary의 LLM 응답은 요약에 기록하되 state의 실제 라우팅 action을 덮어쓰지 않는다.
- 재시도, 리포트 생성, 승인 대기, clarification, 종료 흐름의 기존 동작을 보존한다.
- 새 의존성을 추가하지 않고 관련 없는 파일을 리팩터링하지 않는다.
- 테스트 명령은 상위 저장소 루트 `AI-agent`에서 `.venv/bin/python`으로 실행한다.

---

## 변경 파일 구조

- `DATA_Analyst_Assistant_Agent/supervisor/state.py`: `NextAction` 계약에 내부 재판단 action을 추가한다.
- `DATA_Analyst_Assistant_Agent/supervisor/validation.py`: 일반 하위 에이전트의 Hard Validation 성공 결과를 `decide_next_action`으로 변경한다.
- `DATA_Analyst_Assistant_Agent/supervisor/graph.py`: Semantic Validation이 재판단 action을 보존하게 하고, `summarize_step` 이후 action을 명시적으로 라우팅하며 알 수 없는 값을 거부한다.
- `DATA_Analyst_Assistant_Agent/supervisor/prompts.py`: 결과 검증과 단계 요약 프롬프트의 action 용어를 실제 의미에 맞춘다.
- `DATA_Analyst_Assistant_Agent/tests/supervisor/test_validation.py`: 새 validation 계약을 단위 테스트한다.
- `DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py`: 명시적 라우팅, 오류 처리, 전체 성공·재시도 흐름을 검증한다.
- `DATA_Analyst_Assistant_Agent/tests/supervisor/test_decision.py`: 다음 행동 결정 LLM에는 내부 재판단 action이 노출되지 않는지 검증한다.

---

### Task 1: Hard Validation 성공 계약을 `decide_next_action`으로 변경

**Files:**
- Modify: `DATA_Analyst_Assistant_Agent/supervisor/state.py:16-27`
- Modify: `DATA_Analyst_Assistant_Agent/supervisor/validation.py:122-139`
- Test: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_validation.py:151-164`

**Interfaces:**
- Consumes: `AgentCompactResult`, `SupervisorState`, `validate_subagent_result(state, result)`
- Produces: `NextAction`이 허용하는 `"decide_next_action"` 값과 일반 Agent 성공 시 `ResultValidationDecision(next_action="decide_next_action")`

- [ ] **Step 1: 일반 Agent 성공 시 새 action을 기대하는 실패 테스트 작성**

`test_validate_success_with_only_validation_warnings_is_valid()`를 다음처럼 수정하고 일반 success 케이스를 추가한다.

```python
def test_validate_success_with_only_validation_warnings_requests_supervisor_redecision() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="eda_agent",
        status="warning",
        summary="EDA 경고 포함 완료",
        validation_warnings=["표본 수가 적습니다"],
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is True
    assert decision.next_action == "decide_next_action"
    assert "다음 행동 재판단" in decision.reason


def test_validate_success_requests_supervisor_redecision() -> None:
    state = _state()
    result = AgentCompactResult(
        agent="sql_agent",
        status="success",
        summary="SQL 실행 완료",
        artifact_ids=["artifact_sql"],
    )

    decision = validate_subagent_result(state, result)

    assert decision.valid is True
    assert decision.next_action == "decide_next_action"
```

- [ ] **Step 2: 대상 테스트를 실행해 기존 구현에서 실패하는지 확인**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
.venv/bin/python -m pytest DATA_Analyst_Assistant_Agent/tests/supervisor/test_validation.py -k "supervisor_redecision" -v
```

Expected: `create_plan`과 `decide_next_action` 불일치로 2개 테스트가 FAIL한다.

- [ ] **Step 3: `NextAction`에 내부 재판단 action 추가**

`supervisor/state.py`의 계약을 다음처럼 변경한다. `create_plan`은 실제 계획 생성 용도로 유지한다.

```python
NextAction = Literal[
    "clarify",
    "create_plan",
    "decide_next_action",
    "call_sql_agent",
    "call_eda_agent",
    "call_analysis_agent",
    "call_report_agent",
    "finalize",
    "fail",
]
```

- [ ] **Step 4: 일반 Agent 성공 validation 결과 변경**

`supervisor/validation.py`의 report가 아닌 성공·warning 분기를 다음처럼 변경한다.

```python
return ResultValidationDecision(
    valid=True,
    next_action="decide_next_action",
    reason=f"{result.agent} 결과가 유효해 Supervisor의 다음 행동 재판단으로 이동합니다.",
)
```

Report Agent의 유효한 결과는 기존처럼 `next_action="finalize"`를 유지한다.

- [ ] **Step 5: validation 단위 테스트 통과 확인**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
.venv/bin/python -m pytest DATA_Analyst_Assistant_Agent/tests/supervisor/test_validation.py -v
```

Expected: 모든 `test_validation.py` 테스트가 PASS한다.

- [ ] **Step 6: 계약 변경 커밋**

```bash
git add DATA_Analyst_Assistant_Agent/supervisor/state.py \
        DATA_Analyst_Assistant_Agent/supervisor/validation.py \
        DATA_Analyst_Assistant_Agent/tests/supervisor/test_validation.py
git commit -m "refactor: validation 이후 재판단 action 명확화"
```

---

### Task 2: `summarize_step` 이후 라우팅을 폐쇄적인 명시적 매핑으로 변경

**Files:**
- Modify: `DATA_Analyst_Assistant_Agent/supervisor/graph.py:858-867`
- Modify: `DATA_Analyst_Assistant_Agent/supervisor/graph.py:453-508`
- Test: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py:1-24`
- Test: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py`의 라우팅 단위 테스트 영역

**Interfaces:**
- Consumes: `_route_after_summarize(state: SupervisorState) -> str`, state의 `terminal_state`와 `next_action`
- Produces: 허용 action의 명시적 노드 이름 또는 잘못된 action에 대한 `ValueError`

- [ ] **Step 1: private router를 테스트에 import하고 허용 action 매핑 테스트 작성**

`tests/supervisor/test_graph.py`에 `pytest`와 `_route_after_summarize` import를 추가한다.

```python
import pytest

from DATA_Analyst_Assistant_Agent.supervisor.graph import (
    _route_after_summarize,
    build_graph,
    make_create_analysis_plan_node,
    make_execute_subagent_node,
    make_semantic_validate_subagent_result_node,
    make_validate_subagent_result_node,
)
```

다음 매핑 테스트를 추가한다.

```python
@pytest.mark.parametrize(
    ("next_action", "expected_node"),
    [
        ("decide_next_action", "decide_next_action"),
        ("call_sql_agent", "execute_subagent"),
        ("call_eda_agent", "execute_subagent"),
        ("call_analysis_agent", "execute_subagent"),
        ("call_report_agent", "generate_report"),
        ("finalize", "finalize"),
        ("fail", "finalize"),
    ],
)
def test_route_after_summarize_maps_supported_action_explicitly(
    next_action: str,
    expected_node: str,
) -> None:
    state = _state()
    state["next_action"] = next_action

    assert _route_after_summarize(state) == expected_node
```

- [ ] **Step 2: 잘못된 action과 누락된 action의 실패 테스트 작성**

```python
@pytest.mark.parametrize("next_action", ["create_plan", "unknown_action", None])
def test_route_after_summarize_rejects_unsupported_action(next_action: str | None) -> None:
    state = _state()
    if next_action is None:
        state.pop("next_action", None)
    else:
        state["next_action"] = next_action

    with pytest.raises(ValueError, match="지원하지 않는 next_action"):
        _route_after_summarize(state)
```

`create_plan`은 clarification 경로에서는 유효하지만 `summarize_step` 이후 action으로는 더 이상 허용하지 않는다는 것을 이 테스트로 고정한다.

- [ ] **Step 3: terminal state 우선순위 보존 테스트 작성**

```python
def test_route_after_summarize_prioritizes_terminal_state() -> None:
    state = _state()
    state["terminal_state"] = "failed_terminal"
    state.pop("next_action", None)

    assert _route_after_summarize(state) == "finalize"
```

- [ ] **Step 4: 라우팅 테스트를 실행해 fallback 때문에 오류 테스트가 실패하는지 확인**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
.venv/bin/python -m pytest DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py -k "route_after_summarize" -v
```

Expected: 허용 매핑은 PASS하고, `create_plan`, `unknown_action`, `None` 오류 테스트는 기존 fallback이 `decide_next_action`을 반환하므로 FAIL한다.

- [ ] **Step 5: `_route_after_summarize()`를 명시적이고 폐쇄적으로 구현**

`supervisor/graph.py`의 함수를 다음 코드로 교체한다.

```python
def _route_after_summarize(state: SupervisorState) -> str:
    if state.get("terminal_state") in TERMINAL_STATES:
        return "finalize"

    next_action = state.get("next_action")
    if next_action == "decide_next_action":
        return "decide_next_action"
    if next_action == "call_report_agent":
        return "generate_report"
    if next_action in SUBAGENT_ACTION_TO_AGENT:
        return "execute_subagent"
    if next_action in {"finalize", "fail"}:
        return "finalize"

    raise ValueError(f"summarize_step 이후 지원하지 않는 next_action입니다: {next_action!r}")
```

`build_graph()`의 conditional edge mapping에는 이미 `decide_next_action`, `execute_subagent`, `generate_report`, `finalize`가 등록되어 있으므로 노드나 edge를 추가하지 않는다.

- [ ] **Step 6: Semantic Validation의 기본 재판단 action 정렬**

`make_semantic_validate_subagent_result_node()`의 advisory LLM 실패 경로와 정상 경로에서 state에 `next_action`이 없을 때 쓰는 기본값을 모두 다음처럼 변경한다.

```python
"next_action": state.get("next_action", "decide_next_action"),
```

기존 `"create_plan"` 기본값은 실제 계획 생성과 의미가 다르므로 남기지 않는다. 정상 Hard Validation 흐름에서는 이미 설정된 `next_action="decide_next_action"`을 그대로 보존한다.

- [ ] **Step 7: 라우팅 단위 테스트 통과 확인**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
.venv/bin/python -m pytest DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py -k "route_after_summarize" -v
```

Expected: 허용 action, 잘못된 action, terminal 우선순위 테스트가 모두 PASS한다.

- [ ] **Step 8: 명시적 라우팅 변경 커밋**

```bash
git add DATA_Analyst_Assistant_Agent/supervisor/graph.py \
        DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py
git commit -m "refactor: summarize 이후 action 라우팅 명시화"
```

---

### Task 3: LLM 기록 용어와 통합 테스트를 새 계약에 정렬

**Files:**
- Modify: `DATA_Analyst_Assistant_Agent/supervisor/prompts.py:115-150`
- Modify: `DATA_Analyst_Assistant_Agent/supervisor/prompts.py:191-218`
- Modify: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py:146-221`
- Modify: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py:583-628`
- Modify: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py:724-810`
- Modify: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_decision.py:299-317`

**Interfaces:**
- Consumes: `RESULT_VALIDATION_DECISION_PROMPT`, `STEP_SUMMARY_DECISION_PROMPT`, `_summary_decision()` 테스트 fixture, `build_next_action_context()`
- Produces: 검증 후 재판단을 `decide_next_action`으로 기록하는 프롬프트·fixture와 내부 action 비노출 회귀 테스트

- [ ] **Step 1: Step Summary fixture와 semantic advisory 상태를 새 action으로 변경**

`tests/supervisor/test_graph.py`의 helper 기본값을 변경한다.

```python
def _summary_decision(agent: str, next_action: str = "decide_next_action") -> dict[str, Any]:
    return {
        "step": "validate_subagent_result",
        "agent": agent,
        "action": f"call_{agent}",
        "summary": f"{agent} 단계 요약",
        "artifact_ids": [f"artifact_{agent}"],
        "next_action": next_action,
        "reason": "summary decision",
    }
```

`_agent_flow_decisions()`의 기본 목록도 변경한다.

```python
summary_next_actions = summary_next_actions or ["decide_next_action"] * len(actions)
```

일반 성공 흐름을 표현하는 명시적 `summary_next_actions`와 semantic validation 테스트 state의 `next_action`/validation result도 `"create_plan"`에서 `"decide_next_action"`으로 변경한다. 재시도 테스트의 첫 번째 요약 action인 `"call_sql_agent"`는 유지하고, 재시도 성공 뒤 요약 action만 `"decide_next_action"`으로 바꾼다.

- [ ] **Step 2: 다음 행동 결정 컨텍스트가 내부 action을 노출하지 않는 테스트 보강**

`tests/supervisor/test_decision.py`의 compact snapshot 테스트에서 기존 `available_next_actions` 기대 목록은 그대로 유지하고 다음 assertion을 추가한다.

```python
assert "decide_next_action" not in snapshot["available_next_actions"]
```

이 테스트는 `NextAction` 타입에 값이 추가되어도 다음 행동 결정 LLM이 자기 자신을 선택하는 action을 받지 않도록 경계를 고정한다.

- [ ] **Step 3: 결과 검증과 Step Summary 프롬프트의 실패 테스트 작성**

`tests/supervisor/test_decision.py`에 다음 import를 추가한다.

```python
from DATA_Analyst_Assistant_Agent.supervisor.prompts import (
    RESULT_VALIDATION_DECISION_PROMPT,
    STEP_SUMMARY_DECISION_PROMPT,
)
```

같은 파일에 다음 테스트를 추가한다.

```python
def test_validation_and_summary_prompts_use_explicit_redecision_action() -> None:
    assert "- decide_next_action" in RESULT_VALIDATION_DECISION_PROMPT
    assert '"next_action":"decide_next_action"' in RESULT_VALIDATION_DECISION_PROMPT
    assert "- decide_next_action" in STEP_SUMMARY_DECISION_PROMPT
    assert '"next_action":"decide_next_action"' in STEP_SUMMARY_DECISION_PROMPT
```

- [ ] **Step 4: 프롬프트 계약 테스트가 기존 문자열에서 실패하는지 확인**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
.venv/bin/python -m pytest \
  DATA_Analyst_Assistant_Agent/tests/supervisor/test_decision.py \
  -k "explicit_redecision_action" -v
```

Expected: 두 프롬프트에 `decide_next_action`이 아직 없으므로 테스트가 FAIL한다.

- [ ] **Step 5: 결과 검증 프롬프트의 재판단 action 용어 변경**

`RESULT_VALIDATION_DECISION_PROMPT`의 허용 action 목록에서 검증 후 재판단 의미로 쓰이던 `create_plan`을 `decide_next_action`으로 교체하고, 성공 예시를 다음처럼 바꾼다.

```text
{"valid":true,"next_action":"decide_next_action","reason":"SQL 결과가 유효해 다음 행동을 다시 판단합니다.","terminal_state":"running","final_answer":""}
```

이 프롬프트는 현재 그래프의 Hard Validation에서 직접 호출되지 않더라도 동일 계약을 설명하므로 실제 코드 의미와 맞춰 둔다.

- [ ] **Step 6: Step Summary 프롬프트의 성공 후 action 용어 변경**

`STEP_SUMMARY_DECISION_PROMPT`의 허용 action 목록에서 검증 후 재판단 의미로 쓰이던 `create_plan`을 `decide_next_action`으로 교체하고 예시를 다음처럼 변경한다.

```text
{"step":"validate_subagent_result","agent":"sql_agent","action":"call_sql_agent","summary":"월별 매출 집계 SQL 산출물이 생성되었습니다.","artifact_ids":["artifact_sql"],"next_action":"decide_next_action","reason":"Supervisor가 다음 행동을 다시 판단할 수 있도록 요약합니다."}
```

`CLARIFY_DECISION_PROMPT`, clarification 노드의 `next_action="create_plan"`, `DECIDE_NEXT_ACTION_PROMPT`, `build_next_action_context().available_next_actions`는 변경하지 않는다.

- [ ] **Step 7: 프롬프트 계약 테스트 통과 확인**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
.venv/bin/python -m pytest \
  DATA_Analyst_Assistant_Agent/tests/supervisor/test_decision.py \
  -k "explicit_redecision_action or compact_json_snapshot" -v
```

Expected: 프롬프트 용어 테스트와 내부 action 비노출 테스트가 모두 PASS한다.

- [ ] **Step 8: 성공·재시도·clarification 통합 흐름 확인**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
.venv/bin/python -m pytest DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py -v
```

Expected:

- 전체 SQL → EDA → Analysis → Report 흐름이 PASS한다.
- Hard Validation 성공 후 `semantic_validate_subagent_result → summarize_step → decide_next_action` 순서가 유지된다.
- 재시도 가능한 SQL 실패는 `summarize_step → execute_subagent`로 이동해 SQL Agent를 즉시 재실행한다.
- clarification interrupt resume은 `collect_clarification → create_analysis_plan`으로 이동한다.
- approval 및 report 종료 테스트가 PASS한다.

- [ ] **Step 9: 프롬프트 및 fixture 정렬 커밋**

```bash
git add DATA_Analyst_Assistant_Agent/supervisor/prompts.py \
        DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py \
        DATA_Analyst_Assistant_Agent/tests/supervisor/test_decision.py
git commit -m "test: supervisor 재판단 action 계약 정렬"
```

---

### Task 4: Supervisor 회귀 테스트와 전체 테스트 검증

**Files:**
- Verify: `DATA_Analyst_Assistant_Agent/supervisor/`
- Verify: `DATA_Analyst_Assistant_Agent/tests/supervisor/`
- Verify: `tests/`, `data_agent_backend/tests/`, `DATA_Analyst_Assistant_Agent/tests/`

**Interfaces:**
- Consumes: Tasks 1~3에서 변경한 action 계약과 그래프 라우팅
- Produces: 기존 Supervisor 동작 및 저장소 전체에 대한 회귀 테스트 결과

- [ ] **Step 1: Supervisor 테스트 전체 실행**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
.venv/bin/python -m pytest DATA_Analyst_Assistant_Agent/tests/supervisor -v
```

Expected: Supervisor 테스트가 모두 PASS한다.

- [ ] **Step 2: 남아 있는 잘못된 validation 의미 검색**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
rg -n 'next_action="create_plan"|"next_action": "create_plan"|next_action.*create_plan' \
  DATA_Analyst_Assistant_Agent/supervisor \
  DATA_Analyst_Assistant_Agent/tests/supervisor
```

Expected: clarification 및 실제 계획 생성 의미의 사용만 남는다. Hard Validation 성공, Step Summary 성공 fixture, semantic validation 성공 상태에는 `create_plan`이 남지 않는다.

- [ ] **Step 3: 새 재판단 action 사용 위치 확인**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
rg -n 'decide_next_action' \
  DATA_Analyst_Assistant_Agent/supervisor/state.py \
  DATA_Analyst_Assistant_Agent/supervisor/validation.py \
  DATA_Analyst_Assistant_Agent/supervisor/graph.py \
  DATA_Analyst_Assistant_Agent/supervisor/prompts.py \
  DATA_Analyst_Assistant_Agent/tests/supervisor
```

Expected: 타입 계약, validation 성공 결과, 명시적 summarize 라우팅, 관련 프롬프트와 테스트에서 새 action이 확인된다. `build_next_action_context().available_next_actions`에는 포함되지 않는다.

- [ ] **Step 4: 저장소 전체 테스트 실행**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
.venv/bin/python -m pytest tests data_agent_backend/tests DATA_Analyst_Assistant_Agent/tests
```

Expected: 전체 테스트가 PASS한다. 환경변수나 외부 서비스가 필요한 기존 테스트가 skip되면 skip 사유를 최종 보고에 기록한다.

- [ ] **Step 5: 최종 diff 검토**

Run:

```bash
cd /Users/kyoho/Documents/DAAAT/AI-agent
git diff --check
git diff -- DATA_Analyst_Assistant_Agent/supervisor \
            DATA_Analyst_Assistant_Agent/tests/supervisor
```

Expected: whitespace 오류가 없고, 변경 범위가 action 계약·라우팅·프롬프트·관련 테스트로 제한된다.

- [ ] **Step 6: 검증 보완 커밋**

검증 과정에서 테스트 보완이 발생한 경우에만 실행한다.

```bash
git add DATA_Analyst_Assistant_Agent/tests/supervisor
git commit -m "test: supervisor 명시적 라우팅 회귀 검증"
```

검증 보완 변경이 없다면 추가 커밋을 만들지 않는다.
