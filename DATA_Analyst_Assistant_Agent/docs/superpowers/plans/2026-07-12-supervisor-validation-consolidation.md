# Supervisor Validation 통합 리팩터링 구현 계획

> **에이전트 작업자용:** 이 계획은 `superpowers:executing-plans`와 `superpowers:test-driven-development`를 사용해 순서대로 구현한다.

**목표:** 외부 에이전트·백엔드 계약을 유지하면서 Supervisor 내부의 Hard, Evidence, Semantic Validation을 단일 파이프라인과 `ValidationRecord` 계약으로 통합한다.

**아키텍처:** `stage_candidate → validate_candidate → resolve_validation` 흐름으로 검증 계산과 상태·라우팅 변경을 분리한다. 신규 상태는 v3 `validation_history`만 기록하고, v2의 세 검증 배열은 정규화 시 통합 이력으로 변환한다.

**기술 스택:** Python 3.12, Pydantic, LangGraph, pytest

## 전역 제약

- `AgentEnvelope`, `AgentToolResult`, `AgentCompactResult`, `SubAgentAdapter`, `BackendAdapter` 외부 계약을 변경하지 않는다.
- Semantic Validation은 모든 결정론적 검증 통과 후보에 실행한다.
- Hard Validation의 재시도 가능 실패만 재시도하고 Evidence/Semantic 실패는 종료하는 현재 정책을 유지한다.
- 후보 격리, 승인 ID, validation ID, content hash 구조와 기존 Backend 이벤트 이름을 유지한다.

---

### 작업 1: 통합 검증 계약과 변환 함수

**파일:**
- 수정: `DATA_Analyst_Assistant_Agent/supervisor/validation.py`
- 테스트: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_validation.py`

- [ ] `ValidationCheckResult`, `ValidationOutcome`, `ValidationRecord` 모델의 실패 테스트를 작성한다.
- [ ] 기존 Hard/Evidence/Semantic 결과를 통합 check와 outcome으로 변환하는 순수 함수 테스트를 작성하고 실패를 확인한다.
- [ ] 최소 구현 후 대상 테스트를 통과시킨다.

### 작업 2: 단일 Validation 파이프라인

**파일:**
- 수정: `DATA_Analyst_Assistant_Agent/supervisor/graph.py`
- 테스트: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py`

- [ ] contract → result → evidence → semantic 순서, 단락 평가, 승인 확정 순서의 실패 테스트를 작성한다.
- [ ] `validate_candidate()`를 구현해 하나의 `ValidationRecord`를 반환하게 한다.
- [ ] Semantic 모델 1회 재시도와 반복 실패 종료 동작을 보존하고 대상 테스트를 통과시킨다.

### 작업 3: 상태 변경을 resolve_validation으로 통합

**파일:**
- 수정: `DATA_Analyst_Assistant_Agent/supervisor/graph.py`
- 테스트: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_graph.py`

- [ ] 후보 거부·격리·이벤트·retry count가 한 번만 반영되는 실패 테스트를 작성한다.
- [ ] `resolve_validation`만 검증 이력과 후보 승격/거부, 재시도, 종료 상태를 변경하도록 구현한다.
- [ ] 그래프를 두 통합 노드와 disposition 라우터로 교체한다.

### 작업 4: SupervisorState v3 마이그레이션

**파일:**
- 수정: `DATA_Analyst_Assistant_Agent/supervisor/state.py`
- 테스트: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_state.py`

- [ ] v2 세 검증 배열과 승인 대기 체크포인트 변환 테스트를 작성한다.
- [ ] `validation_history`, `state_schema_version=3`을 도입하고 legacy 항목을 순서대로 병합한다.
- [ ] 신규 상태에서 구형 검증 배열을 기록하지 않으며 기존 상태 필드를 보존한다.

### 작업 5: Decision·Summary·Finalization 컨텍스트 호환

**파일:**
- 수정: `DATA_Analyst_Assistant_Agent/supervisor/decision.py`
- 수정: `DATA_Analyst_Assistant_Agent/supervisor/reporting.py`
- 수정: `DATA_Analyst_Assistant_Agent/supervisor/summarizer.py`
- 테스트: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_decision.py`
- 테스트: `DATA_Analyst_Assistant_Agent/tests/supervisor/test_reporting.py`

- [ ] 통합 이력에서 기존 프롬프트 JSON 키와 의미를 파생하는 실패 테스트를 작성한다.
- [ ] 모든 컨텍스트 생성기가 `validation_history`를 원본으로 사용하도록 구현한다.
- [ ] 기존 프롬프트 payload 회귀 테스트를 통과시킨다.

### 작업 6: 구형 경로 제거와 전체 검증

**파일:**
- 수정: `DATA_Analyst_Assistant_Agent/supervisor/graph.py`
- 수정: 관련 Supervisor 테스트

- [ ] 사용되지 않는 개별 Validation 노드·라우터·상태 기록을 제거한다.
- [ ] `python -m pytest DATA_Analyst_Assistant_Agent/tests/supervisor -q`를 실행한다.
- [ ] `python -m pytest tests data_agent_backend/tests DATA_Analyst_Assistant_Agent/tests`를 실행한다.
- [ ] adapter와 Backend 호출 계약이 유지되는지 회귀 결과를 확인한다.
