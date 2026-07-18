# Olist Query Rule Retrieval Design

## Goal

사용자 query와 가장 관련 높은 Olist 분석 규칙 문서 한 건을 Pinecone에서 검색하고, 문서 전문이 아닌 실행에 필요한 규칙만 구조화해 supervisor state에 저장한 뒤 기존 `AnalysisPlan` 전달 경로로 SQL agent에 제공한다.

핵심 제약은 retrieval 이후의 SQL LangGraph node와 EDA/Analysis agent를 변경하지 않는 것이다.

## Scope

- 쿼리 유형별 독립 Markdown 문서를 Pinecone record로 적재한다.
- query마다 `top_k=1`로 문서 한 건만 검색한다.
- retrieval node에서 문서 내용을 compact rule context로 변환한다.
- compact rule context를 `SupervisorState`와 `AnalysisPlan`에 저장한다.
- SQL agent가 이미 사용하는 `supervisor_plan_context -> planner_selection_reason` 경로로 규칙을 전달한다.
- Pinecone 설정 누락, 검색 실패, 결과 없음은 기존 분석을 중단시키지 않는다.

## Non-Goals

- 검색된 Markdown 전문을 checkpoint state에 저장하지 않는다.
- 사용자 query 문자열에 규칙을 이어 붙이지 않는다.
- SQL agent 내부 `AgentState`에 별도 retrieval 필드를 추가하지 않는다.
- SQL LangGraph의 plan/generate/validate node별 prompt를 각각 수정하지 않는다.
- EDA, Analysis, Insight agent에 규칙 전문을 전달하지 않는다.
- 한 query에서 문서 여러 건을 결합하지 않는다.

## Data Flow

```text
latest_user_query
  -> retrieve_analysis_rules
  -> Pinecone search(top_k=1, doc_type=analysis_query_rule)
  -> selected document text
  -> rule extraction/normalization
  -> SupervisorState.analysis_rule_context
  -> create_analysis_plan
  -> AnalysisPlan.query_rules
  -> to_orchestration_state
  -> SQLAgent._run_main_sql_agent
  -> supervisor_plan_context["query_rules"]
  -> existing planner_selection_reason
  -> existing SQL graph
```

## State Contract

`SupervisorState.analysis_rule_context`는 `dict | None`이다. Pinecone SDK 객체나 문서 전문을 보관하지 않는다.

```json
{
  "document_id": "purchase-frequency",
  "query_type": "purchase_frequency",
  "score": 0.91,
  "definition": "구매 빈도는 실제 고객별 distinct 주문 횟수다.",
  "default_metrics": ["COUNT(DISTINCT order_id)"],
  "entity_grain": "customer_unique_id",
  "required_tables": ["customers", "orders"],
  "constraints": [
    "매출 합계를 구매 빈도로 사용하지 않는다.",
    "customer_id를 고객 식별 grain으로 사용하지 않는다."
  ],
  "clarify_when": [
    "자주 구매한다는 임계값이 결과에 필요하지만 지정되지 않았다."
  ],
  "prohibited_interpretations": [
    "order_items 행 수를 주문 횟수로 해석하지 않는다."
  ],
  "source_path": "docs/olist_rag_context/analysis_rules/purchase_frequency.md",
  "version": "1.0"
}
```

`SupervisorState.analysis_rule_retrieval`은 검색 자체의 상태와 진단만 보관한다.

```json
{
  "status": "success",
  "query": "자주 구매하는 유저의 특징을 분석해줘",
  "selected_document_id": "purchase-frequency"
}
```

허용 status는 `not_started`, `success`, `empty`, `skipped`, `disabled`, `failed`다.

`AnalysisPlan.query_rules`는 위 compact context의 복사본이다. 기존 `AnalysisPlan`이 supervisor와 specialist agent 사이의 전달 계약이므로 별도 전달 모델을 만들지 않는다.

## Retrieval Node

`retrieve_analysis_rules`를 `START`와 `clarify_query` 사이에 둔다.

1. `clarified_query or latest_user_query`를 검색어로 선택한다.
2. 빈 query면 Pinecone을 호출하지 않고 `skipped`를 기록한다.
3. `doc_type=analysis_query_rule`, `top_k=1`로 검색한다.
4. 결과가 없으면 context를 `None`, status를 `empty`로 기록한다.
5. 결과가 있으면 문서의 구조화 section을 local parser로 읽어 compact rule context를 만든다.
6. parsing 결과는 허용 필드와 최대 항목 수를 검증한 뒤 state에 저장한다.
7. 설정 누락과 검색 오류는 각각 `disabled`, `failed`로 기록하고 `clarify_query`로 계속 진행한다.

Rule extraction은 별도 LLM 호출을 사용하지 않는다. 규칙 문서가 정해진 section과 machine-readable front matter를 갖도록 validation하여 결정적으로 추출한다.

## SQL Agent Propagation

`create_analysis_plan`은 선택된 compact context를 `analysis_plan["query_rules"]`에 복사한다. `to_orchestration_state()`가 이를 `AnalysisPlan.query_rules`로 검증한다.

`SQLAgent._run_main_sql_agent()`는 현재 구성하는 `supervisor_plan_context`에 다음 한 필드만 추가한다.

```python
"query_rules": state.plan.query_rules,
```

기존 코드는 이 context 전체를 `planner_selection_reason`에 JSON으로 포함하고 SQL graph의 첫 planner prompt에 전달한다. 따라서 SQL 내부 state와 후속 node는 변경하지 않는다.

## Prompt And Precedence Rules

SQL planner가 context를 해석할 때 우선순위는 다음과 같다.

1. 실제 datasource catalog와 schema
2. integrity/validation 결과
3. 검색된 Olist query rule
4. LLM 일반 지식

규칙 문서가 schema와 충돌하면 schema가 우선이다. 규칙은 metric 정의, grain, join/filter 주의사항을 제공하지만 live 데이터 값이나 분석 결과의 근거가 아니다.

## Error Handling

- Pinecone API key 없음: `disabled`, 분석 계속
- 빈 query: `skipped`, 분석 계속
- 검색 결과 없음: `empty`, 분석 계속
- timeout/API exception: `failed`, limitation과 run event 기록 후 분석 계속
- 문서 parsing 실패: `failed`, 원문을 state에 넣지 않고 분석 계속
- 선택 문서가 복합 query의 주목적과 맞지 않음: 복수 문서를 추가 검색하지 않고 clarification 대상으로 처리

## Test Strategy

### Unit

- 규칙 문서에서 compact context가 결정적으로 추출된다.
- 허용되지 않은 필드와 과도한 항목 수는 validation에서 거부된다.
- retrieval은 `top_k=1`과 metadata filter를 사용한다.
- success/empty/skipped/disabled/failed state가 각각 정확히 생성된다.
- state에는 원문 전문과 Pinecone SDK 객체가 남지 않는다.

### State And Graph

- empty state에 신규 기본값이 존재한다.
- 기존 checkpoint normalization이 신규 필드를 채운다.
- graph 시작 순서가 `retrieve_analysis_rules -> clarify_query`다.
- retrieval 실패 후에도 clarification node가 실행된다.
- create-plan 결과에 `query_rules`가 복사된다.

### SQL Integration

- `to_orchestration_state()`가 `AnalysisPlan.query_rules`를 보존한다.
- SQL agent가 기존 `planner_selection_reason`에 `query_rules`를 포함한다.
- “자주 구매하는 고객” 규칙은 distinct 주문 횟수와 `customer_unique_id` grain을 SQL planner 입력에 전달한다.
- query rule이 없을 때 기존 SQL agent 입력이 의미상 동일하게 유지된다.

## Compatibility Boundary

변경 대상은 retrieval/document 계층, supervisor state/graph/plan contract, SQL agent 진입 adapter다. SQL LangGraph node, EDA, Analysis, Insight 구현은 변경 대상이 아니다.
