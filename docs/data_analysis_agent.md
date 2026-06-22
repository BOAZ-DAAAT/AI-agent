# 데이터 분석 에이전트

## 목적

데이터 분석 에이전트는 SQL 결과 아티팩트와 선택적인 EDA 프로파일을 입력받아,
근거를 추적할 수 있는 구조화된 `AnalysisResult`를 생성한다. 기존 supervisor가
사용하는 `run(state, runtime) -> AgentEnvelope` 호출 규약은 그대로 유지한다.

설계에는 LangChain 공식 문서의 다음 원칙을 반영했다.

- 자유 형식의 도구 지시문 대신 타입이 정의된 도구를 사용한다.
- planner와 최종 결과를 Pydantic 스키마로 검증한다.
- 전체 상태나 데이터셋 대신 작업에 필요한 범위로 제한한 컨텍스트를 전달한다.
- 사람의 검토가 필요한 상황을 명시적인 출력 계약으로 표현한다.
- 통계 계산은 결정론적 도구가 담당하고, LLM은 매 실행마다 분석 계획을 수립한다.

참고 문서:

- <https://docs.langchain.com/oss/python/langchain/agents>
- <https://docs.langchain.com/oss/python/langchain/tools>
- <https://docs.langchain.com/oss/python/langchain/structured-output>
- <https://docs.langchain.com/oss/python/langchain/context-engineering>
- <https://docs.langchain.com/oss/python/langchain/human-in-the-loop>
- <https://docs.langchain.com/oss/python/langgraph/workflows-agents>

## 패키지 구조

```text
agents/analysis/
|-- agent.py       # 아티팩트 입출력과 AgentEnvelope 경계
|-- context.py     # 분석에 필요한 범위로 컨텍스트 구성
|-- planner.py     # 필수 LLM 구조화 계획 및 계획 검증
|-- prompts.py     # planner 시스템 프롬프트와 도구 카탈로그
|-- schemas.py     # 입력, 계획, 근거, 결과, 검토용 Pydantic 계약
|-- tools.py       # 결정론적 통계 계산을 수행하는 LangChain 도구
|-- methods.py     # 계획 실행과 구조화된 결과 조립
|-- insight.py     # 계산 근거를 기반으로 결과 해석과 가설 판단
|-- self_check.py  # 로컬 계약 및 계획-근거 추적성 검증
`-- workflow.py    # 내부 LangGraph: plan -> execute -> validate -> end
```

## 실행 흐름

1. `AnalysisAgent`가 등록된 SQL CSV 및 EDA JSON 아티팩트를 읽는다.
2. `build_analysis_context`가 스키마, 컬럼 프로파일, 분석 힌트, 품질 신호와 최대 5개의 샘플 행만 구성한다.
3. `build_analysis_plan`이 supervisor가 전달한 질문 유형을 우선 사용해 세부 분석 방법을 결정한다.
4. `with_structured_output`을 사용해 LLM에서 `AnalysisExecutionPlan`을 생성한다.
   모델 설정, 호출 또는 계획 검증이 실패하면 실행도 실패하며 다른 planner로 대체하지 않는다.
5. 선택된 LangChain 도구가 아티팩트 데이터를 바탕으로 통계를 계산한다.
6. `AnalysisResult`가 각 결과를 도구 실행 근거와 연결하고 한계 사항을 기록한다.
7. `run_analysis_self_check`가 결과 스키마, 질문 유형 정합성, 계획한 모든 도구의 실행 여부와 근거 추적성을 검증한다.
8. 최종 JSON 아티팩트의 `parent_ids`에 입력 SQL 및 EDA 아티팩트 ID를 기록하고
   `derived_from` lineage로 연결한다.

여기서 `parent_ids`는 폴더나 객체 계층에서의 "하위 아티팩트"라는 뜻이 아니다.
분석 결과가 어떤 입력 근거에서 파생됐는지 추적하기 위한 provenance 관계다. 이 연결을
제거하면 report와 validation에서 결과의 원천을 역추적하기 어려워진다.

## LangChain 문서 반영 범위

LangChain 문서의 모든 기능을 한 에이전트에 무조건 넣지는 않는다. analysis 에이전트의
책임에 해당하는 기능은 직접 반영하고, 애플리케이션 또는 supervisor 책임은 경계를
명시한다.

| 문서 주제 | 반영 상태 | 적용 방식 또는 책임 경계 |
| --- | --- | --- |
| Models | 반영 | 공용 `get_chat_model`을 통해 provider 독립 모델 사용 |
| Messages | 반영 | `SystemMessage`와 `HumanMessage`로 planner 입력 구성 |
| Tools | 반영 | 타입 스키마가 있는 LangChain 도구 8개 사용 |
| Structured output | 반영 | LLM 계획과 최종 `AnalysisResult`를 Pydantic으로 검증 |
| Context engineering | 반영 | 컬럼 프로파일, 작은 샘플, EDA 품질 신호만 LLM에 제공 |
| Guardrails | 반영 | 허용 도구·컬럼·질문 유형 검사 및 실행 후 self-check |
| Human-in-the-loop | 계약 반영 | analysis가 검토 필요성을 반환하고 실제 중단·재개는 supervisor 담당 |
| Custom workflow | 반영 | LangGraph `plan -> execute -> validate` 그래프 사용 |
| `create_agent` | 미사용 | 자유로운 ReAct 루프보다 실행 단계가 고정된 custom workflow가 이 분석에 적합 |
| Middleware | 미사용 | `create_agent` 수명주기를 사용하지 않으며 graph node에서 동일 경계 검증 수행 |
| Short/long-term memory | supervisor 책임 | 실행 간 대화·사용자 메모리는 analysis 아티팩트 계산과 분리 |
| Streaming | API/UI 책임 | 토큰·진행 이벤트 전송은 상위 실행 계층에서 처리 |
| Multi-agent routing | supervisor 책임 | SQL·EDA·analysis 간 선택과 반복은 supervisor가 처리 |

따라서 "문서의 기능을 전부 넣었다"가 아니라, analysis 책임에 필요한 공식 패턴을
선택해 구현하고 나머지는 소유 계층을 분리한 구조다.

## Supervisor 질문 유형 전달 규약

향후 supervisor는 현재 공용 state 모델을 변경하지 않고도 다음과 같이 질문 유형을
전달할 수 있다.

```python
envelope = analysis_agent.run(state, runtime, question_type="prediction")
```

추후 공용 계약에 필드가 추가되면 `state.question_type` 또는
`state.plan.question_type`에서도 자동으로 읽는다. `run`에 직접 전달한 값의 우선순위가
가장 높으며, 어떤 유형도 전달되지 않았을 때만 사용자 질문 문장에서 유형을 추론한다.

| Supervisor 질문 유형 | 분석 유형 | 세부 분석 및 도구 |
| --- | --- | --- |
| `descriptive` | 기술 분석 | 기술통계 요약 |
| `comparison` | 그룹 비교 | 2개 그룹은 Welch t-test, 3개 이상은 일원분산분석 |
| `correlation` | 상관 분석 | Pearson 상관계수 및 p-value |
| `trend` | 추세 분석 | 기울기, p-value, 설명력을 포함한 순서 기반 선형 추세 |
| `prediction` | 회귀 분석 | holdout 평가를 적용한 선형회귀 기준 모델 |
| `classification` | 분류 분석 | stratified holdout 평가를 적용한 로지스틱 회귀 기준 모델 |
| `anomaly_detection` | 이상탐지 | Isolation Forest |
| `causal` | 상관 근거 분석 | 연관성 분석 후 사람의 검토 요청 |

추가 기업 분석 capability:

- 고객·제품: `cohort`, `retention`, `funnel`, `journey`, `segmentation`, `rfm`, `churn`, `survival`
- 매출·마케팅·재무: `contribution`, `mix_shift`, `profitability`, `unit_economics`, `attribution`, `scenario`
- 운영·리스크·VOC: `demand`, `inventory`, `root_cause`, `text`

각 capability는 [Analysis Agent 팀 공유 문서](./analysis_agent_team_guide.md)의 데이터 역할과
계산 방법을 따른다.

계획에는 상위 유형인 `question_type`과 실제 적용된 세부 유형인
`analysis_subtype`을 모두 보존한다. 따라서 downstream validation에서 요청 유형과 실제
통계 방법이 일치하는지 검사할 수 있다.

## 구조화된 출력

기존 downstream 코드가 사용하는 다음 필드는 유지한다.

- `method_summary`
- `key_findings`
- `limitations`
- `source_artifacts`
- `data_quality_notes`

다음 필드는 에이전트의 제어 가능성과 감사 가능성을 높이기 위해 추가했다.

- `plan`: 질문 유형, 세부 분석 유형, 선택한 도구와 컬럼, 신뢰성 조건, planner 모드
- `evidence`: 도구 이름, 입력값, 계산된 통계, 결과 문장, 주의 사항
- `hypotheses`: 귀무가설, 대립가설, 판단과 판단 근거
- `human_review`: 사람의 검토 필요 여부와 사유

## Human-in-the-loop 규약

인과관계를 요구하는 질문은 현재 도구가 관측 데이터의 연관성만 분석할 수 있으므로
`human_review.required=true`로 설정한다. `AnalysisAgent`는 이 값을
`approval_type="analysis.review"`인 `AgentEnvelope.approval`에도 반영한다.

실제 중단, 승인 요청 저장, 사용자 결정 처리 및 재개는 supervisor의 책임이다.
analysis 패키지는 supervisor 상태를 직접 변경하지 않는다.

## 안전 및 신뢰성 규칙

- 도구는 제한된 컨텍스트에 존재하는 컬럼만 참조할 수 있다.
- LLM planner 결과는 허용된 도구와 컬럼 목록을 기준으로 다시 검사한다.
- 도구 실행 실패를 통계 결과로 꾸미지 않고 한계 사항으로 기록하며 validation을 실패시킨다.
- 상관 분석 결과에는 인과관계 근거가 아니라는 주의 사항을 항상 포함한다.
- SQL 결과가 비었거나 읽을 수 없어도 통계를 만들어내지 않고 유효한 빈 결과를 반환한다.
- LLM planner는 필수이며 API 키 또는 모델 호출이 없으면 실패한다. 결정론적 fallback은 없다.
- 회귀는 고정 holdout의 R2·MAE·RMSE, 분류는 stratified holdout의 accuracy·balanced accuracy·F1·ROC-AUC를 제공한다.
- 그룹 검정은 p-value뿐 아니라 Cohen's d 또는 eta-squared와 Levene 진단을 제공한다.
- 가설검정과 추세 계수 검정은 statsmodels 결과 객체와 95% 신뢰구간을 사용한다.
- 상관 분석은 Benjamini-Hochberg 방식으로 다중검정 p-value를 보정한다.
- 이상탐지는 점수가 낮은 후보 행을 반환하지만 오류나 부정행위로 단정하지 않는다.

## 테스트

단위 테스트는 실제 외부 API 대신 구조화 출력을 반환하는 가짜 채팅 모델을 주입한다.
production 경로에서는 모델을 주입하지 않으므로 공용 `get_chat_model`이 항상 호출된다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_analysis_agent.py -q
```
