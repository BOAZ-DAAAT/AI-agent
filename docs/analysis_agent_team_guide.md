# Analysis Agent 팀 공유 문서

## 1. 한 줄 설명

Analysis Agent는 supervisor가 전달한 **사용자 질문 유형**과 SQL·EDA 결과를 바탕으로,
LLM이 분석 계획을 세우고 검증된 통계 도구가 실제 계산을 수행한 뒤, 근거와 한계가
포함된 구조화된 분석 결과를 반환하는 전문 에이전트다.

핵심 원칙은 다음과 같다.

> LLM은 무엇을 분석할지 계획하고, 통계 도구는 실제 수치를 계산하며,
> validation node는 요청한 분석과 실행 결과가 일치하는지 검사한다.

---

## 2. 전체 구조에서의 역할

전체 시스템은 supervisor가 SQL, EDA, Analysis 등의 전문 에이전트를 선택하고 결과를
다음 에이전트에 전달하는 구조다.

```text
사용자 질문
    ↓
Supervisor Plan
    ↓
SQL Agent → 데이터 조회 및 분석용 결과 생성
    ↓
EDA Agent → 데이터 구조·분포·품질 탐색
    ↓
Analysis Agent → 통계 검정·예측·분류·이상탐지·추세 분석
    ↓
Validation / Visualization / Report
```

Analysis Agent의 책임은 다음과 같다.

- SQL 결과 데이터를 실제 통계 분석에 사용한다.
- EDA 결과에서 데이터 품질과 주의 사항을 가져온다.
- supervisor가 분류한 질문 유형을 세부 분석 방법으로 구체화한다.
- LLM을 이용해 구조화된 분석 계획을 반드시 생성한다.
- 계산은 LLM이 직접 만들지 않고 검증 가능한 LangChain 도구로 수행한다.
- 분석 결과와 사용한 통계값, 가설, 한계, 입력 아티팩트를 함께 반환한다.
- 인과 해석처럼 사람의 판단이 필요한 요청은 HITL 대상으로 표시한다.

Analysis Agent가 담당하지 않는 것은 다음과 같다.

- 데이터베이스 조회 및 SQL 생성: SQL Agent 책임
- 전체 데이터 프로파일링과 시각적 탐색: EDA Agent 책임
- 에이전트 간 순서 결정과 반복 실행: Supervisor 책임
- 최종 보고서 통합: Report Agent 책임
- 실제 사용자 승인 중단·재개: Supervisor 책임

---

## 3. 입력

Analysis Agent의 주요 입력은 다음과 같다.

| 입력 | 설명 |
| --- | --- |
| `OrchestrationState` | run ID, 사용자 질문, 목표, SQL/EDA 아티팩트 ID 등의 실행 상태 |
| `question_type` | supervisor가 판단한 상위 질문 유형 |
| SQL result artifact | 분석 대상 데이터가 담긴 CSV 아티팩트 |
| EDA profile artifact | 데이터 품질 상태, 결측, 주요 이슈 등의 JSON 아티팩트 |
| `AgentRuntime` | 아티팩트 조회·등록과 실행 컨텍스트를 제공하는 공용 runtime |

Supervisor 연동 형태는 다음과 같다.

```python
envelope = analysis_agent.run(
    state,
    runtime,
    question_type="prediction",
)
```

우선순위는 다음과 같다.

1. `AnalysisAgent.run(..., question_type=...)`에 직접 전달한 값
2. 향후 추가될 `state.question_type`
3. 향후 추가될 `state.plan.question_type`
4. 별도 유형이 없으면 LLM이 사용자 질문과 컨텍스트를 바탕으로 판단

---

## 4. 내부 LangGraph 실행 흐름

Analysis Agent 내부는 다음 세 노드로 구성된다.

```mermaid
flowchart LR
    A["START"] --> B["Plan Node"]
    B --> C["Execute Node"]
    C --> D["Validate Node"]
    D --> E["END"]
```

### 4.1 Plan Node

Plan Node는 매 실행마다 LLM을 호출해 `AnalysisExecutionPlan`을 생성한다.

LLM에 전달하는 컨텍스트는 전체 데이터가 아니라 다음 정보로 제한한다.

- 사용자 질문과 분석 목표
- supervisor가 전달한 질문 유형
- metric·dimension 힌트
- 행 수와 컬럼 목록
- 수치형·범주형·시간형 컬럼 목록
- 컬럼별 dtype, 결측 수, 고유값 수, 기초 통계
- 최대 5개의 샘플 행
- EDA 품질 상태와 주요 이슈
- 입력 아티팩트 ID

LLM 계획은 Pydantic 스키마로 강제되며 다음 내용을 포함한다.

- 상위 질문 유형 `question_type`
- 실제 적용할 분석 유형 `analysis_kind`
- 구체적인 방법 `analysis_subtype`
- 실행할 도구 순서 `tool_names`
- target, metric, dimension, time column, feature columns
- 신뢰성 조건
- 사람의 검토 필요 여부와 사유

LLM이 존재하지 않는 도구나 컬럼을 선택하거나 supervisor의 질문 유형을 임의로
변경하면 계획을 거부한다.

결정론적 planner fallback은 없다. 모델 설정, LLM 호출 또는 구조화 출력 검증이
실패하면 Analysis Agent 실행도 실패한다.

### 4.2 Execute Node

Execute Node는 LLM이 선택한 도구를 순서대로 실행한다.

LLM은 통계 수치를 직접 생성하지 않는다. 실제 결과는 Pandas, statsmodels,
scikit-learn을 사용하는 분석 도구에서 계산한다. SciPy는 Levene 검정과 같은 보조
진단 및 수치 계산에만 사용한다.

각 도구의 출력은 다음 정보와 함께 `AnalysisEvidence`로 저장된다.

- 실행한 도구 이름
- 사용한 입력 컬럼
- 계산 방법
- 실제 계산된 통계값
- 해당 통계에서 도출한 결과 문장
- 해석 시 주의 사항

### 4.3 Validate Node

Validate Node는 분석 결과가 다음 조건을 만족하는지 검사한다.

- 최종 출력이 `AnalysisResult` 스키마에 맞는가?
- `method_summary`, `key_findings`, `limitations`가 존재하는가?
- 모든 근거가 LLM 계획에 포함된 도구에서 생성됐는가?
- LLM이 계획한 모든 도구가 실제로 실행됐는가?
- supervisor 질문 유형과 실제 분석 유형이 일치하는가?
- HITL이 필요한 경우 검토 사유가 작성됐는가?

오류 등급의 검사가 하나라도 실패하면 `terminal_reason="validation_failed"`가 된다.
모두 통과하면 `terminal_reason="validated_result"`로 종료한다.

---

## 5. 기업 분석 Capability Catalog

DAAA는 범용 코딩 에이전트가 아니라 기업 데이터 의사결정 에이전트다. 따라서 분석
기능을 알고리즘 목록이 아니라 `업무 질문 유형 → 분석 섹터 → 필요한 데이터 역할 →
실행 도구` 형태의 capability catalog로 관리한다.

### 공통·통계·모델링

| Supervisor 질문 유형 | 세부 분석 | 주요 결과 |
| --- | --- | --- |
| `descriptive` | 기술통계 | 평균, 중앙값, 표준편차, 사분위수, IQR, 최소·최대 |
| `comparison` | 그룹 비교 | 그룹 집계, Welch t-test 또는 일원분산분석, 효과크기 |
| `correlation` | 상관 분석 | Pearson 상관계수, p-value, 다중검정 보정값 |
| `trend` | 추세 분석 | 기울기, 방향, p-value, 설명력, 표준오차 |
| `prediction` | 회귀 분석 | holdout R2, MAE, RMSE, 주요 계수 |
| `classification` | 분류 분석 | holdout accuracy, balanced accuracy, F1, ROC-AUC, 혼동행렬 |
| `anomaly_detection` | 이상탐지 | Isolation Forest 기반 이상 후보 수·비율·점수·행 위치 |
| `causal` | 연관성 근거 분석 | 상관 근거를 계산하되 인과 결론은 HITL 검토 요청 |

### 고객·제품 분석

| 질문 유형 | 분석 목적 | 필요 데이터 |
| --- | --- | --- |
| `cohort` | 유입 시점별 활동 변화 비교 | entity ID, event time, 선택적 cohort time |
| `retention` | 기간별 재방문·잔존율 계산 | entity ID, event time |
| `funnel` | 단계별 전환·이탈 측정 | entity ID, event, event time, ordered steps |
| `journey` | 주요 고객 경로와 이벤트 전이 분석 | entity ID, event, event time |
| `segmentation` | 행동 feature 기반 고객 군집화 | entity ID, numeric features |
| `rfm` | Recency·Frequency·Monetary 고객 가치 구분 | customer ID, transaction time, amount |
| `churn` | 고객 이탈 위험 baseline 분류 | churn target, customer features |
| `survival` | 이탈·재구매까지 걸리는 시간 분석 | duration, event observed |

### 매출·마케팅·재무 분석

| 질문 유형 | 분석 목적 | 필요 데이터 |
| --- | --- | --- |
| `contribution` | 그룹별 KPI 기여도와 Pareto 집중도 | dimension, metric |
| `mix_shift` | 최근 두 기간의 그룹별 증감 기여 | period, dimension, metric |
| `profitability` | 상품·고객·채널별 이익 기여 비교 | dimension, profit or margin metric |
| `unit_economics` | 단위별 매출·비용·이익 비교 | unit, revenue/cost/profit metric |
| `attribution` | 채널별 성과 기여도 설명 | channel, outcome metric |
| `scenario` | 명시적 변화율 기반 what-if 계산 | metric, change assumption |

`attribution`은 현재 기술적 기여도 분석이며 인과적 마케팅 attribution으로 단정하지
않는다. 인과 결론에는 실험 설계나 별도 causal method가 필요하다.

### 운영·리스크·VOC 분석

| 질문 유형 | 분석 목적 | 필요 데이터 |
| --- | --- | --- |
| `demand` | 수요 성장·변동성·계절성 분석 | demand metric, time |
| `inventory` | 재고 추세와 이상 재고 탐지 | inventory metric, time, features |
| `anomaly_detection` | 비정상 관측치 후보 탐색 | numeric features |
| `root_cause` | KPI 변화와 연관된 변수·그룹 탐색 | KPI metric, candidate drivers |
| `text` | VOC·리뷰의 빈도 기반 텍스트 프로파일 | text column |

### 현재 도구 수

현재 Analysis Agent에는 다음 23개 LangChain 도구가 등록되어 있다.

- 기술통계, 그룹 집계, 그룹 차이 검정, 상관 분석
- 단순 추세, 고급 시계열, 회귀, 분류, 이상탐지
- 코호트/리텐션, 퍼널, 고객 여정
- 고객 세그먼트, RFM, 기여도, mix-shift
- 생존분석, 텍스트 프로파일, 시나리오 분석
- Bayesian MMM, 확률적 CLV, 공간 hotspot, OR-Tools 배분 최적화

새 분석을 추가할 때는 `catalog.py`에 capability를 등록하고 `tools.py`의 계산 도구,
`insight.py`의 해석, `self_check.py`의 질문 유형 정합성, 유형별 테스트를 함께 추가한다.

### 그룹 비교

- 그룹이 2개이면 statsmodels `CompareMeans`의 Welch t-test와 평균 차이 신뢰구간을 사용한다.
- 그룹이 3개 이상이면 statsmodels의 Welch ANOVA를 사용한다.
- p-value뿐 아니라 Cohen's d 또는 eta-squared 효과크기를 제공한다.
- Levene 검정으로 등분산 관련 진단값도 기록한다.

### 상관 분석

- 각 수치형 컬럼 쌍의 Pearson 상관계수를 계산한다.
- 여러 컬럼 쌍을 동시에 검정할 때 Benjamini-Hochberg 방식으로 p-value를 보정한다.
- 상관관계를 인과관계로 표현하지 않는다.

### 예측과 분류

- 학습 데이터와 평가 데이터를 분리한다.
- 회귀는 고정 holdout으로 R2, MAE, RMSE를 계산한다.
- 분류는 stratified holdout으로 클래스 비율을 유지한다.
- 분류 결과에는 accuracy뿐 아니라 balanced accuracy와 weighted F1을 포함한다.
- 현재 모델은 최종 배포 모델이 아니라 비교 기준이 되는 baseline이다.

### 이상탐지

- 여러 수치형 feature를 이용해 Isolation Forest를 실행한다.
- 이상 후보의 행 위치와 anomaly score를 반환한다.
- 결과를 데이터 오류나 부정행위로 단정하지 않고 검토 후보로만 해석한다.

### 시계열

- 일·주·월·분기·연 단위로 metric을 집계한다.
- 전체 성장률, 평균 기간 성장률, 성장률 변동성, 추세 기울기와 p-value를 계산한다.
- 충분한 기간이 있고 계절 주기가 주어지면 계절성 강도를 계산한다.

### 코호트와 리텐션

- 최초 활동 시점을 cohort로 사용하거나 별도 cohort 컬럼을 받을 수 있다.
- 월 또는 주 단위 period index별 활성 entity와 retention rate를 계산한다.
- 결과는 cohort-retention matrix 형태로 반환한다.

### 퍼널과 고객 여정

- 퍼널은 지정된 단계 순서를 엄격하게 통과한 entity만 다음 단계 전환으로 인정한다.
- 각 단계의 시작 대비 전환율, 직전 단계 대비 전환율, drop-off를 계산한다.
- 고객 여정은 퍼널 단계를 미리 정하지 않고 주요 event path와 transition을 집계한다.

---

## 6. 구조화된 출력

최종 출력인 `AnalysisResult`의 주요 필드는 다음과 같다.

```json
{
  "run_id": "run_xxx",
  "goal": "카테고리별 매출 차이 분석",
  "plan": {
    "question_type": "comparison",
    "analysis_kind": "group_comparison",
    "analysis_subtype": "welch_t_test_or_one_way_anova",
    "tool_names": [
      "describe_metric",
      "compare_groups",
      "test_group_difference"
    ]
  },
  "method_summary": "...",
  "key_findings": ["..."],
  "evidence": [
    {
      "tool_name": "test_group_difference",
      "method": "welch_t_test",
      "statistics": {
        "p_value": 0.012,
        "effect_size_name": "cohen_d",
        "effect_size": 0.81
      },
      "finding": "...",
      "caveats": ["..."]
    }
  ],
  "hypotheses": ["..."],
  "limitations": ["..."],
  "source_artifacts": {
    "sql": ["artifact_sql_xxx"],
    "eda": ["artifact_eda_xxx"]
  },
  "human_review": {
    "required": false,
    "reason": ""
  }
}
```

기존 Visualization Agent와 Report Agent가 사용하던 다음 필드는 그대로 유지한다.

- `method_summary`
- `key_findings`
- `limitations`
- `source_artifacts`
- `data_quality_notes`

---

## 7. 아티팩트와 Lineage

Analysis 결과 JSON은 독립적인 분석 결과 아티팩트로 등록된다.

등록할 때 입력으로 사용한 SQL 및 EDA 아티팩트 ID를 `parent_ids`에 기록하고,
`lineage_edge_type="derived_from"`으로 연결한다.

```text
SQL Result Artifact ─┐
                     ├─ derived_from → Analysis Result Artifact
EDA Profile Artifact ┘
```

여기서 `parent_ids`는 파일 구조에서 Analysis 결과를 SQL·EDA의 하위 파일로 만든다는
뜻이 아니다. 어떤 근거에서 분석 결과가 파생됐는지 추적하기 위한 provenance 관계다.

이 lineage를 통해 다음을 확인할 수 있다.

- 어떤 SQL 결과를 분석했는가?
- 어떤 EDA 품질 정보를 참고했는가?
- 보고서 결과의 원천 데이터는 무엇인가?
- validation 실패 시 어느 입력부터 다시 확인해야 하는가?

---

## 8. Human-in-the-loop

Analysis Agent는 다음과 같은 경우 `human_review.required=true`를 반환할 수 있다.

- 관측 데이터만으로 인과관계 결론을 요구하는 경우
- target 또는 feature 정의가 모호한 경우
- 결과의 업무적 해석에 사람의 판단이 필요한 경우

Analysis Agent는 검토 필요 여부와 사유만 반환한다.

실제 동작은 supervisor가 담당한다.

1. 실행 중단
2. 승인 요청 저장
3. 사용자에게 approve/edit/reject 요청
4. 승인 또는 수정된 입력으로 실행 재개

---

## 9. LangChain 공식 문서 반영 내용

| 공식 문서 개념 | Analysis Agent 적용 방식 |
| --- | --- |
| Models | 공용 `get_chat_model`로 모델 provider 분리 |
| Messages | `SystemMessage`, `HumanMessage` 기반 planner 요청 |
| Tools | 타입이 정의된 LangChain 분석 도구 사용 |
| Structured Output | LLM 계획과 최종 결과를 Pydantic으로 검증 |
| Context Engineering | 전체 데이터 대신 제한된 샘플·컬럼 프로파일·EDA 신호 전달 |
| Guardrails | 허용 도구, 컬럼, 질문 유형과 실행 결과 검증 |
| Human-in-the-loop | 검토 필요 여부와 사유를 `AgentEnvelope.approval`로 전달 |
| LangGraph Workflow | `plan -> execute -> validate` custom workflow |

다음 기능은 Analysis Agent 내부가 아니라 상위 계층의 책임이다.

| 기능 | 담당 계층 |
| --- | --- |
| SQL·EDA·Analysis 간 multi-agent routing | Supervisor |
| short-term / long-term memory | Supervisor 및 공용 memory 계층 |
| 사용자 승인 중단·재개 | Supervisor |
| streaming 및 UI 진행 상태 | API/UI 계층 |
| 최종 결과 통합 | Report Agent |

`create_agent`의 자유로운 ReAct 반복 대신 custom LangGraph를 사용한 이유는 분석 실행이
`계획 → 계산 → 검증`이라는 명시적인 순서를 가져야 하기 때문이다. LLM이 임의로 계산
과정을 생략하거나 같은 도구를 무제한 반복하지 못하게 하고, 각 단계의 상태와 종료
조건을 추적할 수 있도록 했다.

---

## 10. 실패 정책

결정론적 planner fallback은 없다.

다음 상황에서는 Analysis Agent가 성공 결과를 임의로 만들어내지 않고 실패한다.

- LLM API 키 또는 모델 설정이 없음
- LLM 호출 실패
- LLM 구조화 출력이 스키마와 맞지 않음
- 존재하지 않는 도구 선택
- 존재하지 않는 컬럼 참조
- supervisor가 전달한 질문 유형을 LLM이 임의로 변경
- LLM이 계획한 도구가 실제 실행되지 않음
- 결과와 근거 간 추적성 검증 실패

통계 도구의 계산 오류는 `limitations`에 기록되고, 계획된 도구가 근거를 만들지 못했기
때문에 validation도 실패한다.

---

## 11. 현재 구현의 해석 범위

- 회귀와 분류는 baseline 모델이며 최종 운영 모델이 아니다.
- 한 번의 holdout 평가를 사용하므로 향후 cross-validation을 추가할 수 있다.
- 통계적 유의성과 업무적 중요성은 다르므로 p-value와 효과크기를 함께 확인해야 한다.
- 상관 분석만으로 인과관계를 결론 내리지 않는다.
- 이상탐지 결과는 검토 후보이며 실제 오류 여부는 사람이 확인해야 한다.

---

## 12. 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_analysis_agent.py -q
```

테스트에서는 외부 API를 호출하지 않고 구조화 계획을 반환하는 가짜 채팅 모델을
주입한다. production 실행에서는 모델을 주입하지 않기 때문에 공용 `get_chat_model`과
실제 LLM planner가 항상 호출된다.

고급 분석의 데이터 계약, 구현 순서와 설치된 전문 라이브러리는
[DAAA 고급 분석 Capability 확장 설계](./advanced_analysis_capabilities.md)를 참고한다.
