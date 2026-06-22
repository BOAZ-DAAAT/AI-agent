"""LLM 기반 인사이트 / 가설 / 핸드오프 요약 생성.

원래 main/eda_agent.py 의 insight_node / hypothesis_node 로직을 기능우선 구조에 맞게
analysis 에이전트로 이전한 것이다. 통계 수치 계산(statistical_metadata)과 프롬프트는
원본을 그대로 유지하고, 입력 소스만 EDA 산출물(main_eda_agent payload)로 바꿨다.

LLM 호출이 불가능한 환경(API 키 없음/네트워크 실패)에서는 규칙기반 분석 결과를 그대로
두고 인사이트만 폴백 문자열로 채운다 — 파이프라인을 절대 중단시키지 않는다.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.tools.missing import detect_missing
from DATA_Analyst_Assistant_Agent.agents.eda.tools.outlier import detect_outliers_iqr
from DATA_Analyst_Assistant_Agent.agents.eda.tools.quality import check_duplicates_fn
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model


def _run_with_retry(fn, *, fallback: str, max_retries: int = 2) -> tuple[str, str | None]:
    """fn 을 max_retries+1 회까지 시도하고, 모두 실패하면 (fallback, 에러메시지) 반환."""
    last_err: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - best-effort LLM call
            last_err = exc
    return fallback, f"{last_err}"


def build_statistical_metadata(
    df: pd.DataFrame | None,
    *,
    measure_cols: list[str] | None,
    key_col: str | None,
    clustering_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """downstream/프롬프트 주입용 검증 수치. 원본 insight_node 의 계산을 그대로 옮긴 것."""
    if df is None or df.empty:
        return {}

    clustering = clustering_result or {}
    numeric_cols = [
        c for c in (measure_cols or []) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        numeric_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)

    dist_stats: dict[str, Any] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        dist_stats[col] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "skewness": round(float(s.skew()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
        }

    corr_pairs: dict[str, float] = {}
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                key = f"corr_{numeric_cols[i]}_vs_{numeric_cols[j]}"
                corr_pairs[key] = round(float(corr.iloc[i, j]), 3)

    missing_info = detect_missing(df)
    outlier_info = detect_outliers_iqr(df, measure_cols=measure_cols)
    dup_info = check_duplicates_fn(df)
    outliers_by_col = {
        col: v.get("outlier_count", 0)
        for col, v in outlier_info.items()
        if isinstance(v, dict)
    }

    group_comparison: dict[str, Any] = {}
    if key_col and key_col in df.columns:
        for col in numeric_cols:
            try:
                grp = df.groupby(key_col)[col].mean().dropna()
                if grp.empty:
                    continue
                group_comparison[col] = {
                    "top3_groups": {str(k): round(float(v), 4) for k, v in grp.nlargest(3).items()},
                    "bottom3_groups": {str(k): round(float(v), 4) for k, v in grp.nsmallest(3).items()},
                    "group_max": round(float(grp.max()), 4),
                    "group_min": round(float(grp.min()), 4),
                    "group_std": round(float(grp.std()), 4),
                }
            except Exception:  # noqa: BLE001
                pass

    return {
        "row_count": len(df),
        "distribution": dist_stats,
        "group_comparison": group_comparison,
        "correlation_pairs": corr_pairs,
        "missing_total": missing_info.get("total_missing", 0),
        "outliers_by_column": outliers_by_col,
        "duplicate_count": dup_info.get("duplicate_count", 0),
        "clustering": {
            "n_clusters": clustering.get("n_clusters"),
            "silhouette_score": clustering.get("silhouette_score"),
            "cluster_centroids": clustering.get("cluster_centroids", {}),
        }
        if not clustering.get("skip")
        else {"skip": True},
    }


def _insight_prompt(user_question: str, statistical_metadata: dict, eda_main: dict) -> str:
    all_results = f"""
[품질 점검] {eda_main.get('quality_result', '해당 없음')}
[분포 분석] {eda_main.get('distribution_result', '해당 없음')}
[그룹 비교] {eda_main.get('comparison_result', '해당 없음')}
[관계 탐색] {eda_main.get('relationship_result', '해당 없음')}
[시간 분석] {eda_main.get('time_result', '해당 없음')}
[클러스터링] {json.dumps(eda_main.get('clustering_result', {}), ensure_ascii=False, default=str)}
"""
    return f"""
너는 EDA 종합 분석가다. 데이터를 읽는 것을 넘어, 이 데이터가 어떤 시장·비즈니스 구조를 보여주는지 해석하는 것이 네 역할이다.

[사용자 쿼리]
{user_question}

[검증된 수치 (구체적인 숫자를 쓸 때는 이 값을 우선 참고하라)]
{json.dumps(statistical_metadata, ensure_ascii=False, indent=2, default=str)}

[전체 EDA 결과 (패턴 해석의 주요 근거 — 분포/그룹비교/관계/시간 결과를 모두 활용하라)]
{all_results}

위 결과를 바탕으로 아래 두 섹션을 작성하라.
마크다운 기호(###, **, * 등)는 절대 사용하지 마라. 일반 텍스트로만 작성하라.

[핵심 패턴]
3~5가지를 번호 목록으로 작성하라.
각 항목은 반드시 아래 구조로 작성하라: 수치 근거 → 패턴 해석 → 해석 한계 또는 주의

작성 규칙:
- 검증된 수치의 clustering.skip이 False이고 n_clusters >= 2인 경우, 클러스터 결과를 번호 항목 중 하나로 반드시 포함하라.
  이때 각 클러스터를 centroid 수치 기반으로 직접 명명하고 (예: "배송지연·저만족 그룹", "고볼륨·균형 그룹"),
  각 그룹의 대표 항목을 cluster_labels에서 2~3개 직접 언급하라.
  클러스터 간 차이가 작으면 "해석에 주의가 필요하다"고 명시하고 비중을 줄여라.
- 클러스터 관련 내용을 별도 섹션으로 분리하거나 [핵심 패턴] 밖에 쓰지 마라. 반드시 번호 목록 안에 포함하라.

나쁜 예시 (쓰지 마라):
  "total_orders 평균 1314.43, 최소 2, 최대 9272" → 기술통계 복붙이지 패턴 해석이 아님
  "배송일을 줄이는 것이 중요한 요소다" → 단정적 결론 금지
  "클러스터링 결과, 네 개의 군집이 나뉜다" → 클러스터를 명명하지 않은 추상적 언급 금지

[구조 해석]
핵심 패턴들을 종합해 이 데이터가 어떤 시장·비즈니스 구조를 시사하는지 2~3문장으로 작성하라.
개별 수치를 나열하는 것이 아니라, 패턴들이 모여서 만드는 큰 그림을 해석하는 것이 목적이다.

아래 유형의 구조 언어를 참고하되, 데이터가 실제로 지지하는 경우에만 사용하라:
  - 집중형 구조: "소수 카테고리/고객에게 매출이 집중되는 헤드 집중형 구조를 보인다"
  - 분절형 구조: "성과 지표 간 뚜렷한 군집이 존재해, 수요 혹은 고객층이 분절되어 있을 가능성이 있다"
  - 편차형 구조: "카테고리별 충성도·재구매 편차가 크며, 이는 제품 특성보다 카테고리 고유의 구매 맥락 차이를 반영할 수 있다"
  - 트레이드오프 구조: "볼륨과 단위 수익성 간 음의 관계가 나타나, 규모와 마진 사이의 구조적 트레이드오프가 존재할 수 있다"

반드시 "~를 시사한다", "~일 가능성이 있다", "~로 해석될 수 있다" 같은 헤지 표현을 사용하라.
데이터에서 직접 관찰되지 않은 원인을 단정하지 마라.

[해석 주의사항]
- 결측치, 이상치, 표본 수 신뢰도 등 구체적 수치와 함께 작성
- 표본 수가 적어 신뢰도가 낮은 항목은 반드시 명시
- 한 줄로 간결하게

한국어로 작성하라.
"""


def _hypothesis_prompt(user_question: str, insight_result: str) -> str:
    return f"""
너는 데이터 분석 가설 설계자다. 네 가설은 다음 단계의 분석 에이전트(통계 검정, 모델링 수행)가 바로 실행할 수 있는 수준이어야 한다.

[사용자 쿼리]
{user_question}

[핵심 인사이트 및 구조 해석]
{insight_result}

위 인사이트를 바탕으로 아래 형식으로 작성하라.
마크다운 기호(###, **, * 등)는 절대 사용하지 마라. 일반 텍스트로만 작성하라.

[가설 1] ~ [가설 3] 형식으로 3개를 작성하라. 핵심 패턴에서 검증 가능한 것만 골라라.
각 가설은 아래 구조를 그대로 따르라. 섹션 레이블([가설 1] 등, 관찰:, H0:, H1:, 검증방법:, 필요변수:, 현재데이터:)은 반드시 그대로 유지하라.

[가설 1]
관찰: (이 가설의 근거가 된 수치나 패턴을 1문장으로. 인사이트에서 끌어와라.)
H0: (귀무가설 — A와 B 사이에 유의미한 관계가 없다)
H1: (대립가설 — IF [조건] THEN [결과]. 방향을 명시하라.)
검증방법: (구체적인 통계 검정명. 예: 단순선형회귀, 스피어만 상관검정, 일원배치 ANOVA 등)
필요변수: target=[변수명], feature=[변수명]
현재데이터: (현재 마트로 검증 가능 / 추가 필요: [무엇이 필요한지])

[가설 2]
관찰: ...
H0: ...
H1: ...
검증방법: ...
필요변수: ...
현재데이터: ...

[가설 3]
관찰: ...
H0: ...
H1: ...
검증방법: ...
필요변수: ...
현재데이터: ...

작성 규칙:
- 클러스터 레이블(cluster_group, cluster_id 등)을 feature로 사용하지 마라 → 순환논리. 클러스터에서 발견한 패턴을 원래 measure 변수로 재표현하라.
- 특정 항목 이름(카테고리명, 상품명 등)을 H1 조건에 직접 넣지 마라 → 관찰이지 가설이 아님. measure 변수 패턴으로 일반화하라.
- 현재 마트에 없는 변수를 필요변수로 적으면서 검증 가능하다고 쓰지 마라.

[다음 분석 방향]
위 3개 가설은 단순 이변량(A→B) 관계를 다룬다. 교란변수(confounding variable)가 결과를 왜곡할 수 있으므로 아래 2가지를 작성하라.
1. 교란변수 후보: 위 가설들의 결과에 영향을 미칠 수 있는 변수를 현재 마트에서 골라라. 없으면 추가 확보가 필요한 변수를 명시하라.
2. 통제 방법: 교란변수를 통제하기 위한 다음 분석 방법 제안 (예: 다중회귀로 확장, 층화 분석, 그룹별 하위분석 등)

한국어로 작성하라.
"""


def _final_summary_prompt(insight_result: str, hypotheses: str) -> str:
    return f"""
아래 EDA 인사이트와 가설을 다음 분석 에이전트에게 전달할 핸드오프 요약으로 압축하라.
마크다운 기호 사용 금지. 일반 텍스트로만 작성하라.

[인사이트]
{insight_result}

[가설]
{hypotheses}

작성 규칙:
- 4~6문장으로 압축
- 첫 문장: 데이터 구조의 핵심 특성 1가지 (수치 포함)
- 중간 문장: 현재 데이터로 바로 검증 가능한 가설을 우선 언급 (검증방법 포함)
- 마지막 문장: "다음 에이전트는 [검증방법]으로 [target]~[feature] 관계를 우선 검증하라"로 마무리
- 수치는 인사이트에서 확인된 것만 포함

한국어로 작성하라.
"""


def enrich_with_llm_insight(
    result: dict[str, Any],
    *,
    user_question: str,
    df: pd.DataFrame | None,
    measure_cols: list[str] | None,
    key_col: str | None,
    eda_main: dict[str, Any],
) -> dict[str, Any]:
    """analysis result 에 LLM 인사이트/가설/핸드오프 요약을 덧붙인다.

    LLM 사용 불가 시 규칙기반 result 를 그대로 두고 폴백 문자열만 채운다.
    insight 의 [핵심 패턴]은 key_findings 에 합류시켜 report 까지 자동으로 흐르게 한다.
    """
    statistical_metadata = build_statistical_metadata(
        df, measure_cols=measure_cols, key_col=key_col, clustering_result=eda_main.get("clustering_result")
    )
    result["statistical_metadata"] = statistical_metadata

    errors: list[str] = []
    try:
        llm = get_chat_model(temperature=0)
    except Exception as exc:  # noqa: BLE001 - API 키 없음 등: 결정론적 분석은 유지
        result["llm_insight"] = "인사이트 생성 생략 (LLM 사용 불가)"
        result["hypotheses"] = "가설 생성 생략 (LLM 사용 불가)"
        result["final_summary"] = "요약 생성 생략 (LLM 사용 불가)"
        result.setdefault("limitations", []).append(f"LLM 인사이트 단계 생략: {exc}")
        return result

    insight_result, err0 = _run_with_retry(
        lambda: llm.invoke(_insight_prompt(user_question, statistical_metadata, eda_main)).content.strip(),
        fallback="인사이트 생성 실패",
    )
    hypotheses, err1 = _run_with_retry(
        lambda: llm.invoke(_hypothesis_prompt(user_question, insight_result)).content.strip(),
        fallback="가설 생성 실패",
    )
    final_summary, err2 = _run_with_retry(
        lambda: llm.invoke(_final_summary_prompt(insight_result, hypotheses)).content.strip(),
        fallback="요약 생성 실패",
    )

    result["llm_insight"] = insight_result
    result["hypotheses"] = hypotheses
    result["final_summary"] = final_summary
    for err in (err0, err1, err2):
        if err:
            errors.append(err)
    if errors:
        result.setdefault("limitations", []).append("LLM 인사이트 단계 일부 실패: " + "; ".join(errors))

    # [핵심 패턴] 본문을 report 가 읽는 key_findings 로 합류
    if insight_result and insight_result != "인사이트 생성 실패":
        result.setdefault("key_findings", []).append("LLM 종합 해석: " + insight_result)

    return result
