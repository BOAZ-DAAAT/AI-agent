"""hypothesis 노드 + 핸드오프 요약 프롬프트."""

from __future__ import annotations

# ─────────────────────────────
# 가설 6유형 가이드 토글 (모듈 상수)
#   True  : target 성격 → 유형 판별표 + '유형:' 필드를 프롬프트에 주입(유형↔검증방법 정합 강제)
#   False : 기존 형식(관찰/H0/H1/검증방법/...)만 — 유형 가이드 없음
# ─────────────────────────────
HYPOTHESIS_TYPE_GUIDE = True


_TYPE_GUIDE_BLOCK = """
[가설 유형 판별 — 먼저 target 성격으로 유형을 정하고, 그 유형에 맞는 검증방법만 써라]
아래 표로 target 성격 → 유형 → 검증방법을 정한다. (괄호는 EDA 근거)
- 연속형 수치를 예측      → 회귀     → 단순/다중 선형회귀(또는 RandomForestRegressor)  (근거: correlation_pairs, distribution의 정규성·왜도, outliers)
- 범주/이진을 예측        → 분류     → 로지스틱 회귀·의사결정나무(Accuracy·AUC)         (근거: group_comparison의 범주별 비율, 클래스 불균형) ※주의: target 파생 필요(연속→이진화)가 흔함
- 두 변수 관계 유무·강도만 → 관계추론 → 피어슨/스피어만 상관검정(예측·계수 추정 아님)    (근거: correlation_pairs)
- 집단 간 평균·분포 비교   → 그룹차이 → 일원배치 ANOVA(2집단이면 t검정)+사후 Tukey      (근거: group_comparison의 top3/bottom3, group_std)
- 레이블 없는 구조 발견    → 군집     → K-means(+silhouette로 K 선택)                   (근거: clustering의 n_clusters·silhouette·centroids)
- 시간축 수치 추세·계절성  → 시계열   → 추세분해·자기상관(예측은 ARIMA·Prophet)         (근거: time_result의 추세·시즌성)
각 가설의 '유형:'에는 위 6가지(회귀/분류/관계추론/그룹차이/군집/시계열) 중 하나만 적고, '검증방법:'은 그 유형에 해당하는 방법만 써라.
"""


def hypothesis_prompt(user_question: str, insight_result: str,
                      include_type_guide: bool = HYPOTHESIS_TYPE_GUIDE,
                      target_hint: str = "", data_level: dict = None,
                      low_n_groups: list = None) -> str:
    type_field = "유형: (회귀/분류/관계추론/그룹차이/군집/시계열 중 하나 — 아래 판별표 기준)\n" if include_type_guide else ""
    type_guide = _TYPE_GUIDE_BLOCK if include_type_guide else ""
    labels_note = "관찰:, 유형:, H0:, H1:, 검증방법:, 필요변수:, 현재데이터:" if include_type_guide \
        else "관찰:, H0:, H1:, 검증방법:, 필요변수:, 현재데이터:"
    target_block = (
        f"\n[분석 대상(target) 후보] {target_hint}\n"
        "이 변수를 기본 target으로 두고 유형/검증방법을 정하라. "
        "인사이트상 더 적합한 target이 있으면 바꿔도 되지만, 근거를 관찰에 적어라.\n"
        if target_hint else ""
    )

    # 데이터 한계 가드 — 자가점검(data_level/표본)을 가설이 어기지 않게
    level = (data_level or {}).get("level")
    low_names = ", ".join(f"{g.get('group')}(n={g.get('n')})" for g in (low_n_groups or [])[:6])
    guard_lines = []
    if level == "aggregated":
        guard_lines.append(
            "이 데이터는 집계본(그룹당 1행)이다. ANOVA·t검정·개별 회귀처럼 '개별 관측'이 필요한 검정은 "
            "지금 데이터로 못 돌린다 → 그런 가설은 '현재데이터: 추가 필요(원본 행 필요)'로 솔직히 표기하라. "
            "집계 단위 방법(평균 간 상관·순위 비교)만 '검증 가능'으로 써라.")
        guard_lines.append(
            "집계 평균끼리의 상관을 개인 수준 결론으로 확대하지 마라(생태학적 오류). 관찰에 '카테고리 평균 기준'임을 명시하라.")
    if low_names:
        guard_lines.append(f"표본 30 미만 항목({low_names})을 가설의 관찰 근거로 쓰지 마라 — 신뢰도가 낮다.")
    guard_block = ("\n[데이터 한계 — 반드시 준수]\n" + "\n".join(f"- {x}" for x in guard_lines) + "\n"
                   if guard_lines else "")

    return f"""
너는 데이터 분석 가설 설계자다. 네 가설은 다음 단계의 분석 에이전트(통계 검정, 모델링 수행)가 바로 실행할 수 있는 수준이어야 한다.

[사용자 쿼리]
{user_question}

[핵심 인사이트 및 구조 해석]
{insight_result}
{target_block}{guard_block}{type_guide}
위 인사이트를 바탕으로 아래 형식으로 작성하라.
마크다운 기호(###, **, * 등)는 절대 사용하지 마라. 일반 텍스트로만 작성하라.

[가설 1], [가설 2] 형식으로 2~4개를 작성하라. 핵심 패턴에서 검증 가능한 것만 골라라.
데이터가 강하게 받쳐주는 것만 — 억지로 수를 채우지 말고, 근거가 약하면(표본 부족·자명한 관계·집계 한계) 빼라. 최소 2개, 최대 4개.
각 가설은 아래 구조를 그대로 따르라. 섹션 레이블([가설 1] 등, {labels_note})은 반드시 그대로 유지하라.

[가설 1]
관찰: (이 가설의 근거가 된 수치나 패턴을 1문장으로. 인사이트에서 끌어와라.)
{type_field}H0: (귀무가설 — A와 B 사이에 유의미한 관계가 없다)
H1: (대립가설 — IF [조건] THEN [결과]. 방향을 명시하라.)
검증방법: (구체적인 통계 검정명. 예: 단순선형회귀, 스피어만 상관검정, 일원배치 ANOVA 등)
필요변수: target=[변수명], feature=[변수명]
현재데이터: (현재 마트로 검증 가능 / 추가 필요: [무엇이 필요한지])

[가설 2]
관찰: ...
{type_field}H0: ...
H1: ...
검증방법: ...
필요변수: ...
현재데이터: ...

([가설 3], [가설 4]는 데이터가 강하게 받쳐줄 때만 위와 같은 형식으로 추가하라. 약하거나 자명하면 추가하지 마라.)

작성 규칙:
- 클러스터 레이블(cluster_group, cluster_id 등)을 feature로 사용하지 마라 → 순환논리. 클러스터에서 발견한 패턴을 원래 measure 변수로 재표현하라.
- 특정 항목 이름(카테고리명, 상품명 등)을 H1 조건에 직접 넣지 마라 → 관찰이지 가설이 아님. measure 변수 패턴으로 일반화하라.
- 현재 마트에 없는 변수를 필요변수로 적으면서 검증 가능하다고 쓰지 마라.
- 한 변수가 다른 변수의 구성요소면(예: 매출=가격×수량, 합계=항목들의 합) 그 상관은 자명하니 가설로 쓰지 마라. 비자명한 관계만 가설로.

[다음 분석 방향]
위 가설들은 단순 이변량(A→B) 관계를 다룬다. 교란변수(confounding variable)가 결과를 왜곡할 수 있으므로 아래 2가지를 작성하라.
1. 교란변수 후보: 위 가설들의 결과에 영향을 미칠 수 있는 변수를 현재 마트에서 골라라. 없으면 추가 확보가 필요한 변수를 명시하라.
2. 통제 방법: 교란변수를 통제하기 위한 다음 분석 방법 제안 (예: 다중회귀로 확장, 층화 분석, 그룹별 하위분석 등)

한국어로 작성하라.
"""


def handoff_summary_prompt(insight_result: str, hypotheses: str) -> str:
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
