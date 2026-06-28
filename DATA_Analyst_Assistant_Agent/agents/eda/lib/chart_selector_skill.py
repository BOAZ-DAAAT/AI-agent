import os
import json

from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
import DATA_Analyst_Assistant_Agent.shared.config  # noqa: F401  (.env 로드 + DB_*/MYSQL_* 별칭 정규화)

TOTAL_MAX = 8
WEAK_CORR_THRESHOLD = 0.2  # |r| 이 이 값 미만인 변수쌍의 scatter 는 정보가 없어 제거


def _load_llm():
    return get_chat_model(temperature=0)


# ─────────────────────────────
# 결정론 필터: 약한 상관 scatter 제거 (가드레일)
# ─────────────────────────────
def _corr_of_pair(pair: str, correlation_pairs: dict):
    """'A_vs_B' 형태에서 |상관계수| 를 찾는다. 없으면 None."""
    if not correlation_pairs:
        return None
    direct = correlation_pairs.get(f"corr_{pair}")
    if direct is not None:
        return abs(direct)
    if "_vs_" in pair:
        a, b = pair.split("_vs_", 1)
        rev = correlation_pairs.get(f"corr_{b}_vs_{a}")
        if rev is not None:
            return abs(rev)
    return None


def _drop_weak_scatters(paths: list, correlation_pairs: dict) -> list:
    """scatter_A_vs_B.png 중 |r| < 임계값인 것을 제거 (cluster_scatter 는 제외)."""
    kept = []
    for p in paths:
        name = os.path.basename(p)
        if name.startswith("scatter_") and name.endswith(".png"):
            pair = name[len("scatter_"):-len(".png")]
            r = _corr_of_pair(pair, correlation_pairs)
            if r is not None and r < WEAK_CORR_THRESHOLD:
                continue  # 약한 상관 → 의미 없는 산점도, 제거
        kept.append(p)
    return kept


def _call_llm_remove(
    filenames: list,
    user_question: str,
    question_type: str,
    analysis_results: dict,
    statistical_metadata: dict,
    extra_instruction: str = "",
    priority_metrics: list = None,
) -> dict:
    """LLM에게 '이상적 차트 구성'을 그리게 하고, 거기 부합하지 않는 차트를 제거시킨다."""
    priority_info = ""
    if priority_metrics:
        names = ", ".join(m.get("metric", "") for m in priority_metrics if m.get("metric"))
        priority_info = f"\n[우선 지표]\n{names}\n"

    # 클러스터링 품질 평가 — 실루엣 점수 기반 판단 지침 생성
    clustering = statistical_metadata.get("clustering", {})
    cluster_chart_rule = ""
    if clustering.get("skip"):
        cluster_chart_rule = (
            "\n클러스터링 차트 처리 기준:\n"
            "clustering이 실행되지 않았다. cluster_ 관련 차트는 제거하라.\n"
        )
    else:
        sil = clustering.get("silhouette_score", 0)
        n_k = clustering.get("n_clusters", 0)
        if sil >= 0.5:
            quality = f"실루엣 점수 {sil} (0.5 이상 — 클러스터 구분이 뚜렷함)"
            guidance = "cluster_profile, cluster_scatter 차트는 다차원 그룹 구조를 명확히 보여주므로 유지하라."
        elif sil >= 0.25:
            quality = f"실루엣 점수 {sil} (0.25~0.5 — 클러스터 구분이 보통)"
            guidance = "cluster_profile 차트는 유지하되, cluster_scatter는 다른 차트와 중복 여부를 판단해 결정하라."
        else:
            quality = f"실루엣 점수 {sil} (0.25 미만 — 클러스터 구분이 약함)"
            guidance = "cluster_ 차트의 해석 가치가 낮으므로 다른 차트보다 낮은 우선순위로 처리하라."
        cluster_chart_rule = (
            f"\n클러스터링 차트 처리 기준:\n"
            f"clustering 결과: n_clusters={n_k}, {quality}\n"
            f"{guidance}\n"
        )

    prompt = f"""
너는 데이터 분석 보고서의 차트 큐레이터다.

먼저 '이 질문에 이상적으로 어떤 차트들이 있어야 하는가'를 머릿속에 그려라.
그 다음 후보 차트 중 그 이상적 구성에 부합하는 것만 남기고 나머지를 제거하라.

[사용자 질문]
{user_question}

[question_type]
{question_type}
{priority_info}
[분석 결과 요약]
{json.dumps(analysis_results, ensure_ascii=False, indent=2)}

[통계 메타데이터]
{json.dumps(statistical_metadata, ensure_ascii=False, indent=2)}

[전체 차트 후보]
{json.dumps(filenames, ensure_ascii=False)}

{extra_instruction}

── 이상적 차트 구성 가이드 (유지) ──
- 비교/순위/"성과 좋은 ~" 류 질문이면: 여러 지표를 종합 비교하는 차트(heatmap_matrix, grouped_bar, radar)를
  핵심으로 반드시 1~2개 유지하라. 이게 '어느 것이 종합적으로 우수한가'에 직접 답하는 차트다.
- 각 핵심 지표의 순위를 보여주는 bar_top(낮을수록 좋은 지표는 bar_bottom)을 유지하라.
- 3개 이상 지표를 한 장에 보여주는 bubble은 유지 가치가 높다.

── 제거 대상 ──
1. 사용자 질문과 무관한 차트
2. 같은 정보를 반복하는 차트 (동일 지표의 dist/box/violin 중 정보량 적은 것 등)
3. 변수 간 '관계'가 질문의 핵심이 아닌데 들어있는 scatter (단순 비교 질문에서 scatter는 부차적)
4. heatmap_matrix(카테고리×지표)와 correlation_heatmap(지표×지표)이 둘 다 있으면:
   - 비교/성과/순위 질문 → heatmap_matrix 유지, correlation_heatmap 제거 (카테고리 성과를 직접 보여줌)
   - 변수 간 상관관계가 핵심 질문 → correlation_heatmap 유지
5. 해석 가치가 낮거나 보고서에서 설명하기 어려운 차트
{cluster_chart_rule}

최대 {TOTAL_MAX}개 이하로 남겨라.

반드시 아래 JSON만 출력하라.
{{
  "remove": ["파일명1.png", "파일명2.png"],
  "reason": {{
    "파일명1.png": "제거 이유"
  }}
}}
"""
    llm = _load_llm()
    response = llm.invoke(prompt).content.strip()
    response = response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(response)
    except Exception:
        return {"remove": [], "reason": {}}


def run_chart_selector_skill(
    chart_paths: list,
    user_question: str,
    analysis_results: dict,
    question_type: str = "",
    statistical_metadata: dict = None,
    priority_metrics: list = None,
) -> list:
    """
    1단계: 기계적 품질 필터 (파일 존재 여부, 중복 경로 제거)
    1.5단계: 약한 상관 scatter 결정론 제거 (가드레일)
    2단계: LLM이 '이상적 구성'에 부합하지 않는 차트 제거
    3단계: 8개 초과 시 LLM이 추가 제거
    """
    if not chart_paths:
        return []

    stat = statistical_metadata or {}
    correlation_pairs = stat.get("correlation_pairs", {})

    # ── 1단계: 기계적 품질 필터 ──
    seen_paths = set()
    valid_paths = []
    for p in chart_paths:
        abs_p = os.path.abspath(p)
        if abs_p in seen_paths:
            continue
        if not os.path.exists(p):
            continue
        seen_paths.add(abs_p)
        valid_paths.append(p)

    if not valid_paths:
        return []

    # ── 1.5단계: 약한 상관 scatter 결정론 제거 ──
    valid_paths = _drop_weak_scatters(valid_paths, correlation_pairs)

    name_to_path = {os.path.basename(p): p for p in valid_paths}
    filenames = list(name_to_path.keys())

    # ── 2단계: LLM이 불필요 차트 제거 ──
    result = _call_llm_remove(
        filenames=filenames,
        user_question=user_question,
        question_type=question_type,
        analysis_results=analysis_results,
        statistical_metadata=stat,
        priority_metrics=priority_metrics,
    )

    to_remove = set(result.get("remove", []))
    filtered = [p for p in valid_paths if os.path.basename(p) not in to_remove]

    # ── 3단계: 8개 초과 시 LLM이 추가 제거 ──
    if len(filtered) > TOTAL_MAX:
        excess = len(filtered) - TOTAL_MAX
        filtered_names = [os.path.basename(p) for p in filtered]

        result2 = _call_llm_remove(
            filenames=filtered_names,
            user_question=user_question,
            question_type=question_type,
            analysis_results=analysis_results,
            statistical_metadata=stat,
            extra_instruction=f"현재 차트가 {len(filtered)}개로 {TOTAL_MAX}개를 초과한다. "
                              f"가장 중복되거나 임팩트가 낮은 {excess}개를 추가로 제거하되, "
                              f"종합 비교 차트(heatmap_matrix/grouped_bar/radar)는 우선 보존하라.",
            priority_metrics=priority_metrics,
        )

        to_remove2 = set(result2.get("remove", []))
        filtered = [p for p in filtered if os.path.basename(p) not in to_remove2]

    return filtered[:TOTAL_MAX]
