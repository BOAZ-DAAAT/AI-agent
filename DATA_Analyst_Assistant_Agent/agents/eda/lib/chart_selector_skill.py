import os
import json
import base64

from langchain_core.messages import HumanMessage

from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model
import DATA_Analyst_Assistant_Agent.shared.config  # noqa: F401  (.env 로드 + DB_*/MYSQL_* 별칭 정규화)
from DATA_Analyst_Assistant_Agent.agents.eda.lib.chart_guards import drop_degenerate_charts

TOTAL_MAX = 10
WEAK_CORR_THRESHOLD = 0.2  # |r| 이 이 값 미만인 변수쌍의 scatter 는 정보가 없어 제거


def _load_llm():
    return get_chat_model(temperature=0)


def _load_chart_reader_llm():
    return get_chat_model(model_env="CHART_READER_MODEL", default_model="gpt-5")


# ─────────────────────────────
# 결정론 필터: 약한 상관 scatter 제거 (가드레일)
# ─────────────────────────────
def _corr_of_pair(pair: str, correlation_pairs: dict):
    """'A_vs_B' 형태에서 |상관계수| 를 찾는다. 없으면 None."""
    if not correlation_pairs:
        return None

    def _abs_r(v):
        if v is None:
            return None
        r = v.get("pearson_r") if isinstance(v, dict) else v   # 관계객체/실수 둘 다 지원
        return abs(r) if r is not None else None

    r = _abs_r(correlation_pairs.get(f"corr_{pair}"))
    if r is not None:
        return r
    if "_vs_" in pair:
        a, b = pair.split("_vs_", 1)
        return _abs_r(correlation_pairs.get(f"corr_{b}_vs_{a}"))
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


_PREFERRED_PREFIXES = ("segment_profile_", "interval_", "ecdf_")
_LIGHTWEIGHT_PREFIXES = ("dist_", "box_", "violin_")


def _selection_priority(path: str) -> tuple[int, str]:
    name = os.path.basename(path)
    if name.startswith("segment_profile_"):
        return (0, name)
    if name.startswith("interval_"):
        return (1, name)
    if name.startswith("ecdf_"):
        return (2, name)
    return (3, name)


def _promote_preferred_new_families(paths: list[str]) -> list[str]:
    return sorted(paths, key=_selection_priority)


def _ensure_preferred_survives(paths: list[str]) -> list[str]:
    if len(paths) <= TOTAL_MAX:
        return paths

    selected = list(paths[:TOTAL_MAX])
    selected_names = {os.path.basename(p) for p in selected}
    preferred = [p for p in paths if os.path.basename(p).startswith(_PREFERRED_PREFIXES)]
    if not preferred:
        return selected
    if any(os.path.basename(p) in selected_names for p in preferred):
        return selected

    candidate = preferred[0]
    for idx in range(len(selected) - 1, -1, -1):
        if os.path.basename(selected[idx]).startswith(_LIGHTWEIGHT_PREFIXES):
            selected[idx] = candidate
            return selected
    selected[-1] = candidate
    return selected


_VISUAL_CHECK_BATCH_SIZE = 4  # 배치당 이미지 수 — 단순 ok/issue 판정이라 여러 장 묶어도 정확도 저하가 적다
                              # (분석 에이전트의 자유서술 판독과 달리 판정 자체가 단순해서 배치에 유리, #194)

_VISUAL_CHECK_PROMPT = (
    "아래 차트들이 각각 렌더링 결함 없이 정상적으로 보이는지만 판정해줘. 데이터 해석이나 인사이트의 "
    "좋고나쁨은 판단하지 마 — 순수하게 시각적 결함만 봐:\n"
    "- 범례/텍스트 박스가 실제 데이터 점이나 선을 가리고 있는지\n"
    "- 축 라벨이 0/1 같은 원시값만 있어 무슨 그룹인지 알아볼 수 없는지\n"
    "- 그려져야 할 자리가 비어있거나(데이터가 있는데 안 그려짐) 점/선이 하나뿐인지\n"
    "각 차트 이미지 바로 앞에 그 파일명이 텍스트로 붙어 있다. 반드시 그 파일명을 키로 쓴 "
    "JSON 객체 하나만 출력해라(설명 금지):\n"
    '{"파일명1.png": {"ok": true 또는 false, "issue": "문제 설명(없으면 빈 문자열)"}, "파일명2.png": {...}}'
)


def _chunked(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _visual_sanity_check(paths: list[str]) -> tuple[list[str], list[dict], int]:
    """최종 선정된 차트(보통 TOTAL_MAX 이하)만 멀티모달로 훑어 렌더링 결함을 거른다.

    비용 통제: 전체 후보가 아니라 이미 좁혀진 최종 목록에만, _VISUAL_CHECK_BATCH_SIZE장씩
    묶어 호출한다(#194 — 1장당 1콜이던 걸 배치화, 최대 10장이면 3콜로 줄어듦).
    반환: (유지 경로, 드롭 메타 [{"chart","reason"}], 점검 자체가 실패한 횟수).
    check_failures를 dropped와 분리하는 이유 — 이미지 읽기/모델 호출/JSON 파싱이 실패해
    보수적으로 통과시킨 경우와, 모델이 실제로 "결함 있음"이라 판정해 드롭한 경우를
    구분 못 하면 "멀티모달 검사가 꺼졌는데 아무도 모르는" 상태를 감지할 수 없다.
    """
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        return paths, [], 0

    model = _load_chart_reader_llm()
    kept: list[str] = []
    dropped: list[dict] = []
    check_failures = 0

    for batch in _chunked(paths, _VISUAL_CHECK_BATCH_SIZE):
        content: list[dict] = [{"type": "text", "text": _VISUAL_CHECK_PROMPT}]
        batch_names: list[str] = []
        for p in batch:
            try:
                with open(p, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("ascii")
            except Exception:
                kept.append(p)
                check_failures += 1
                continue
            name = os.path.basename(p)
            batch_names.append(name)
            content.append({"type": "text", "text": f"[파일명: {name}]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})

        if not batch_names:
            continue  # 배치 전원이 파일 읽기 실패

        try:
            response = model.invoke([HumanMessage(content=content)])
            raw = str(getattr(response, "content", response)).replace("```json", "").replace("```", "").strip()
            verdicts = json.loads(raw)
            if not isinstance(verdicts, dict):
                raise ValueError("batch verdict is not a dict")
        except Exception:
            # 배치 콜 자체가 실패하면 배치 전체를 보수적으로 통과시킨다(단일 이미지 실패 처리와 동일 철학).
            for p in batch:
                if os.path.basename(p) in batch_names:
                    kept.append(p)
            check_failures += len(batch_names)
            continue

        for p in batch:
            name = os.path.basename(p)
            if name not in batch_names:
                continue  # 파일 읽기 실패로 이미 처리됨
            verdict = verdicts.get(name)
            if not isinstance(verdict, dict):
                kept.append(p)
                check_failures += 1
                continue
            if verdict.get("ok", True):
                kept.append(p)
            else:
                dropped.append({"chart": name, "reason": str(verdict.get("issue", "시각 결함"))})

    return kept, dropped, check_failures


def _backfill_after_visual_drop(
    final: list[str], filtered: list[str], target_count: int, exclude_names: set
) -> tuple[list[str], list[dict], int]:
    """멀티모달이 차트를 드롭해 자리가 비면, filtered의 다음 우선순위 후보로 채우고
    그 후보만 다시 시각 점검한다(이미 검사한 것 재검사 안 함 — 비용 통제).

    이게 없으면 _ensure_preferred_survives가 지켜준 segment_profile/interval/ecdf
    보호가 바로 다음 단계(시각 점검)에서 허무하게 깨질 수 있다.
    """
    if len(final) >= target_count:
        return final, [], 0
    final_names = {os.path.basename(p) for p in final}
    remaining = [p for p in filtered
                if os.path.basename(p) not in final_names and os.path.basename(p) not in exclude_names]
    remaining = sorted(remaining, key=_selection_priority)
    candidates = remaining[: target_count - len(final)]
    if not candidates:
        return final, [], 0
    backfill_kept, backfill_dropped, backfill_failures = _visual_sanity_check(candidates)
    return final + backfill_kept, backfill_dropped, backfill_failures


def _call_llm_remove(
    filenames: list,
    user_question: str,
    question_type: str,
    analysis_results: dict,
    statistical_metadata: dict,
    extra_instruction: str = "",
    priority_metrics: list = None,
    hypotheses: str = "",
) -> dict:
    """LLM에게 '이상적 차트 구성'을 그리게 하고, 거기 부합하지 않는 차트를 제거시킨다.

    유지 차트에는 선정 이유 캡션(keep_captions)을 함께 받는다 — key 차트는 아티팩트로 남아
    분석 에이전트가 멀티모달로 읽으므로 '무엇을 보여주는 차트인지'가 따라가야 한다(#71 B).
    """
    priority_info = ""
    if priority_metrics:
        names = ", ".join(m.get("metric", "") for m in priority_metrics if m.get("metric"))
        priority_info = f"\n[우선 지표]\n{names}\n"
    hypotheses_info = (f"\n[검증 예정 가설 — 각 가설의 근거가 되는 차트를 최소 1장 유지하라]\n{hypotheses}\n"
                       if hypotheses else "")

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
{priority_info}{hypotheses_info}
[분석 결과 요약]
{json.dumps(analysis_results, ensure_ascii=False, indent=2)}

[통계 메타데이터]
{json.dumps(statistical_metadata, ensure_ascii=False, indent=2)}

[전체 차트 후보]
{json.dumps(filenames, ensure_ascii=False)}

{extra_instruction}

── 이상적 차트 구성 가이드 (유지) ──
- **질문에 직답하는 차트가 최우선이다** — 예: 질문이 특정 target(재구매 여부 등)의 차이를 물으면
  distbytarget_*(target별 분포 비교)가 그 직답 차트다. 종합 차트보다 먼저 유지하라.
- 가설이 주어졌으면 각 가설의 근거 차트를 최소 1장 유지하라(가설-차트 대응).
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

반드시 아래 JSON만 출력하라. keep_captions에는 **남기는 모든 차트**에 대해
'이 차트가 무엇을 보여주고, 질문/어느 가설의 근거인지' 한 줄 캡션을 적어라
(이 캡션은 아티팩트 메타데이터로 남아 다음 에이전트가 차트를 읽을 때 참고한다).
{{
  "remove": ["파일명1.png", "파일명2.png"],
  "reason": {{
    "파일명1.png": "제거 이유"
  }},
  "keep_captions": {{
    "파일명3.png": "state별 재구매율 순위 — 가설1(지역별 차이)의 근거"
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
    hypotheses: str = "",
) -> tuple:
    """
    1단계: 기계적 품질 필터 (파일 존재 여부, 중복 경로 제거)
    1.5단계: 약한 상관 scatter 결정론 제거 (가드레일)
    2단계: LLM이 '이상적 구성'에 부합하지 않는 차트 제거 (+유지 차트 캡션 수집)
    3단계: TOTAL_MAX 초과 시 LLM이 추가 제거
    4단계: 최종 목록만 멀티모달로 렌더링 결함 점검 + 드롭된 자리 백필

    반환: (선별된 경로 리스트, {파일명: 선정 이유 캡션}, 시각점검 디버그 정보)
    시각점검 디버그 정보 = {"dropped": [{"chart","reason"}, ...], "check_failures": int}
    """
    if not chart_paths:
        return [], {}, {"dropped": [], "check_failures": 0}

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
        return [], {}, {"dropped": [], "check_failures": 0}

    # ── 1.5단계: 결정론 가드레일 (약한 상관 scatter + 퇴화 차트 제거) ──
    # _drop_weak_scatters 는 scatter_* 만, drop_degenerate_charts 는 그 외(catdist/dist/bar/heatmap)만
    # 건드려 대상이 겹치지 않는다. 순차 적용이라 이미 빠진 차트는 다음 가드가 보지 않음(중복 드롭 없음).
    valid_paths = _drop_weak_scatters(valid_paths, correlation_pairs)
    valid_paths, _ = drop_degenerate_charts(valid_paths, stat)  # 명백 퇴화(상수·단일범주·평탄) 제거
    if not valid_paths:
        return [], {}, {"dropped": [], "check_failures": 0}   # 결정론 가드로 전부 걸러졌으면 LLM 호출 없이 종료

    valid_paths = _promote_preferred_new_families(valid_paths)
    name_to_path = {os.path.basename(p): p for p in valid_paths}
    filenames = list(name_to_path.keys())

    # ── 2단계: LLM이 불필요 차트 제거 (+유지 캡션 수집) ──
    result = _call_llm_remove(
        filenames=filenames,
        user_question=user_question,
        question_type=question_type,
        analysis_results=analysis_results,
        statistical_metadata=stat,
        priority_metrics=priority_metrics,
        hypotheses=hypotheses,
    )

    to_remove = set(result.get("remove", []))
    captions = {k: str(v) for k, v in (result.get("keep_captions") or {}).items()}
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
                              f"질문 직답 차트(distbytarget 등)·가설 근거 차트·"
                              f"종합 비교 차트(heatmap_matrix/grouped_bar/radar)는 우선 보존하라.",
            priority_metrics=priority_metrics,
            hypotheses=hypotheses,
        )

        to_remove2 = set(result2.get("remove", []))
        captions.update({k: str(v) for k, v in (result2.get("keep_captions") or {}).items()})
        filtered = [p for p in filtered if os.path.basename(p) not in to_remove2]

    final = _ensure_preferred_survives(filtered)
    pre_visual_count = len(final)

    # ── 4단계: 최종 목록만 멀티모달로 렌더링 결함 사후 점검 + 드롭 시 백필 ──
    # 통계/이름 기준 판단(2단계)은 실제 픽셀을 못 본다 — 배지가 점을 가리는 식의 순수
    # 렌더링 버그는 여기서만 잡힌다. 이미 좁혀진 최종 목록(TOTAL_MAX 이하)에만 적용해 비용을 통제한다.
    # 드롭돼서 자리가 비면 _ensure_preferred_survives가 지켜준 보호가 여기서 허무하게 깨질 수
    # 있으므로, filtered의 다음 우선순위 후보로 채우고 그 후보만 다시 점검한다.
    final, visual_dropped, check_failures = _visual_sanity_check(final)
    if visual_dropped:
        exclude_names = {d["chart"] for d in visual_dropped} | {os.path.basename(p) for p in final}
        final, backfill_dropped, backfill_failures = _backfill_after_visual_drop(
            final, filtered, pre_visual_count, exclude_names)
        visual_dropped = visual_dropped + backfill_dropped
        check_failures += backfill_failures

    final_names = {os.path.basename(p) for p in final}
    visual_debug = {"dropped": visual_dropped, "check_failures": check_failures}
    return final, {k: v for k, v in captions.items() if k in final_names}, visual_debug
