"""가설 사후 재검증 (LLM 없음, 순수 코드).

hypothesis 노드가 LLM으로 만든 가설 텍스트를, 이미 계산된 통계 숫자
(statistical_metadata 의 correlation_pairs·clustering)로 사후 재검증한다.
LLM 은 창의적 제안을 계속하되, 여기서 근거 숫자를 대조해:
  1) 각 가설에 사전신호 강도를 태그하고(강함/중간/약함/미측정)
  2) 강한 순으로 재정렬하며
  3) 명백 무상관(관계추론/회귀인데 pearson·spearman 둘 다 |r|<0.05)만 드롭한다(최소 1개 보존).

보수적 정책(정책 A) — 신뢰 가능한 신호만 쓴다:
  - 관계추론/회귀 → correlation_pairs 의 max(|pearson|,|spearman|)
  - 군집        → clustering.silhouette_score (skip 이면 미측정)
  - 그룹차이/분류/시계열 → 집계본에서 신호가 불안정(group_comparison 이 엔티티키 기준) →
    '미측정'으로 두고 건드리지 않는다(태그만, 드롭·재정렬 영향 최소).

'미측정' = "사전 채점에 맞는 숫자가 없어 강도를 미리 못 쟀다"는 뜻(가설이 나쁘다는 뜻 아님).
하류 분석 에이전트가 '미측정=버려라'로 오해하지 않도록, 태그에 "분석에서 직접 검정"을 명시한다.

기존 패턴(reliability.correct_hypothesis_feasibility 의 블록 파싱,
chart_selector_skill._drop_weak_scatters 의 상관 조회)을 가설 강도로 확장한 것.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# 강도 임계값 (관계추론/회귀는 |r|, 군집은 silhouette)
STRONG_ABS_R = 0.4
MEDIUM_ABS_R = 0.2
STRONG_SILHOUETTE = 0.5
MEDIUM_SILHOUETTE = 0.25
# 명백 무상관 드롭 기준: pearson·spearman 둘 다 이 값 미만일 때만 드롭
NULL_ABS_R = 0.05

# 정규 유형 13종 (기존 6 + analysis_agent가 codegen/vetted primitive로 커버하는 5개 특화기법 +
# 코호트/퍼널 — EDA는 '제안'만 하고 최종 방법 선택은 analysis_agent가 다시 하므로, 유형 폭을
# analysis_agent 커버리지에 맞춰 넓혀둔다).
_TYPES = ("회귀", "분류", "관계추론", "그룹차이", "군집", "시계열",
          "생애가치", "생존분석", "지역분석", "마케팅믹스", "자원배분", "코호트", "퍼널")

# 유형 라인이 없을 때(HYPOTHESIS_TYPE_GUIDE=False) 검증방법 텍스트로 유형 추론
_METHOD_TYPE_HINTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("관계추론", ("상관", "피어슨", "스피어만", "correlation", "pearson", "spearman")),
    ("군집", ("군집", "클러스터", "kmeans", "k-means", "silhouette", "실루엣")),
    ("시계열", ("시계열", "arima", "prophet", "추세", "계절", "자기상관", "autocorrelation")),
    ("그룹차이", ("anova", "분산분석", "t검정", "t-검정", "ttest", "t test", "tukey", "튜키",
                "mann-whitney", "mann whitney", "만-휘트니", "만휘트니", "kruskal", "크루스칼",
                "wilcoxon", "윌콕슨", "카이제곱", "chi-square", "chi square", "chi2")),
    ("분류", ("분류", "로지스틱", "logistic", "의사결정", "decision tree", "randomforestclassifier", "auc")),
    ("회귀", ("회귀", "regression", "ols", "randomforestregressor")),
    ("생애가치", ("ltv", "생애가치", "clv", "bg/nbd", "gamma-gamma", "감마-감마")),
    ("생존분석", ("생존", "survival", "kaplan", "카플란", "cox", "콕스")),
    ("지역분석", ("지역", "공간", "geospatial", "moran", "모란", "hotspot", "핫스팟")),
    ("마케팅믹스", ("mmm", "마케팅믹스", "adstock", "애드스톡", "saturation")),
    ("자원배분", ("배분", "최적화", "optimization", "선형계획", "linear program")),
    ("코호트", ("코호트", "cohort", "유지율", "리텐션", "retention")),
    ("퍼널", ("퍼널", "funnel", "전환율", "이탈률", "conversion")),
]

# 신호로 검증 가능한 유형(나머지는 미측정으로 둠)
_VERIFIABLE_TYPES = {"관계추론", "회귀", "군집"}

# 재정렬 우선순위: 강함 > 중간 > 미측정(신호없음=중립) > 약함(신호있는데 약함=후순위)
#   약함을 미측정보다 뒤로 두는 이유: 약한 신호는 '재미없다'는 증거지만, 미측정(군집 skip 등)은
#   '아직 모름'이라 적극 강등하면 실제 답(예: 군집 세그먼트)을 묻을 수 있어 중립에 둔다.
_STRENGTH_ORDER = {"강함": 0, "중간": 1, "미측정": 2, "약함": 3}


def _norm(name: str) -> str:
    """비교용 정규화(소문자 + 영숫자만)."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _columns_from_corr(correlation_pairs: Dict[str, Any]) -> List[str]:
    """correlation_pairs 키('corr_{a}_vs_{b}')에서 등장한 컬럼명을 모은다(퍼지 매칭용)."""
    cols: set = set()
    for key in correlation_pairs or {}:
        body = key[len("corr_"):] if key.startswith("corr_") else key
        if "_vs_" in body:
            a, b = body.split("_vs_", 1)
            cols.add(a)
            cols.add(b)
    return list(cols)


def _resolve_name(name: Optional[str], cols: List[str]) -> Optional[str]:
    """가설이 줄여 쓴 변수명을 실제 컬럼명으로 퍼지 매칭(correct_hypothesis_feasibility 방식)."""
    if not name:
        return None
    if name in cols:
        return name
    nl = _norm(name)
    if not nl:
        return None
    for c in cols:
        cl = _norm(c)
        if cl and (nl in cl or cl in nl):
            return c
    return None


def _lookup_corr(correlation_pairs: Dict[str, Any], a: str, b: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(a,b) 쌍의 상관 객체를 두 방향 키로 조회. (객체, 매칭키) 반환."""
    for key in (f"corr_{a}_vs_{b}", f"corr_{b}_vs_{a}"):
        v = (correlation_pairs or {}).get(key)
        if isinstance(v, dict):
            return v, key
    return None, None


def _abs(v: Any) -> float:
    try:
        return abs(float(v))
    except (TypeError, ValueError):
        return 0.0


# ── 블록 분리 ────────────────────────────────────────────────────────────────
def _split_blocks(text: str) -> Tuple[str, List[str], str]:
    """텍스트를 (프리앰블, [가설 블록...], 트레일링)으로 분리한다.

    트레일링은 첫 가설 이후 등장하는 '[가설'이 아닌 대괄호 헤더([다음 분석 방향] 등) 또는
    '다음 분석 방향' 헤더(LLM이 대괄호를 빼먹는 경우)부터 끝까지. 이 섹션은 특정 가설에 속하지
    않는 전역 내용이므로, 마지막 가설 블록에 흡수돼 재정렬 때 딸려 나가면 안 된다(그대로 보존).
    """
    lines = text.split("\n")
    starts = [i for i, ln in enumerate(lines) if ln.strip().startswith("[가설")]
    if not starts:
        return text, [], ""

    trailing_idx: Optional[int] = None
    for i in range(starts[0] + 1, len(lines)):
        s = lines[i].strip()
        s_core = s.lstrip("[").strip()  # 대괄호 유무와 무관하게 헤더 텍스트 비교
        if (s.startswith("[") and not s.startswith("[가설")) or s_core.startswith("다음 분석 방향"):
            trailing_idx = i
            break

    end = trailing_idx if trailing_idx is not None else len(lines)
    preamble = "\n".join(lines[:starts[0]])
    bounds = [s for s in starts if s < end] + [end]
    blocks = ["\n".join(lines[bounds[k]:bounds[k + 1]]) for k in range(len(bounds) - 1)]
    trailing = "\n".join(lines[end:]) if trailing_idx is not None else ""
    return preamble, blocks, trailing


def _parse_block(block_text: str) -> Dict[str, Any]:
    """블록에서 유형·target·feature·검증방법을 뽑는다."""
    info: Dict[str, Any] = {"type": "", "target": None, "features": [], "method": ""}
    for ln in block_text.split("\n"):
        s = ln.strip()
        if s.startswith("유형:"):
            body = s.split(":", 1)[1]
            for t in _TYPES:
                if t in body:
                    info["type"] = t
                    break
        elif s.startswith("검증방법:"):
            info["method"] = s.split(":", 1)[1].strip()
        elif s.startswith("필요변수:"):
            tm = re.search(r"target\s*=\s*\[?\s*([A-Za-z0-9_]+)", s)
            if tm:
                info["target"] = tm.group(1)
            if "feature" in s:
                fpart = s.split("feature", 1)[1]
                fm = re.search(r"\[([^\]]*)\]", fpart)
                raw = fm.group(1) if fm else fpart.lstrip("=[ ")
                info["features"] = [x.strip() for x in re.split(r"[,\s]+", raw) if x.strip()]
    if not info["type"]:
        ml = info["method"].lower()
        for t, hints in _METHOD_TYPE_HINTS:
            if any(h in ml for h in hints):
                info["type"] = t
                break
    return info


def _classify(info: Dict[str, Any], correlation_pairs: Dict[str, Any],
              clustering: Dict[str, Any], cols: List[str]) -> Dict[str, Any]:
    """유형별 신호를 조회해 강도/드롭후보를 판정한다."""
    typ = info["type"] or "미상"

    if typ in ("관계추론", "회귀"):
        target = _resolve_name(info["target"], cols)
        best: Optional[Tuple[float, Dict[str, Any], str, str]] = None
        for feat in info["features"]:
            f = _resolve_name(feat, cols)
            if not (target and f) or target == f:
                continue
            entry, key = _lookup_corr(correlation_pairs, target, f)
            if entry:
                ar = max(_abs(entry.get("pearson_r")), _abs(entry.get("spearman_r")))
                if best is None or ar > best[0]:
                    best = (ar, entry, f, key)
        if best is None:
            return {"type": typ, "strength": "미측정", "matched_signal": None, "score_value": None,
                    "reason": "상관쌍 신호 없음(컬럼 매칭 실패)"}
        ar, entry, f, key = best
        pear, spear = _abs(entry.get("pearson_r")), _abs(entry.get("spearman_r"))
        strength = "강함" if ar >= STRONG_ABS_R else "중간" if ar >= MEDIUM_ABS_R else "약함"
        is_null = (pear < NULL_ABS_R and spear < NULL_ABS_R)
        return {
            "type": typ, "strength": strength, "feature": f,
            "matched_signal": key, "score_value": round(ar, 3),
            "abs_r": round(ar, 3), "pearson_r": entry.get("pearson_r"),
            "spearman_r": entry.get("spearman_r"),
            "drop_candidate": is_null,
            "reason": "명백 무상관(pearson·spearman 둘 다 |r|<0.05)" if is_null else "",
        }

    if typ == "군집":
        if not clustering or clustering.get("skip"):
            return {"type": typ, "strength": "미측정", "matched_signal": None, "score_value": None,
                    "reason": "군집 미실행/skip"}
        sil = clustering.get("silhouette_score")
        try:
            s = float(sil)  # raw 값 사용(음수 = 나쁜 군집 → abs 쓰면 안 됨)
        except (TypeError, ValueError):
            return {"type": typ, "strength": "미측정", "matched_signal": None, "score_value": None,
                    "reason": "silhouette 없음/파싱 실패"}
        strength = "강함" if s >= STRONG_SILHOUETTE else "중간" if s >= MEDIUM_SILHOUETTE else "약함"
        return {"type": typ, "strength": strength, "silhouette": sil, "drop_candidate": False,
                "matched_signal": "clustering.silhouette_score", "score_value": s}

    # 그룹차이/분류/시계열/미상 → 집계본에서 신호 불안정 → 미측정(안 건드림)
    return {"type": typ, "strength": "미측정", "matched_signal": None, "score_value": None,
            "reason": "집계본 신호 불안정 또는 유형 미상 → 재검증 보류"}


def _tag_line(cls: Dict[str, Any]) -> str:
    """블록에 삽입할 '사전신호:' 라인.

    '미측정'은 '버려라'가 아니라 '사전 채점 못 함 → 분석에서 직접 검정'이라는 뜻이므로
    그 안내를 라벨에 명시한다(하류 분석 에이전트의 오해 방지).
    """
    if "abs_r" in cls:
        detail = f"|r|={cls['abs_r']}, pearson={cls.get('pearson_r')}, spearman={cls.get('spearman_r')}"
    elif "silhouette" in cls:
        detail = f"silhouette={cls['silhouette']}"
    else:
        detail = cls.get("reason", "")
    if cls["strength"] == "미측정":
        base = "사전신호: 미측정 — 분석에서 직접 검정"
    else:
        base = f"사전신호: {cls['strength']}"
    return base + (f" ({detail})" if detail else "")


def _apply_tag(block_text: str, tag_line: str) -> str:
    """'관찰:' 라인 바로 뒤(없으면 헤더 바로 뒤)에 사전신호 라인을 삽입."""
    lines = block_text.split("\n")
    # 이미 태그가 있으면(재실행) 교체
    lines = [ln for ln in lines if not ln.strip().startswith("사전신호:")]
    out: List[str] = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if not inserted and ln.strip().startswith("관찰:"):
            out.append(tag_line)
            inserted = True
    if not inserted:
        # 관찰 라인이 없으면 헤더([가설 N]) 바로 뒤에 삽입
        insert_at = 1 if out and out[0].strip().startswith("[가설") else 0
        out.insert(insert_at, tag_line)
    return "\n".join(out)


def screen_hypotheses(hypotheses_text: str,
                      statistical_metadata: Optional[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """가설 텍스트를 통계 숫자로 사후 재검증 → (재작성 텍스트, 신호 메타 리스트).

    - 각 가설에 '사전신호:' 강도 태그 삽입
    - 명백 무상관(관계추론/회귀, 둘 다 |r|<0.05)만 드롭(최소 1개 보존)
    - 강한 순으로 재정렬(같은 강도는 원래 순서 유지)
    파싱 실패/블록 없음/신호 없음이면 원문을 최대한 보존한다(보수적).
    """
    if not hypotheses_text or not hypotheses_text.strip():
        return hypotheses_text, []

    stat = statistical_metadata or {}
    correlation_pairs = stat.get("correlation_pairs", {}) or {}
    clustering = stat.get("clustering", {}) or {}
    cols = _columns_from_corr(correlation_pairs)

    preamble, blocks, trailing = _split_blocks(hypotheses_text)
    if not blocks:
        return hypotheses_text, []

    entries: List[Dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        info = _parse_block(block)
        cls = _classify(info, correlation_pairs, clustering, cols)
        tagged = _apply_tag(block, _tag_line(cls))
        entries.append({
            "orig_index": idx,
            "type": cls.get("type"),
            "target": info.get("target"),
            "features": info.get("features"),
            "strength": cls["strength"],
            "matched_signal": cls.get("matched_signal"),  # 어떤 통계로 판정했나(디버깅)
            "score_value": cls.get("score_value"),        # 판정에 쓴 수치(|r| 또는 silhouette)
            "abs_r": cls.get("abs_r"),
            "silhouette": cls.get("silhouette"),
            "drop_candidate": bool(cls.get("drop_candidate")),
            "reason": cls.get("reason", ""),
            "drop_reason": None,
            "_block": tagged,
        })

    # 드롭: 명백 무상관 후보만. 최소 1개 보존(전부 드롭되면 드롭 취소).
    non_drop = [e for e in entries if not e["drop_candidate"]]
    if not non_drop:
        # 전부 드롭 후보 → 정보 손실 방지 위해 드롭 취소(원문 유지)
        for e in entries:
            e["dropped"] = False
    else:
        for e in entries:
            e["dropped"] = bool(e["drop_candidate"])
            if e["dropped"]:
                e["drop_reason"] = e.get("reason") or "명백 무상관(pearson·spearman 둘 다 |r|<0.05)"

    kept = [e for e in entries if not e["dropped"]]

    # 재정렬: 강한 순(안정 정렬 → 같은 강도는 원래 순서).
    kept_sorted = sorted(kept, key=lambda e: (_STRENGTH_ORDER.get(e["strength"], 3), e["orig_index"]))

    parts: List[str] = []
    if preamble.strip():
        parts.append(preamble.rstrip("\n"))
    parts.extend(e["_block"].strip("\n") for e in kept_sorted)
    new_text = "\n\n".join(p for p in parts if p)
    if trailing.strip():
        new_text = new_text + "\n\n" + trailing.strip("\n")

    signals_meta = [{k: v for k, v in e.items() if k != "_block"} for e in entries]
    return new_text, signals_meta
