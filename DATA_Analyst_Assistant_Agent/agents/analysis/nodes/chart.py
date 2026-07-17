from __future__ import annotations

import base64
import json
import os
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model


ChartArtifactLoader = Callable[[str], bytes]
# 배치 시그니처(#194) — chart_images 전체를 한 번에 받아 같은 순서의 결과 리스트를 반환한다.
# (이전엔 차트 1개당 1번 호출하는 (dict, bytes, state) -> dict|str 시그니처였음)
ChartReader = Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any] | str]]


def decide_chart_inspection(state: dict[str, Any], *, max_charts: int = 3) -> dict[str, Any]:
    profiles = state.get("eda_profiles", []) or []
    chart_requests = _build_chart_requests(profiles)
    selected_charts = _select_available_charts(profiles, chart_requests, max_charts=max_charts)

    if not chart_requests:
        status = "not_needed"
    elif selected_charts:
        status = "needed_available"
    else:
        status = "needed_unavailable"

    return {
        "chart_requests": chart_requests,
        "selected_charts": selected_charts,
        "chart_images": [],
        "visual_evidence": [],
        "chart_status": status,
    }


def fetch_chart_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    loader: ChartArtifactLoader | None = state.get("chart_artifact_loader")
    chart_images: list[dict[str, Any]] = []
    visual_evidence: list[dict[str, Any]] = list(state.get("visual_evidence", []) or [])

    for chart in state.get("selected_charts", []) or []:
        artifact_id = chart.get("artifact_id")
        if not artifact_id:
            visual_evidence.append(_visual_status(chart, "not_available", "chart artifact_id is missing."))
            continue
        if loader is None:
            visual_evidence.append(_visual_status(chart, "reader_unavailable", "chart artifact byte loader is not configured."))
            continue
        try:
            chart_images.append({"chart": chart, "image_bytes": loader(str(artifact_id))})
        except Exception as exc:  # noqa: BLE001
            visual_evidence.append(_visual_status(chart, "fetch_failed", f"{type(exc).__name__}: {exc}"))

    status = "fetched" if chart_images else state.get("chart_status", "needed_unavailable")
    if not chart_images and state.get("selected_charts"):
        status = "read_failed"
    return {"chart_images": chart_images, "visual_evidence": visual_evidence, "chart_status": status}


def read_chart_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    reader: ChartReader | None = state.get("chart_reader")
    visual_evidence: list[dict[str, Any]] = list(state.get("visual_evidence", []) or [])
    read_success = False

    chart_images = state.get("chart_images", []) or []
    if chart_images:
        try:
            raw_results = (reader(chart_images, state) if reader
                           else _default_multimodal_chart_reader(chart_images, state))
            for item, raw in zip(chart_images, raw_results):
                evidence = _coerce_reader_output(item["chart"], raw)
                visual_evidence.append(evidence)
                read_success = evidence.get("status") == "read_success" or read_success
        except Exception as exc:  # noqa: BLE001
            # 배치 콜 자체가 실패하면 배치 전체를 실패로 기록한다(부분 성공을 가장하지 않음).
            error_msg = f"{type(exc).__name__}: {exc}"
            for item in chart_images:
                visual_evidence.append(_visual_status(item["chart"], "reader_failed", error_msg))

    return {
        "visual_evidence": visual_evidence,
        "chart_status": "read_success" if read_success else "read_failed",
    }


def attach_visual_evidence(state: dict[str, Any]) -> dict[str, Any]:
    result = dict(state.get("result") or {})
    chart_requests = state.get("chart_requests", []) or []
    visual_evidence = state.get("visual_evidence", []) or []
    chart_status = state.get("chart_status") or "not_needed"

    result["chart_requests"] = chart_requests
    result["visual_evidence"] = visual_evidence
    result["chart_status"] = chart_status

    if chart_requests and any(item.get("status") == "read_success" for item in visual_evidence):
        result.setdefault("key_findings", []).append(
            "Chart artifacts were inspected through a multimodal chart reader when available."
        )
    elif chart_requests and not visual_evidence:
        result.setdefault("limitations", []).append(
            "Relevant chart artifacts were requested, but no visual evidence was inspected."
        )
    if any(item.get("status") != "read_success" for item in visual_evidence):
        result.setdefault("limitations", []).append(
            "Some chart artifacts could not be read; visual claims are limited to successfully inspected charts."
        )
    return {"result": result}


def _build_chart_requests(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for profile in profiles:
        metadata = profile.get("statistical_metadata") or profile
        distribution = metadata.get("distribution", {}) or {}
        for name, stats in distribution.items():
            if not isinstance(stats, dict) or stats.get("type") != "numeric":
                continue
            information_loss_reasons = _distribution_information_loss_reasons(stats)
            if information_loss_reasons:
                requests.append({
                    "related_block": "distribution",
                    "related_keys": [f"distribution.{name}"],
                    "variables": [name],
                    "preferred_chart_types": ["histogram", "box", "violin"],
                    "reason": "numeric summary may hide distribution shape, tail, or outlier patterns",
                    "information_loss": information_loss_reasons,
                })

        for name, stats in _numeric_summary_items(profile).items():
            if not isinstance(stats, dict) or name in distribution:
                continue
            information_loss_reasons = _distribution_information_loss_reasons(stats)
            if information_loss_reasons:
                requests.append({
                    "related_block": "distribution",
                    "related_keys": [f"profile.numeric_summary.{name}"],
                    "variables": [name],
                    "preferred_chart_types": ["histogram", "box", "violin"],
                    "reason": "basic numeric summary may hide distribution shape, tail, or outlier patterns",
                    "information_loss": information_loss_reasons,
                })

        relationships = metadata.get("correlation_pairs", {}) or {}
        for key, stats in relationships.items():
            if isinstance(stats, dict) and _relationship_needs_chart(stats):
                variables = _variables_from_corr_key(key)
                requests.append({
                    "related_block": "relationship",
                    "related_keys": [f"correlation_pairs.{key}"],
                    "variables": variables,
                    "preferred_chart_types": ["scatter", "heatmap"],
                    "reason": "relationship shape is easier to verify visually",
                })

        comparisons = metadata.get("group_comparison", {}) or {}
        for metric, stats in comparisons.items():
            if isinstance(stats, dict) and _comparison_needs_chart(stats):
                requests.append({
                    "related_block": "group_comparison",
                    "related_keys": [f"group_comparison.{metric}"],
                    "variables": [str(metric)],
                    "preferred_chart_types": ["bar", "box", "groupedbox"],
                    "reason": "group spread, effect size, or small group counts require visual inspection",
                })

        if _looks_like_time_profile(profile, metadata):
            requests.append({
                "related_block": "time",
                "related_keys": ["time"],
                "variables": [],
                "preferred_chart_types": ["line", "time_series"],
                "reason": "time-series or seasonality patterns are better inspected visually",
            })
    return _dedupe_requests(requests)


def _select_available_charts(
    profiles: list[dict[str, Any]], chart_requests: list[dict[str, Any]], *, max_charts: int
) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    for profile in profiles:
        for chart in profile.get("key_charts", []) or []:
            if isinstance(chart, str):
                chart = {"filename": chart, "artifact_id": None}
            if not chart.get("artifact_id"):
                continue
            charts.append(_enrich_chart_metadata(dict(chart)))

    selected: list[dict[str, Any]] = []
    for request in chart_requests:
        match = _best_chart_match(charts, request)
        if match and match not in selected:
            match = dict(match)
            match.setdefault("related_block", request["related_block"])
            match.setdefault("related_keys", request.get("related_keys", []))
            match.setdefault("variables", request.get("variables", []))
            selected.append(match)
        if len(selected) >= max_charts:
            break
    return selected


def _default_multimodal_chart_reader(
    chart_images: list[dict[str, Any]], state: dict[str, Any]
) -> list[dict[str, Any]]:
    """chart_images 전체를 한 콜에 묶어 멀티모달로 읽는다(#194 — 차트당 1콜이던 걸 배치화).

    각 이미지 바로 앞에 chart_artifact_id를 텍스트로 붙여서 보내고, 응답도 그 id를 키로
    쓰게 강제한다 — 여러 장이 한 응답에 섞여도 어느 요약이 어느 차트 것인지 매칭이 깨지지
    않게 하기 위함.
    """
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        return [_visual_status(item["chart"], "reader_unavailable",
                               "GPT multimodal chart reader is not configured.")
                for item in chart_images]

    model = get_chat_model(model_env="CHART_READER_MODEL", default_model="gpt-5")
    prompt = (
        "아래 차트들을 각각 데이터 분석 근거로 읽어줘. "
        "차트 유형, 축/범례, 눈에 띄는 패턴, 이상치/군집/꺾이는 지점, "
        "JSON 통계와 함께 사용할 때의 주의점을 한국어로 간결하게 정리해. "
        "인과관계는 단정하지 마. 각 차트 이미지 바로 앞에 그 식별자(chart_artifact_id)가 "
        "텍스트로 붙어 있다. 반드시 그 식별자를 키로 쓴 JSON 객체 하나만 출력해라(설명 금지):\n"
        '{"식별자1": "차트1 요약 텍스트", "식별자2": "차트2 요약 텍스트"}'
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    ids: list[str] = []
    for item in chart_images:
        chart_id = str(item["chart"].get("artifact_id"))
        ids.append(chart_id)
        encoded = base64.b64encode(item["image_bytes"]).decode("ascii")
        content.append({"type": "text", "text": f"[식별자: {chart_id}]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})

    response = model.invoke([HumanMessage(content=content)])
    raw_text = str(getattr(response, "content", response)).replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("batch chart reader response is not a JSON object")

    results: list[dict[str, Any]] = []
    for item, chart_id in zip(chart_images, ids):
        chart = item["chart"]
        summary = parsed.get(chart_id)
        if summary is None:
            results.append(_visual_status(chart, "reader_failed", "batch response missing this chart's entry"))
            continue
        results.append({
            **_chart_identity(chart),
            "status": "read_success",
            "multimodal_summary": str(summary),
            "cautions": ["visual interpretation was produced by a multimodal chart reader"],
        })
    return results


def _coerce_reader_output(chart: dict[str, Any], raw: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(raw, dict):
        return {**_chart_identity(chart), "status": raw.get("status", "read_success"), **raw}
    return {**_chart_identity(chart), "status": "read_success", "multimodal_summary": str(raw), "cautions": []}


def _visual_status(chart: dict[str, Any], status: str, message: str) -> dict[str, Any]:
    return {**_chart_identity(chart), "status": status, "multimodal_summary": "", "cautions": [message]}


def _chart_identity(chart: dict[str, Any]) -> dict[str, Any]:
    return {
        "chart_artifact_id": chart.get("artifact_id"),
        "filename": chart.get("filename"),
        "chart_type": chart.get("chart_type"),
        "related_block": chart.get("related_block"),
        "related_keys": list(chart.get("related_keys", []) or []),
        "variables": list(chart.get("variables", []) or []),
    }


def _distribution_needs_chart(stats: dict[str, Any]) -> bool:
    return bool(_distribution_information_loss_reasons(stats))


def _distribution_information_loss_reasons(stats: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    skewness = _to_float(stats.get("skewness"))
    outlier_rate = _to_float(stats.get("outlier_rate_iqr"))
    normality = str(stats.get("normality") or "").casefold()

    if skewness is not None and abs(skewness) >= 1:
        reasons.append("skewness suggests mean/median alone would hide asymmetric tails")
    if outlier_rate is not None and outlier_rate >= 0.05:
        reasons.append("IQR outlier rate suggests extrema may materially affect interpretation")
    if normality in {"skewed", "heavy_tailed"}:
        reasons.append("distribution is marked as skewed or heavy-tailed")

    q1 = _first_float(stats, "q1", "25%", "p25")
    median = _first_float(stats, "median", "50%", "p50")
    q3 = _first_float(stats, "q3", "75%", "p75")
    mean = _to_float(stats.get("mean"))
    minimum = _first_float(stats, "min", "minimum")
    maximum = _first_float(stats, "max", "maximum")
    std = _to_float(stats.get("std"))

    if q1 is not None and q3 is not None:
        iqr = q3 - q1
        if iqr > 0:
            if mean is not None and median is not None and abs(mean - median) / iqr >= 0.5:
                reasons.append("mean and median diverge enough that central tendency may be misleading")
            if maximum is not None and maximum > q3 + 1.5 * iqr:
                reasons.append("upper tail extends beyond the IQR fence")
            if minimum is not None and minimum < q1 - 1.5 * iqr:
                reasons.append("lower tail extends beyond the IQR fence")
            if maximum is not None and minimum is not None and (maximum - minimum) / iqr >= 10:
                reasons.append("range is much wider than the interquartile range")
    elif mean is not None and median is not None and std not in (None, 0):
        if abs(mean - median) / abs(std) >= 0.5:
            reasons.append("mean and median diverge relative to standard deviation")

    return reasons


def _relationship_needs_chart(stats: dict[str, Any]) -> bool:
    pearson = float(stats.get("pearson_r") or 0)
    spearman = float(stats.get("spearman_r") or 0)
    return abs(pearson) >= 0.2 or abs(pearson - spearman) >= 0.15 or "binned_trend" in stats


def _comparison_needs_chart(stats: dict[str, Any]) -> bool:
    return (
        stats.get("eta_interpretation") in {"medium", "large"}
        or float(stats.get("spread_ratio") or 0) >= 2
        or int(stats.get("min_group_n") or 999_999) < 30
    )


def _looks_like_time_profile(profile: dict[str, Any], metadata: dict[str, Any]) -> bool:
    text = " ".join(str(item).casefold() for item in profile.get("profile", {}).get("columns", []) or [])
    return any(token in text for token in ("month", "date", "time", "year")) and bool(profile.get("key_charts"))


def _numeric_summary_items(profile: dict[str, Any]) -> dict[str, Any]:
    nested_profile = profile.get("profile") or {}
    summary = profile.get("numeric_summary") or nested_profile.get("numeric_summary") or {}
    return summary if isinstance(summary, dict) else {}


def _variables_from_corr_key(key: str) -> list[str]:
    if key.startswith("corr_") and "_vs_" in key:
        left, right = key.removeprefix("corr_").split("_vs_", 1)
        return [left, right]
    return []


def _enrich_chart_metadata(chart: dict[str, Any]) -> dict[str, Any]:
    filename = str(chart.get("filename") or "").casefold()
    if "scatter" in filename:
        chart.setdefault("chart_type", "scatter")
    elif "heatmap" in filename:
        chart.setdefault("chart_type", "heatmap")
    elif "dist" in filename:
        chart.setdefault("chart_type", "histogram")
    elif "box" in filename or "violin" in filename:
        chart.setdefault("chart_type", "box")
    elif "line" in filename or filename.startswith("ts_"):
        chart.setdefault("chart_type", "line")
    elif "bar" in filename:
        chart.setdefault("chart_type", "bar")
    return chart


def _best_chart_match(charts: list[dict[str, Any]], request: dict[str, Any]) -> dict[str, Any] | None:
    preferred = set(request.get("preferred_chart_types", []) or [])
    variables = [str(item).casefold() for item in request.get("variables", []) or []]
    scored: list[tuple[int, dict[str, Any]]] = []
    for chart in charts:
        filename = str(chart.get("filename") or "").casefold()
        chart_type = str(chart.get("chart_type") or "").casefold()
        score = 0
        if chart_type in preferred:
            score += 3
        score += sum(1 for variable in variables if variable and variable in filename)
        if request.get("related_block") == "relationship" and chart_type in {"scatter", "heatmap"}:
            score += 1
        if request.get("related_block") == "distribution" and chart_type in {"histogram", "box"}:
            score += 1
        if request.get("related_block") == "time" and chart_type == "line":
            score += 1
        if score:
            scored.append((score, chart))
    if not scored:
        return None
    return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]


def _dedupe_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[dict[str, Any]] = []
    for request in requests:
        key = (request["related_block"], tuple(request.get("related_keys", [])))
        if key not in seen:
            seen.add(key)
            deduped.append(request)
    return deduped


def _first_float(stats: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _to_float(stats.get(key))
        if value is not None:
            return value
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
