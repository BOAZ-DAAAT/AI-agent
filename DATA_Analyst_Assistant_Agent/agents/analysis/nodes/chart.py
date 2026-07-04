from __future__ import annotations

import base64
import os
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model


ChartArtifactLoader = Callable[[str], bytes]
ChartReader = Callable[[dict[str, Any], bytes, dict[str, Any]], dict[str, Any] | str]


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

    for item in state.get("chart_images", []) or []:
        chart = item["chart"]
        image_bytes = item["image_bytes"]
        try:
            raw = reader(chart, image_bytes, state) if reader else _default_multimodal_chart_reader(chart, image_bytes, state)
            evidence = _coerce_reader_output(chart, raw)
            visual_evidence.append(evidence)
            read_success = evidence.get("status") == "read_success" or read_success
        except Exception as exc:  # noqa: BLE001
            visual_evidence.append(_visual_status(chart, "reader_failed", f"{type(exc).__name__}: {exc}"))

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
            if _distribution_needs_chart(stats):
                requests.append({
                    "related_block": "distribution",
                    "related_keys": [f"distribution.{name}"],
                    "variables": [name],
                    "preferred_chart_types": ["histogram", "box", "violin"],
                    "reason": "distribution is skewed, heavy-tailed, or has notable outliers",
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


def _default_multimodal_chart_reader(chart: dict[str, Any], image_bytes: bytes, state: dict[str, Any]) -> dict[str, Any]:
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        return _visual_status(chart, "reader_unavailable", "GPT multimodal chart reader is not configured.")

    model = get_chat_model(model_env="CHART_READER_MODEL", default_model=os.getenv("CHART_READER_MODEL", "gpt-5"))
    encoded = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "이 차트를 데이터 분석 근거로 읽어줘. "
        "차트 유형, 축/범례, 눈에 띄는 패턴, 이상치/군집/꺾이는 지점, "
        "JSON 통계와 함께 사용할 때의 주의점을 한국어로 간결하게 정리해. "
        "인과관계는 단정하지 마."
    )
    response = model.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ])
    ])
    return {
        **_chart_identity(chart),
        "status": "read_success",
        "multimodal_summary": str(getattr(response, "content", response)),
        "cautions": ["visual interpretation was produced by a multimodal chart reader"],
    }


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
    return (
        abs(float(stats.get("skewness") or 0)) >= 1
        or float(stats.get("outlier_rate_iqr") or 0) >= 0.05
        or stats.get("normality") in {"skewed", "heavy_tailed"}
    )


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
