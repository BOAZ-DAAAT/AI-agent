"""Classify node: extract analysis intent + resolve a deterministic time grain.

Replaces the fixed question_type catalog. The LLM produces free-form intent and
picks a domain label (hybrid: known label or ``general``). The time grain is NOT
an LLM decision — it is computed from the actual dataframe span so a "last month"
question is analyzed daily instead of being forced to a monthly resample.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from DATA_Analyst_Assistant_Agent.agents.analysis.prompts.domains import (
    KNOWN_DOMAINS,
    domain_catalog_text,
)
from DATA_Analyst_Assistant_Agent.agents.analysis.schemas import (
    AnalysisContext,
    AnalysisIntent,
    TimeGrain,
)
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model


# Grain thresholds: pick the finest grain that still yields enough periods for a
# stable fit. Ordered from finest to coarsest; first matching span wins.
_GRAIN_BY_MAX_SPAN_DAYS: tuple[tuple[int, TimeGrain], ...] = (
    (45, "D"),      # up to ~6 weeks -> daily
    (365, "W"),     # up to ~1 year  -> weekly
    (1095, "M"),    # up to ~3 years -> monthly
    (1825, "Q"),    # up to ~5 years -> quarterly
)
_COARSEST_GRAIN: TimeGrain = "Y"


CLASSIFY_SYSTEM_PROMPT = """You classify one data-analysis request before code is generated.

Return the AnalysisIntent schema. Rules:
- objective: restate, in one sentence, what the analysis must answer.
- analysis_focus: the concrete quantities or relationships to compute.
- domain: choose exactly one label from the provided domain list; use "general"
  if none clearly fits. Do not invent new labels.
- metric_hints/dimension_hints/entity_hints: column names from the context that
  are likely the metric(s), grouping dimension(s), and entity id(s). Use only
  columns that exist.
- time_column: the single best time column if the question is time-based, else null.
- is_time_based: true only if answering requires ordering or aggregating over time.
- Do NOT set time_grain or time_span_days; those are computed deterministically.
- requires_human_review + review_reason is not clarification. Set it when the
  request is likely to require an analyst-chosen operational definition, proxy
  label, heuristic segment, threshold, scoring rule, or other analysis convention
  that could materially change interpretation. Do not ask the user before
  analysis when the data can support candidate options; proceed with analysis
  and let the generated result return review_request only after it has evidence
  for an approval or a meaningful choice.
- EDA candidate insights/hypotheses are exploratory hints only. Use them to
  clarify relevant focus when they match the user request; do not turn every
  candidate into required analysis work.
- Treat EDA final_summary as a handoff summary, not as a final answer or tested
  statistical result.
"""


def resolve_time_grain(
    dataframe: pd.DataFrame, time_column: str | None
) -> tuple[TimeGrain | None, int | None]:
    """Compute (grain, span_days) from the actual data. Deterministic, no LLM."""

    if not time_column or time_column not in dataframe.columns:
        return None, None
    parsed = pd.to_datetime(dataframe[time_column], errors="coerce").dropna()
    if parsed.empty:
        return None, None
    span_days = int((parsed.max() - parsed.min()).days)
    for max_span, grain in _GRAIN_BY_MAX_SPAN_DAYS:
        if span_days <= max_span:
            return grain, span_days
    return _COARSEST_GRAIN, span_days


def classify_intent(
    context: AnalysisContext,
    dataframe: pd.DataFrame,
    *,
    model: Any | None = None,
) -> AnalysisIntent:
    """LLM extracts intent; the node fixes the time grain deterministically."""

    chat_model = model or get_chat_model(temperature=0)
    structured_model = chat_model.with_structured_output(AnalysisIntent)
    human = (
        f"Domain labels:\n{domain_catalog_text()}\n\n"
        f"Analysis context:\n{context.model_dump_json(indent=2)}"
    )
    result = structured_model.invoke([
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=human),
    ])
    intent = result if isinstance(result, AnalysisIntent) else AnalysisIntent.model_validate(result)

    # Hybrid domain fallback: unknown label -> general.
    if intent.domain not in KNOWN_DOMAINS:
        intent.domain = "general"

    # Deterministic grain overrides any value the LLM may have guessed.
    time_column = intent.time_column if intent.time_column in context.columns else None
    intent.time_column = time_column
    intent.time_grain, intent.time_span_days = resolve_time_grain(dataframe, time_column)
    if not time_column:
        intent.is_time_based = False
    return intent
