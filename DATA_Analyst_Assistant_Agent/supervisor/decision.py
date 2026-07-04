from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from DATA_Analyst_Assistant_Agent.supervisor.prompts import DECIDE_NEXT_ACTION_PROMPT
from DATA_Analyst_Assistant_Agent.supervisor.state import NextAction, SupervisorState, artifact_ids_by_agent


_SNAPSHOT_MAX_TEXT = 120
_SNAPSHOT_MAX_ITEMS = 3
_SNAPSHOT_MAX_DEPTH = 3


class SupervisorDecision(BaseModel):
    next_action: NextAction
    reason: str = ""


def parse_decision_json(text: str) -> SupervisorDecision:
    return SupervisorDecision.model_validate_json(_extract_json_object(text))


def decide_next_action(state: SupervisorState, model: Any | None = None) -> SupervisorDecision:
    if model is None:
        raise RuntimeError("Supervisor LLM decision model is required.")

    response = model.invoke(_decision_messages(state))
    content = getattr(response, "content", response)
    return parse_decision_json(str(content))


def _decision_messages(state: SupervisorState) -> list[dict[str, str]]:
    snapshot = _compact_snapshot(state)
    return [
        {"role": "system", "content": DECIDE_NEXT_ACTION_PROMPT},
        {"role": "user", "content": json.dumps(snapshot, ensure_ascii=False, sort_keys=True)},
    ]


def _compact_snapshot(state: SupervisorState) -> dict[str, Any]:
    return {
        "query": _truncate_for_snapshot(state.get("clarified_query") or state.get("latest_user_query", "")),
        "plan": _truncate_for_snapshot(state.get("analysis_plan") or {}),
        "completed_agents": list(state.get("completed_agents", [])),
        "failed_agents": list(state.get("failed_agents", [])),
        "artifacts": artifact_ids_by_agent(state),
        "validation_results": _truncate_for_snapshot(list(state.get("validation_results", []))[-3:]),
        "step_summaries": _truncate_for_snapshot(list(state.get("step_summaries", []))[-5:]),
        "terminal_state": _truncate_for_snapshot(state.get("terminal_state", "")),
    }


def _truncate_for_snapshot(
    value: Any,
    *,
    max_text: int = _SNAPSHOT_MAX_TEXT,
    max_items: int = _SNAPSHOT_MAX_ITEMS,
    depth: int = _SNAPSHOT_MAX_DEPTH,
) -> Any:
    if isinstance(value, str):
        return value[:max_text]
    if depth <= 0:
        if isinstance(value, (dict, list)):
            return "..."
        return value
    if isinstance(value, dict):
        return {
            str(key)[:max_text]: _truncate_for_snapshot(
                item,
                max_text=max_text,
                max_items=max_items,
                depth=depth - 1,
            )
            for key, item in list(value.items())[:max_items]
        }
    if isinstance(value, list):
        return [
            _truncate_for_snapshot(
                item,
                max_text=max_text,
                max_items=max_items,
                depth=depth - 1,
            )
            for item in value[:max_items]
        ]
    return value


def _extract_json_object(text: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1)

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end]
    raise ValueError("No JSON object found in decision text")
