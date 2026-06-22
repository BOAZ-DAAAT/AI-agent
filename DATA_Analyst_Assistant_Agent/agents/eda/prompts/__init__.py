"""EDA 노드용 LLM 프롬프트 패키지.

프롬프트별 모듈로 분리하되, 기존 import 경로(`...eda.prompts import X`)가
그대로 동작하도록 여기서 전부 re-export 한다.
"""

from DATA_Analyst_Assistant_Agent.agents.eda.prompts.analysis import (
    comparison_prompt,
    distribution_prompt,
    inspect_prompt,
    quality_prompt,
    react_fix_prompt,
    relationship_prompt,
    time_prompt,
)
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.classify import classify_columns_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.hypothesis import (
    handoff_summary_prompt,
    hypothesis_prompt,
)
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.insight import insight_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.planner import planner_prompt
from DATA_Analyst_Assistant_Agent.agents.eda.prompts.validator import validator_prompt

__all__ = [
    "classify_columns_prompt",
    "planner_prompt",
    "validator_prompt",
    "inspect_prompt",
    "quality_prompt",
    "distribution_prompt",
    "comparison_prompt",
    "relationship_prompt",
    "time_prompt",
    "react_fix_prompt",
    "insight_prompt",
    "hypothesis_prompt",
    "handoff_summary_prompt",
]
