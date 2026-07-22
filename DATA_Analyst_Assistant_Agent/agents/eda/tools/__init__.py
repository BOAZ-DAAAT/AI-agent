"""LLM에 노출되는 LangChain @tool 모음 (파일당 @tool 1개) + 노드별 툴 그룹.

각 @tool은 lib/의 skill(또는 기초 프로파일)을 호출하는 얇은 래퍼다.
ReAct 루프(run_mini_react 등)는 nodes/tool_runner.py 로 분리되어 있다.
"""

from DATA_Analyst_Assistant_Agent.agents.eda.tools.profile_data import profile_data
from DATA_Analyst_Assistant_Agent.agents.eda.tools.run_quality import run_quality
from DATA_Analyst_Assistant_Agent.agents.eda.tools.run_distribution import run_distribution
from DATA_Analyst_Assistant_Agent.agents.eda.tools.run_comparison import run_comparison
from DATA_Analyst_Assistant_Agent.agents.eda.tools.run_relationship import run_relationship
from DATA_Analyst_Assistant_Agent.agents.eda.tools.run_time import run_time
from DATA_Analyst_Assistant_Agent.agents.eda.tools.run_clustering import run_clustering

# ─────────────────────────────
# 노드별 툴 그룹 (노드가 mini-ReAct에 넘기는 단위)
# ─────────────────────────────
INSPECT_TOOLS      = [profile_data]
QUALITY_TOOLS      = [run_quality]
DISTRIBUTION_TOOLS = [run_distribution]
COMPARISON_TOOLS   = [run_comparison]
RELATIONSHIP_TOOLS = [run_relationship]
TIME_TOOLS         = [run_time]
CLUSTERING_TOOLS   = [run_clustering]

__all__ = [
    "profile_data",
    "run_quality",
    "run_distribution",
    "run_comparison",
    "run_relationship",
    "run_time",
    "run_clustering",
    "INSPECT_TOOLS",
    "QUALITY_TOOLS",
    "DISTRIBUTION_TOOLS",
    "COMPARISON_TOOLS",
    "RELATIONSHIP_TOOLS",
    "TIME_TOOLS",
    "CLUSTERING_TOOLS",
]
