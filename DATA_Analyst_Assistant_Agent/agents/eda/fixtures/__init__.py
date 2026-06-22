"""EDA fixture 패키지.

SQL 에이전트가 '이상적으로' 줄 분석 마트를 grain별로 미리 만들어 둔 것.
SQL/풀 파이프라인 없이 EDA를 개발·테스트하기 위한 입력(df + 입력계약).
"""

from DATA_Analyst_Assistant_Agent.agents.eda.fixtures.contract import EdaInput, load_fixture

__all__ = ["EdaInput", "load_fixture"]
