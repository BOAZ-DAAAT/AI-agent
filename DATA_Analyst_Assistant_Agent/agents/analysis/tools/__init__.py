"""Vetted heavy analysis tools kept as codegen primitives.

The codegen-first analysis path generates its own pandas/scipy/statsmodels code
for most questions. Only these five statistically heavy methods are kept as
vetted primitives (injected into the generate sandbox via
nodes.generate._PRIMITIVE_TOOLS) so generated code composes them instead of
re-deriving PyMC / lifelines / OR-Tools math.
"""

from DATA_Analyst_Assistant_Agent.agents.analysis.tools.bayesian import run_bayesian_mmm
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.business import optimize_business_allocation
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.customer import estimate_probabilistic_clv
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.geo import analyze_geospatial_hotspots
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.survival import analyze_survival

ANALYSIS_TOOLS = {
    item.name: item
    for item in (
        run_bayesian_mmm,
        estimate_probabilistic_clv,
        analyze_survival,
        analyze_geospatial_hotspots,
        optimize_business_allocation,
    )
}

__all__ = ["ANALYSIS_TOOLS"]
