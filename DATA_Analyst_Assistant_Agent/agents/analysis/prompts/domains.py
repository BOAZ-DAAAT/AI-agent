"""Domain prompt templates for the codegen-first analysis path.

Hybrid domain routing: `classify` picks one of the known domain labels, or
falls back to ``"general"`` when nothing fits. The branch is preserved (each
domain carries its own framing), but the branch *endpoint* is LLM-generated
code rather than a fixed tool. `framing` is injected into the generate prompt
so the produced code follows domain-appropriate methodology.

Adding a new domain = adding one entry here. No dispatch/validation code needs
to change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainTemplate:
    label: str
    summary: str  # shown to classify so it can pick the right label
    framing: str  # injected into generate so codegen follows the right method


DOMAIN_TEMPLATES: dict[str, DomainTemplate] = {
    "ecommerce_behavior": DomainTemplate(
        "ecommerce_behavior",
        "User event/session behavior: funnels, journeys, cart actions, "
        "conversion vs drop-off, path/sequence questions.",
        "The data is event- or session-level user behavior. Preserve event "
        "ordering and per-entity sequences; never aggregate away the order of "
        "actions. When asked whether a behavior leads to exit vs continued "
        "exploration, look at what events follow it per session, conversion "
        "downstream, and whether distinct segments exist. State conversion and "
        "drop-off as observed rates, not causal effects.",
    ),
    "timeseries": DomainTemplate(
        "timeseries",
        "Metrics over time: trend, growth, volatility, seasonality, "
        "period-over-period change.",
        "Resample to the provided time_grain before fitting any trend. Report "
        "growth, slope with a significance measure, volatility, and seasonality "
        "only when enough periods exist. If too few periods exist at the chosen "
        "grain, say so explicitly instead of forcing a fit.",
    ),
    "customer": DomainTemplate(
        "customer",
        "Customer lifecycle: cohorts, retention, RFM, churn, segmentation, "
        "lifetime value.",
        "Work at the entity (customer) grain. Require a stable entity id and "
        "consistent event logging. For retention/cohorts use acquisition-period "
        "buckets; for segments report cluster quality and treat them as "
        "descriptive, not prescriptive.",
    ),
    "finance": DomainTemplate(
        "finance",
        "Money metrics: profit, margin, unit economics, cost/revenue "
        "contribution, concentration.",
        "Be explicit about the unit of analysis and the profit/margin "
        "definition. Separate mix effects from within-group changes when "
        "comparing periods. Contribution is descriptive concentration, not a "
        "causal driver claim.",
    ),
    "general": DomainTemplate(
        "general",
        "Anything that does not clearly fit the specialized domains above.",
        "Pick the simplest statistically valid method that answers the "
        "question. State assumptions and effect sizes; never phrase "
        "associations as causal effects.",
    ),
}

KNOWN_DOMAINS: tuple[str, ...] = tuple(DOMAIN_TEMPLATES)


def domain_framing(label: str | None) -> str:
    """Return the framing for a domain label, falling back to ``general``."""

    template = DOMAIN_TEMPLATES.get(label or "", DOMAIN_TEMPLATES["general"])
    return template.framing


def domain_catalog_text() -> str:
    """Compact label+summary list shown to the classify node."""

    return "\n".join(
        f"- {template.label}: {template.summary}" for template in DOMAIN_TEMPLATES.values()
    )
