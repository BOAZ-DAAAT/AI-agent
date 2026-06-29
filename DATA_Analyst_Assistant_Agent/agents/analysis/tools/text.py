from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from DATA_Analyst_Assistant_Agent.agents.analysis.tools.lib.frame import frame_from_records
import re
from collections import Counter

@tool
def analyze_text(
    records: list[dict[str, Any]], text_column: str, top_n: int = 30
) -> dict[str, Any]:
    """Summarize text coverage and frequent terms without inventing semantic labels."""

    df = frame_from_records(records)
    if text_column not in df.columns:
        raise ValueError("Text column is required for text analysis.")
    texts = df[text_column].dropna().astype(str)
    if texts.empty:
        raise ValueError("Text analysis requires non-empty text values.")
    tokens = [
        token.casefold()
        for text in texts
        for token in re.findall(r"\b[^\W\d_]{2,}\b", text, flags=re.UNICODE)
    ]
    counts = Counter(tokens)
    lengths = texts.str.len()
    return {
        "method": "frequency_based_text_profile",
        "document_count": len(texts),
        "token_count": len(tokens),
        "average_text_length": float(lengths.mean()),
        "top_terms": [{"term": term, "count": count} for term, count in counts.most_common(top_n)],
        "semantic_interpretation_required": True,
    }
