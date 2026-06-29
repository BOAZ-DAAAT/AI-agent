from __future__ import annotations

from typing import Any


def add_benjamini_hochberg_adjustment(pairs: list[dict[str, Any]]) -> None:
    if not pairs:
        return
    ranked = sorted(enumerate(pairs), key=lambda item: item[1]["p_value"])
    total = len(ranked)
    adjusted = [1.0] * total
    running = 1.0
    for reverse_rank in range(total - 1, -1, -1):
        original_index, pair = ranked[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(pair["p_value"]) * total / rank)
        adjusted[original_index] = min(1.0, running)
    for index, pair in enumerate(pairs):
        pair["p_value_adjusted_bh"] = adjusted[index]
