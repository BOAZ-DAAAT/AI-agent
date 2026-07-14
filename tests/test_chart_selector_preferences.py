from DATA_Analyst_Assistant_Agent.agents.eda.lib import chart_selector_skill as css


def test_selector_keeps_preferred_new_family_when_truncating():
    paths = [f"dist_metric_{i}.png" for i in range(8)] + ["segment_profile_flag.png"]
    out = css._ensure_preferred_survives(paths)
    assert len(out) == css.TOTAL_MAX
    assert "segment_profile_flag.png" in out
