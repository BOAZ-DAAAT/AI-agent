"""NodeSummaryResult → 마크다운 렌더링 + 차트 로컬 임베드 검증."""

from __future__ import annotations

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.artifacts import ArtifactType

from DATA_Analyst_Assistant_Agent.agents.common import AgentRuntime
from DATA_Analyst_Assistant_Agent.shared.backend_adapter import BackendAdapter
from DATA_Analyst_Assistant_Agent.supervisor.summary.markdown import render_node_summary_markdown
from DATA_Analyst_Assistant_Agent.supervisor.summary.schemas import (
    EDASummaryDetail,
    FindingSection,
    InsightSummaryDetail,
    NodeSummaryResult,
    SQLSummaryDetail,
)

_FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-chart-bytes"


@pytest.fixture()
def adapter(tmp_path) -> BackendAdapter:
    return BackendAdapter(config=BackendConfig(base_data_dir=tmp_path / ".data_agent"))


@pytest.fixture()
def runtime(adapter: BackendAdapter) -> AgentRuntime:
    return AgentRuntime(adapter=adapter)


def _register_chart(adapter, run_id, filename="dist_x.png") -> str:
    return adapter.register_artifact(
        run_id, ArtifactType.chart, content_bytes=_FAKE_PNG_BYTES, filename=filename,
        created_by_tool="test.summary.markdown", metadata={"kind": "eda_chart"},
    ).artifact_id


def test_eda_chart_is_saved_locally_and_embedded_with_relative_path(adapter, runtime, tmp_path):
    run = adapter.create_run(thread_id="thread_md_eda")
    chart_id = _register_chart(adapter, run.run_id, filename="dist_delivery_days.png")

    result = NodeSummaryResult(
        title="EDA 요약", subtitle="부제", background="배경",
        conclusion="결론", key_finding="한줄요약", source_kind="eda_summary",
        detail=EDASummaryDetail(
            data_profile="프로파일",
            statistical_findings=[FindingSection(
                heading="배송일 분포", body="오른쪽 꼬리가 깁니다.", chart_artifact_ids=[chart_id],
            )],
        ),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    markdown = render_node_summary_markdown(result, runtime, out_dir)

    assert "![배송일 분포](charts/dist_delivery_days.png)" in markdown
    saved_path = out_dir / "charts" / "dist_delivery_days.png"
    assert saved_path.exists()
    assert saved_path.read_bytes() == _FAKE_PNG_BYTES


def test_insight_supporting_chart_is_embedded(adapter, runtime, tmp_path):
    run = adapter.create_run(thread_id="thread_md_insight")
    chart_id = _register_chart(adapter, run.run_id, filename="category_share.png")

    result = NodeSummaryResult(
        title="인사이트 요약", subtitle="부제", background="배경",
        conclusion="결론", key_finding="한줄요약", source_kind="insight_payload",
        detail=InsightSummaryDetail(
            answer="toys 카테고리가 1위입니다.",
            supporting_charts=[FindingSection(
                heading="카테고리별 매출 비교", body="카테고리별 매출 비교", chart_artifact_ids=[chart_id],
            )],
        ),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    markdown = render_node_summary_markdown(result, runtime, out_dir)

    assert "![카테고리별 매출 비교](charts/category_share.png)" in markdown
    assert (out_dir / "charts" / "category_share.png").exists()


def test_missing_chart_artifact_is_skipped_without_crashing(adapter, runtime, tmp_path):
    result = NodeSummaryResult(
        title="EDA 요약", subtitle="부제", background="배경",
        conclusion="결론", key_finding="한줄요약", source_kind="eda_summary",
        detail=EDASummaryDetail(
            statistical_findings=[FindingSection(
                heading="없는 차트", body="본문", chart_artifact_ids=["art_does_not_exist"],
            )],
        ),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    markdown = render_node_summary_markdown(result, runtime, out_dir)

    assert "없는 차트" in markdown
    assert "![" not in markdown  # 못 받아온 차트는 이미지 태그 자체를 안 만든다
    assert not (out_dir / "charts").exists() or not list((out_dir / "charts").glob("*"))


def test_no_charts_means_no_charts_folder(adapter, runtime, tmp_path):
    result = NodeSummaryResult(
        title="SQL 요약", subtitle="부제", background="배경",
        conclusion="결론", key_finding="한줄요약", source_kind="sql_plan",
        detail=EDASummaryDetail(data_profile="프로파일"),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    render_node_summary_markdown(result, runtime, out_dir)

    assert not (out_dir / "charts").exists()


def test_sql_mart_preview_renders_as_markdown_table(adapter, runtime, tmp_path):
    result = NodeSummaryResult(
        title="SQL 요약", subtitle="부제", background="배경",
        conclusion="결론", key_finding="한줄요약", source_kind="sql_plan",
        detail=SQLSummaryDetail(
            mart_grain="customer_unique_id 당 1행",
            mart_columns=["customer_unique_id", "monetary_total"],
            mart_preview=[
                {"customer_unique_id": "c1", "monetary_total": 120.5},
                {"customer_unique_id": "c2", "monetary_total": None},
            ],
        ),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    markdown = render_node_summary_markdown(result, runtime, out_dir)

    assert "최종 구성된 데이터마트는 다음과 같습니다" in markdown
    assert "| customer_unique_id | monetary_total |" in markdown
    assert "| c1 | 120.5 |" in markdown
    assert "| c2 |  |" in markdown  # None은 빈 칸으로


def test_eda_quality_issues_render_as_warning_callout(adapter, runtime, tmp_path):
    result = NodeSummaryResult(
        title="EDA", subtitle="부제", background="배경",
        conclusion="결론", key_finding="한줄요약", source_kind="eda_summary",
        detail=EDASummaryDetail(quality_issues=["이상치 674건 존재", "결측 165건 존재"]),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    markdown = render_node_summary_markdown(result, runtime, out_dir)

    assert "> [!WARNING]" in markdown
    assert "> **품질 이슈**" in markdown
    assert "> - 이상치 674건 존재" in markdown


def test_eda_finding_body_is_collapsed_in_toggle_while_chart_stays_visible(adapter, runtime, tmp_path):
    run = adapter.create_run(thread_id="thread_md_toggle")
    chart_id = _register_chart(adapter, run.run_id, filename="scatter.png")

    result = NodeSummaryResult(
        title="EDA", subtitle="부제", background="배경",
        conclusion="결론", key_finding="한줄요약", source_kind="eda_summary",
        detail=EDASummaryDetail(
            statistical_findings=[FindingSection(
                heading="배송일-리뷰점수 관계", body="긴 서술 본문입니다.",
                source_label="관계 분석", chart_artifact_ids=[chart_id],
            )],
        ),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    markdown = render_node_summary_markdown(result, runtime, out_dir)

    assert "![배송일-리뷰점수 관계](charts/scatter.png)" in markdown
    assert "<details>" in markdown
    assert "<summary>배송일-리뷰점수 관계 · 관계 분석</summary>" in markdown
    assert "긴 서술 본문입니다." in markdown
    # 차트 이미지 줄이 <details> 여는 태그보다 먼저 나와야(항상 보이도록) 한다.
    assert markdown.index("![배송일-리뷰점수 관계]") < markdown.index("<details>")


def test_sql_derived_columns_render_as_table(adapter, runtime, tmp_path):
    result = NodeSummaryResult(
        title="SQL", subtitle="부제", background="배경",
        conclusion="결론", key_finding="한줄요약", source_kind="sql_plan",
        detail=SQLSummaryDetail(derived_columns=[
            FindingSection(heading="order_month", body="월 단위로 묶은 파생 컬럼입니다."),
        ]),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    markdown = render_node_summary_markdown(result, runtime, out_dir)

    assert "| 컬럼 | 설명 |" in markdown
    assert "| order_month | 월 단위로 묶은 파생 컬럼입니다. |" in markdown
    assert "### order_month" not in markdown  # 예전처럼 별도 소제목으로 반복되지 않는다


def test_sql_without_preview_has_no_table(adapter, runtime, tmp_path):
    result = NodeSummaryResult(
        title="SQL 요약", subtitle="부제", background="배경",
        conclusion="결론", key_finding="한줄요약", source_kind="sql_plan",
        detail=SQLSummaryDetail(mart_grain="customer_unique_id 당 1행"),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    markdown = render_node_summary_markdown(result, runtime, out_dir)

    assert "최종 구성된 데이터마트는 다음과 같습니다" not in markdown
