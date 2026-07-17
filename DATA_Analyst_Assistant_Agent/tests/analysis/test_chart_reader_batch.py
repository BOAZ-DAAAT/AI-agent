"""분석 에이전트 멀티모달 차트 리더 배치화(#194) 계약 테스트."""

from __future__ import annotations

import json

import pytest

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes import chart as chart_mod


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    def __init__(self, reply) -> None:
        self.reply = reply
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages[0].content)
        if isinstance(self.reply, Exception):
            raise self.reply
        return _FakeResponse(self.reply)


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _chart_images(ids: list[str]) -> list[dict]:
    return [
        {"chart": {"artifact_id": cid, "filename": f"{cid}.png"}, "image_bytes": b"fake-bytes"}
        for cid in ids
    ]


def test_default_reader_batches_all_charts_into_one_call(monkeypatch):
    ids = ["c1", "c2", "c3"]
    reply = json.dumps({cid: f"summary for {cid}" for cid in ids})
    fake = _FakeModel(reply)
    monkeypatch.setattr(chart_mod, "get_chat_model", lambda **kwargs: fake)

    results = chart_mod._default_multimodal_chart_reader(_chart_images(ids), {})

    assert len(fake.calls) == 1  # 3장이어도 콜은 1번
    assert [r["status"] for r in results] == ["read_success"] * 3
    assert [r["multimodal_summary"] for r in results] == [f"summary for {cid}" for cid in ids]


def test_default_reader_missing_id_in_response_marked_failed(monkeypatch):
    ids = ["c1", "c2"]
    reply = json.dumps({"c1": "summary for c1"})  # c2 누락
    fake = _FakeModel(reply)
    monkeypatch.setattr(chart_mod, "get_chat_model", lambda **kwargs: fake)

    results = chart_mod._default_multimodal_chart_reader(_chart_images(ids), {})

    assert results[0]["status"] == "read_success"
    assert results[1]["status"] == "reader_failed"


def test_read_chart_artifacts_marks_whole_batch_failed_on_reader_exception(monkeypatch):
    ids = ["c1", "c2"]

    def failing_reader(chart_images, state):
        raise RuntimeError("model down")

    state = {"chart_images": _chart_images(ids), "chart_reader": failing_reader}
    result = chart_mod.read_chart_artifacts(state)

    assert result["chart_status"] == "read_failed"
    assert len(result["visual_evidence"]) == 2
    assert all(e["status"] == "reader_failed" for e in result["visual_evidence"])
