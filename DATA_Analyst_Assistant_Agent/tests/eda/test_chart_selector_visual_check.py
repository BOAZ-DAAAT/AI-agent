"""_visual_sanity_check 배치화(#194) 계약 테스트 — 파일명 매칭·배치 크기·실패 폴백."""

from __future__ import annotations

import json
import os

import pytest

from DATA_Analyst_Assistant_Agent.agents.eda.lib import chart_selector_skill as css


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def invoke(self, messages):
        self.calls.append(messages[0].content)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _FakeResponse(reply)


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _make_pngs(tmp_path, names: list[str]) -> list[str]:
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"fake-png-bytes")
        paths.append(str(p))
    return paths


def test_batches_into_groups_of_batch_size(tmp_path, monkeypatch):
    names = [f"chart_{i}.png" for i in range(5)]  # 5장 → 배치 4 + 1 = 2콜
    paths = _make_pngs(tmp_path, names)

    def reply_for(batch_names: list[str]) -> str:
        return json.dumps({n: {"ok": True, "issue": ""} for n in batch_names})

    fake = _FakeModel([reply_for(names[:4]), reply_for(names[4:])])
    monkeypatch.setattr(css, "_load_chart_reader_llm", lambda: fake)

    kept, dropped, check_failures = css._visual_sanity_check(paths)

    assert len(fake.calls) == 2  # 1장당 1콜이 아니라 배치 2번
    assert sorted(os.path.basename(p) for p in kept) == sorted(names)
    assert dropped == []
    assert check_failures == 0


def test_filename_keyed_verdict_drops_correct_chart(tmp_path, monkeypatch):
    names = ["good.png", "bad.png"]
    paths = _make_pngs(tmp_path, names)
    reply = json.dumps({
        "good.png": {"ok": True, "issue": ""},
        "bad.png": {"ok": False, "issue": "범례가 데이터를 가림"},
    })
    fake = _FakeModel([reply])
    monkeypatch.setattr(css, "_load_chart_reader_llm", lambda: fake)

    kept, dropped, check_failures = css._visual_sanity_check(paths)

    assert [os.path.basename(p) for p in kept] == ["good.png"]
    assert dropped == [{"chart": "bad.png", "reason": "범례가 데이터를 가림"}]
    assert check_failures == 0


def test_batch_call_failure_conservatively_keeps_whole_batch(tmp_path, monkeypatch):
    names = ["a.png", "b.png"]
    paths = _make_pngs(tmp_path, names)
    fake = _FakeModel([RuntimeError("model down")])
    monkeypatch.setattr(css, "_load_chart_reader_llm", lambda: fake)

    kept, dropped, check_failures = css._visual_sanity_check(paths)

    assert sorted(os.path.basename(p) for p in kept) == names
    assert dropped == []
    assert check_failures == 2  # 배치 전체가 check_failure로 집계됨
