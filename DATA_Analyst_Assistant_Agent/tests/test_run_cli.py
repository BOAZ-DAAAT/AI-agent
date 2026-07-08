from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from DATA_Analyst_Assistant_Agent import run as cli
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    OrchestrationState,
    SupervisorInterruptPayload,
    SupervisorRunResult,
    SupervisorTerminalState,
)


class FakeBackendAdapter:
    base_data_dir = ".data_agent"


class FakeSupervisor:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.run_calls: list[dict[str, Any]] = []
        self.resume_calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, *, thread_id: str | None = None, datasource_id: str | None = None):
        self.run_calls.append(
            {
                "query": query,
                "thread_id": thread_id,
                "datasource_id": datasource_id,
            }
        )
        return self.result

    def resume(self, thread_id: str, resume_payload: dict[str, Any]):
        self.resume_calls.append((thread_id, resume_payload))
        return self.result


def _state_result(user_query: str = "매출\n추가 답변: 최근 6개월 월별 매출") -> SupervisorRunResult:
    return SupervisorRunResult(
        kind="state",
        state=OrchestrationState(
            run_id="run_resumed_001",
            thread_id="thread_1",
            user_query=user_query,
            terminal_state=SupervisorTerminalState.completed,
        ),
    )


def _interrupt_result() -> SupervisorRunResult:
    return SupervisorRunResult(
        kind="interrupt",
        interrupt=SupervisorInterruptPayload(
            type="clarification",
            status="waiting_input",
            run_id="run_001",
            thread_id="thread_1",
            question="어떤 기간과 단위로 매출을 분석할까요?",
            node="collect_clarification",
        ),
    )


def _patch_cli_runtime(monkeypatch: pytest.MonkeyPatch, result: Any) -> FakeSupervisor:
    supervisor = FakeSupervisor(result)
    monkeypatch.setattr(cli, "load_dotenv", lambda _path: None)
    monkeypatch.setattr(cli, "_normalize_env_aliases", lambda: None)
    monkeypatch.setattr(cli, "_ensure_sql_agent_metadata", lambda: None)
    monkeypatch.setattr(cli, "BackendAdapter", lambda: FakeBackendAdapter())
    monkeypatch.setattr(cli, "SupervisorAgent", lambda _adapter: supervisor)
    return supervisor


def test_parser_accepts_resume_answer_with_thread_id() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["--thread-id", "thread_1", "--resume-answer", "최근 6개월 월별 매출"])

    assert args.thread_id == "thread_1"
    assert args.resume_answer == "최근 6개월 월별 매출"
    assert args.query is None


def test_main_resume_answer_calls_resume_without_prompting_or_running(monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor = _patch_cli_runtime(monkeypatch, _state_result())
    monkeypatch.setattr(cli, "_write_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("resume mode must not prompt for query"))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "run.py",
            "--thread-id",
            "thread_1",
            "--resume-answer",
            "최근 6개월 월별 매출",
            "--no-output",
        ],
    )

    cli.main()

    assert supervisor.resume_calls == [("thread_1", {"answer": "최근 6개월 월별 매출"})]
    assert supervisor.run_calls == []


def test_resume_answer_with_query_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_runtime(monkeypatch, _state_result())
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "매출", "--resume-answer", "최근 6개월 월별 매출"],
    )

    with pytest.raises(SystemExit, match="resume-answer"):
        cli.main()


def test_resume_answer_with_datasource_id_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_runtime(monkeypatch, _state_result())
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "run.py",
            "--thread-id",
            "thread_1",
            "--datasource-id",
            "datasource_1",
            "--resume-answer",
            "최근 6개월 월별 매출",
        ],
    )

    with pytest.raises(SystemExit, match="datasource-id"):
        cli.main()


def test_resume_interrupt_result_uses_existing_json_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_cli_runtime(monkeypatch, _interrupt_result())
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "--thread-id", "thread_1", "--resume-answer", "최근 6개월 월별 매출", "--json"],
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "interrupt"
    assert payload["interrupt"]["question"] == "어떤 기간과 단위로 매출을 분석할까요?"
    assert payload["resume_payload"] == {"answer": "..."}


def test_resume_state_result_writes_outputs_with_state_user_query(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    written_queries: list[str] = []

    def fake_write_outputs(_adapter, _state, query, _output_dir):
        written_queries.append(query)
        return {
            "output_dir": Path("/tmp/daaa"),
            "summary": Path("/tmp/daaa/run_summary.json"),
            "generated_sql": Path("/tmp/daaa/generated_sql.sql"),
            "artifact_manifest": Path("/tmp/daaa/artifact_manifest.json"),
        }

    _patch_cli_runtime(monkeypatch, _state_result())
    monkeypatch.setattr(cli, "_write_outputs", fake_write_outputs)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "--thread-id", "thread_1", "--resume-answer", "최근 6개월 월별 매출", "--json"],
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert written_queries == ["매출\n추가 답변: 최근 6개월 월별 매출"]
    assert payload["thread_id"] == "thread_1"


def test_resume_non_result_raises_cli_compatible_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_runtime(monkeypatch, {"resumed": True})
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "--thread-id", "thread_1", "--resume-answer", "최근 6개월 월별 매출"],
    )

    with pytest.raises(RuntimeError, match="CLI-compatible"):
        cli.main()
