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


class SequencedFakeSupervisor(FakeSupervisor):
    def __init__(self, *, run_result: Any, resume_results: list[Any]) -> None:
        super().__init__(run_result)
        self.resume_results = list(resume_results)

    def resume(self, thread_id: str, resume_payload: dict[str, Any]):
        self.resume_calls.append((thread_id, resume_payload))
        return self.resume_results.pop(0)


def _state_result(
    user_query: str = "최근 6개월 월별 매출",
    *,
    final_answer: str = "",
    generated_sql: str = "",
) -> SupervisorRunResult:
    return SupervisorRunResult(
        kind="state",
        state=OrchestrationState(
            run_id="run_resumed_001",
            thread_id="thread_1",
            user_query=user_query,
            terminal_state=SupervisorTerminalState.completed,
            final_answer=final_answer,
            generated_sql=generated_sql,
        ),
    )


def _approval_state_result() -> SupervisorRunResult:
    return SupervisorRunResult(
        kind="state",
        state=OrchestrationState(
            run_id="run_approval_001",
            thread_id="thread_1",
            user_query="월별 매출 추이를 분석해줘",
            terminal_state=SupervisorTerminalState.needs_user_approval,
            approval_ids=["run_approval_001:sql_agent:approval"],
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


def _patch_cli_runtime_with_supervisor(monkeypatch: pytest.MonkeyPatch, supervisor: FakeSupervisor) -> FakeSupervisor:
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


def test_parser_uses_no_default_thread_id_for_new_runs() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["월별 매출 추이를 분석해줘"])

    assert args.thread_id is None


def test_parser_accepts_approval_resume_aliases() -> None:
    parser = cli.build_parser()

    approve_args = parser.parse_args(["--thread-id", "thread_1", "--approve"])
    resume_approved_args = parser.parse_args(["--thread-id", "thread_1", "--resume-approved"])

    assert approve_args.resume_approved is True
    assert resume_approved_args.resume_approved is True


def test_parser_accepts_no_interactive_for_scripted_interrupt_handling() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["매출", "--no-interactive"])

    assert args.interactive is False


def test_main_new_run_generates_unique_thread_id(monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor = _patch_cli_runtime(monkeypatch, _state_result())
    monkeypatch.setattr(cli, "_new_thread_id", lambda: "thread_cli_001")
    monkeypatch.setattr(cli, "_write_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "월별 매출 추이를 분석해줘", "--no-output"],
    )

    cli.main()

    assert supervisor.run_calls == [
        {
            "query": "월별 매출 추이를 분석해줘",
            "thread_id": "thread_cli_001",
            "datasource_id": None,
        }
    ]


def test_main_auto_interactive_clarification_prompts_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    supervisor = SequencedFakeSupervisor(run_result=_interrupt_result(), resume_results=[_state_result()])
    _patch_cli_runtime_with_supervisor(monkeypatch, supervisor)
    monkeypatch.setattr(cli, "_new_thread_id", lambda: "thread_cli_001")
    monkeypatch.setattr(cli, "_write_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "최근 6개월 월별 매출")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "매출", "--no-output"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "Human Input Required" in output
    assert supervisor.run_calls == [{"query": "매출", "thread_id": "thread_cli_001", "datasource_id": None}]
    assert supervisor.resume_calls == [("thread_1", {"answer": "최근 6개월 월별 매출"})]


def test_main_no_interactive_keeps_interrupt_resume_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    supervisor = _patch_cli_runtime(monkeypatch, _interrupt_result())
    monkeypatch.setattr(cli, "_new_thread_id", lambda: "thread_cli_001")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("--no-interactive must not prompt"))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "매출", "--no-interactive"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "resume command:" in output
    assert supervisor.resume_calls == []


def test_main_auto_interactive_approval_prompts_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    supervisor = SequencedFakeSupervisor(run_result=_approval_state_result(), resume_results=[_state_result()])
    _patch_cli_runtime_with_supervisor(monkeypatch, supervisor)
    monkeypatch.setattr(cli, "_write_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "--thread-id", "thread_1", "월별 매출 추이를 분석해줘", "--no-output"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "Approval Resume" in output
    assert supervisor.resume_calls == [("thread_1", {"approved": True})]


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


def test_main_approval_resume_calls_resume_without_prompting_or_running(monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor = _patch_cli_runtime(monkeypatch, _state_result())
    monkeypatch.setattr(cli, "_write_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("approval resume mode must not prompt for query"))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "run.py",
            "--thread-id",
            "thread_1",
            "--approve",
            "--no-output",
        ],
    )

    cli.main()

    assert supervisor.resume_calls == [("thread_1", {"approved": True})]
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


@pytest.mark.parametrize("argv", [["run.py", "--resume-answer", "최근 6개월 월별 매출"], ["run.py", "--approve"]])
def test_resume_modes_without_thread_id_exit(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    _patch_cli_runtime(monkeypatch, _state_result())
    monkeypatch.setattr(cli.sys, "argv", argv)

    with pytest.raises(SystemExit, match="thread-id"):
        cli.main()


def test_approval_resume_with_resume_answer_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cli_runtime(monkeypatch, _state_result())
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "--thread-id", "thread_1", "--approve", "--resume-answer", "최근 6개월 월별 매출"],
    )

    with pytest.raises(SystemExit, match="cannot be used together"):
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
    assert "--thread-id thread_1" in payload["resume_command"]
    assert "--resume-answer" in payload["resume_command"]


def test_interrupt_text_output_includes_copyable_resume_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_cli_runtime(monkeypatch, _interrupt_result())
    monkeypatch.setattr(cli, "_new_thread_id", lambda: "thread_cli_001")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "매출", "--show-sql", "--no-open"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "resume command:" in output
    assert "--thread-id thread_1" in output
    assert "--resume-answer" in output
    assert "--show-sql" in output
    assert "--no-open" in output


def test_approval_waiting_json_output_includes_approval_resume_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_cli_runtime(monkeypatch, _approval_state_result())
    monkeypatch.setattr(cli, "_write_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "--thread-id", "thread_1", "--approve", "--json", "--no-output"],
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["terminal_state"] == "needs_user_approval"
    assert payload["resume_payload"] == {"approved": True}
    assert "--thread-id thread_1" in payload["resume_command"]
    assert "--approve" in payload["resume_command"]


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
    assert written_queries == ["최근 6개월 월별 매출"]
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


def test_state_summary_always_includes_final_answer() -> None:
    state = _state_result(final_answer="분석이 완료되었습니다.").state
    assert state is not None

    summary = cli._state_summary(state)

    assert summary["final_answer"] == "분석이 완료되었습니다."


def test_json_output_includes_final_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_cli_runtime(monkeypatch, _state_result(final_answer="분석이 완료되었습니다."))
    monkeypatch.setattr(cli.sys, "argv", ["run.py", "매출", "--json", "--no-output"])

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["final_answer"] == "분석이 완료되었습니다."


@pytest.mark.parametrize(
    ("final_answer", "shows_section"),
    [("분석이 완료되었습니다.", True), ("", False)],
)
def test_text_output_shows_final_answer_section_only_when_present(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    final_answer: str,
    shows_section: bool,
) -> None:
    _patch_cli_runtime(monkeypatch, _state_result(final_answer=final_answer))
    monkeypatch.setattr(cli.sys, "argv", ["run.py", "매출", "--no-output"])

    cli.main()

    output = capsys.readouterr().out
    assert ("=== Final Answer ===" in output) is shows_section
    if final_answer:
        assert final_answer in output


def test_write_outputs_includes_final_answer_and_clears_stale_sql(tmp_path: Path) -> None:
    state = _state_result(final_answer="계획 생성에 실패했습니다.").state
    assert state is not None
    sql_path = tmp_path / "generated_sql.sql"
    sql_path.write_text("SELECT stale_data", encoding="utf-8")

    outputs = cli._write_outputs(FakeBackendAdapter(), state, state.user_query, tmp_path)

    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["final_answer"] == "계획 생성에 실패했습니다."
    assert outputs["generated_sql"].read_bytes() == b""


def test_write_outputs_writes_generated_sql_without_modification(tmp_path: Path) -> None:
    generated_sql = "SELECT month, SUM(amount)\nFROM sales\nGROUP BY month;\n"
    state = _state_result(generated_sql=generated_sql).state
    assert state is not None

    outputs = cli._write_outputs(FakeBackendAdapter(), state, state.user_query, tmp_path)

    assert outputs["generated_sql"].read_text(encoding="utf-8") == generated_sql


def test_no_output_does_not_create_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_cli_runtime(monkeypatch, _state_result())
    output_dir = tmp_path / "disabled"
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["run.py", "매출", "--no-output", "--output-dir", str(output_dir)],
    )

    cli.main()

    assert not output_dir.exists()
