"""bounded ReAct 루프 — 판단은 LLM, 실행·검증은 코드.

매 라운드 LLM이 look/compute/chart/finish 중 하나를 JSON으로 제안하고 코드가 실행해
관찰을 누적한다. finish 는 숫자 검증 게이트(verify.py)를 반드시 통과해야 종료되고,
실패 사유는 다음 라운드 피드백으로 들어간다. 라운드 캡·검증 실패 캡을 넘으면
증거 기반 보수 답변으로 폴백한다 — 지어낸 숫자로 끝나는 경우는 없다.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.insight.evidence import EvidencePack
from DATA_Analyst_Assistant_Agent.agents.insight.schemas import ChartEntry, InsightResult, ToolCall
from DATA_Analyst_Assistant_Agent.agents.insight.tools import run_chart, run_compute, run_look
from DATA_Analyst_Assistant_Agent.agents.insight.verify import build_evidence_corpus, verify_texts
from DATA_Analyst_Assistant_Agent.shared.llm import get_chat_model

MAX_ROUNDS = 8                                        # LLM 호출 상한 (배회 방지)
MAX_VERIFY_FAILS = 2                                  # finish 검증 실패 허용 횟수
MAX_FAIL_STREAK = 3                                   # 연속 실패(거부·반복·파싱) 시 조기 폴백 — 토큰 낭비 방지
_OBS_CHARS = 700                                      # 관찰 1건당 프롬프트 상한

# 게이트 거부 시 교정 힌트 — codegen 프롬프트 강화와 같은 교훈(거부당할 관용구는 미리/즉시 알려준다)
_GATE_FIX_HINT = ("→ 수정: 단일 표현식만. comprehension·lambda·query·eval·apply·merge 금지. "
                  "조건 필터는 불리언 마스크 + df.loc[...], '그룹 N건 이상' 필터는 "
                  "df.loc[df.groupby('그룹컬럼')['컬럼'].transform('size')>=N] 패턴을 쓰라.")


def run_insight_loop(pack: EvidencePack, llm: Any = None, out_dir: str = ".") -> InsightResult:
    # 표현식 작성 능력이 중요한 루프라 모델을 env 로 선택 가능하게 둔다
    # (INSIGHT_MODEL 미설정 시 LLM_MODEL — 스모크에서 gemini flash 는 lambda/apply 를 고집해 실패,
    #  codegen 과 같은 교훈: 코드 작성은 상위 모델이 안정적).
    llm = llm or get_chat_model(model=os.getenv("INSIGHT_MODEL") or None)
    observations: list[str] = []
    steps: list[dict[str, Any]] = []
    charts: list[ChartEntry] = []
    computes: list[Any] = []                          # 게이트 통과 계산 결과 — 인용 가능 증거로 편입
    seen_calls: set[str] = set()                      # 동일 호출 반복(배회) 감지용
    verify_fails = 0
    fail_streak = 0                                   # 연속 실패 카운트 (성공 시 리셋)

    for round_idx in range(MAX_ROUNDS):
        try:
            raw = llm.invoke(_build_prompt(pack, observations, round_idx)).content
        except Exception as exc:  # noqa: BLE001
            steps.append({"round": round_idx, "tool": "llm_error", "reason": str(exc), "ok": False})
            break
        call = _parse_tool_call(raw)
        if call is None:
            observations.append("[형식 오류] JSON 하나만 출력하라: {\"tool\":...,\"args\":...,\"reason\":...}")
            steps.append({"round": round_idx, "tool": "parse_error", "reason": "", "ok": False})
            fail_streak += 1
            if fail_streak >= MAX_FAIL_STREAK:
                break
            continue

        if call.tool == "finish":
            result, missing = _try_finish(pack, call.args, computes, charts, steps, round_idx)
            if result is not None:
                result.rounds = round_idx + 1
                return result
            verify_fails += 1
            if verify_fails > MAX_VERIFY_FAILS:
                break
            observations.append(
                f"[검증 실패] 다음 숫자가 증거에 없다: {missing} — 증거 요약·look·compute 결과에 있는 "
                "값만 쓰거나, 필요하면 compute 로 계산한 뒤 다시 finish 하라.")
            continue

        # 배회 방지: 완전히 같은 호출을 반복하면 실행하지 않고 강하게 넛지한다
        # (스모크에서 실제로 look 만 8라운드 반복하다 소진되는 패턴이 관찰됨).
        sig = f"{call.tool}:{json.dumps(call.args, ensure_ascii=False, sort_keys=True, default=str)}"
        if sig in seen_calls:
            observations.append("[반복 감지] 이미 했던 호출과 완전히 같다. 지금까지 성공한 compute 결과만으로도 "
                                "finish 할 수 있다 — 다른 표현식을 쓰거나 즉시 finish 하라.")
            steps.append({"round": round_idx, "tool": call.tool, "reason": call.reason,
                          "ok": False, "note": "repeat_call"})
            fail_streak += 1
            if fail_streak >= MAX_FAIL_STREAK:
                break                                  # 같은 실패를 고집 — 더 태워봐야 낭비, 조기 폴백
            continue
        seen_calls.add(sig)

        out = _dispatch(pack, call, out_dir)
        ok = bool(out.get("ok"))
        if ok and call.tool == "compute":
            computes.append(out)                       # 검증 corpus 에 편입 → 이 숫자는 인용 가능
        if ok and call.tool == "chart":
            computes.append(out.get("data_preview"))
            charts.append(ChartEntry(filename=out["filename"], title=out["title"],
                                     kind=out["kind"], local_path=out["local_path"]))
        note = json.dumps(out, ensure_ascii=False, default=str)[:_OBS_CHARS]
        if not ok and "gate_rejected" in str(out.get("error", "")):
            note += f" {_GATE_FIX_HINT}"               # 즉시 교정 힌트 (codegen retry 프롬프트와 같은 원리)
        observations.append(f"[{round_idx + 1}] {call.tool}({call.reason}) → {note}")
        steps.append({"round": round_idx, "tool": call.tool, "reason": call.reason, "ok": ok,
                      "note": out.get("error", "") if not ok else ""})
        fail_streak = 0 if ok else fail_streak + 1
        if fail_streak >= MAX_FAIL_STREAK:
            break

    return _fallback(pack, steps, charts)


def _dispatch(pack: EvidencePack, call: ToolCall, out_dir: str) -> dict:
    if call.tool == "look":
        return run_look(pack, call.args)
    if call.tool == "compute":
        return run_compute(pack, call.args)
    if call.tool == "chart":
        return run_chart(pack, call.args, out_dir)
    return {"ok": False, "error": f"unknown_tool: {call.tool} (look|compute|chart|finish 중 하나)"}


def _try_finish(pack: EvidencePack, args: dict, computes: list, charts: list[ChartEntry],
                steps: list[dict], round_idx: int) -> tuple[InsightResult | None, list[str]]:
    """finish 제안을 강제 게이트에 통과시킨다. 통과 못 하면 (None, 없는 숫자들)."""
    answer = str(args.get("answer", "")).strip()
    key_insights = [str(s) for s in (args.get("key_insights") or [])]
    action_plan = [str(s) for s in (args.get("action_plan") or [])]
    limitations = [str(s) for s in (args.get("limitations") or [])]
    if not answer:
        steps.append({"round": round_idx, "tool": "finish", "reason": "", "ok": False, "note": "빈 answer"})
        return None, ["(answer가 비어 있음)"]

    numbers, corpus = build_evidence_corpus(pack, computes)
    ok, missing = verify_texts([answer, *key_insights, *action_plan], numbers, corpus)
    steps.append({"round": round_idx, "tool": "finish", "reason": "", "ok": ok,
                  "note": "" if ok else f"unverified: {missing}"})
    if not ok:
        return None, missing

    if action_plan:                                    # 권고는 사실이 아니다 — caveat 강제 부착
        limitations.append("액션 플랜은 관찰된 데이터 기반 권고이며, 인과 검증은 별도로 필요합니다.")
    return InsightResult(answer=answer, key_insights=key_insights, action_plan=action_plan,
                         limitations=limitations, charts=charts, steps=steps), []


def _fallback(pack: EvidencePack, steps: list[dict], charts: list[ChartEntry]) -> InsightResult:
    """검증 실패/라운드 소진 — 이미 검증된 상류 요약으로 보수 답변 (LLM 생성 숫자 없음)."""
    answer = (pack.eda.get("final_summary")
              or pack.analysis.get("method_summary")
              or f"'{pack.user_question}'에 대한 집계 결과를 표로 제공합니다. 세부 수치는 결과 테이블을 참고하세요.")
    return InsightResult(
        answer=str(answer), charts=charts, steps=steps, fallback_used=True,
        limitations=["답변 검증을 통과하지 못해 상류 요약 기반의 보수적 답변으로 대체되었습니다."],
        rounds=len(steps))


# ─────────────────────────────
# 프롬프트 / 파싱
# ─────────────────────────────
def _build_prompt(pack: EvidencePack, observations: list[str], round_idx: int) -> str:
    eda_brief = {k: pack.eda.get(k) for k in ("final_summary", "cautions", "out_of_domain") if pack.eda.get(k)}
    # codegen 이 이미 계산해 둔 질문 맞춤 결과(adhoc_analysis)는 가장 강한 증거라 기본 노출한다
    adhoc = (pack.eda.get("statistical_metadata") or {}).get("adhoc_analysis")
    if adhoc:
        eda_brief["adhoc_analysis"] = {k: adhoc.get(k) for k in ("intent", "result") if adhoc.get(k)}
    obs_text = "\n".join(observations[-6:]) or "(아직 없음 — 첫 라운드)"
    pressure = ("\n⚠️ 라운드가 거의 소진됐다. look 을 더 하지 말고, 필요하면 compute 한 번 뒤 즉시 finish 하라."
                if round_idx >= MAX_ROUNDS - 3 else "")
    return f"""너는 데이터 분석 파이프라인 '마지막'의 인사이트 에이전트다. 상류(SQL/EDA/분석)가 만든
검증된 증거를 읽고 사용자 질문에 직답하는 결론을 구성한다. 새 통계분석을 하는 자리가 아니다.

[사용자 질문] {pack.user_question}
[경로] {pack.route_kind}
[생성 SQL] {(pack.generated_sql or "(없음)")[:400]}
[결과 테이블 요약] {json.dumps(pack.table_summary, ensure_ascii=False, default=str)[:1800]}
[EDA 요약] {json.dumps(eda_brief, ensure_ascii=False, default=str)[:1200] or "(없음)"}
[분석 결과] {json.dumps(pack.analysis, ensure_ascii=False, default=str)[:1200] or "(없음)"}

[지금까지 관찰]
{obs_text}

[도구 — 반드시 하나만 JSON으로 제안]
- look: 증거 더 보기. args={{"target":"table|eda|analysis|sql","path":"eda/analysis 내부 점표기 경로(선택, 예: statistical_metadata.group_comparison)"}}
- compute: 보조 계산. args={{"expression":"df 단일 pandas 표현식"}}
  · 허용: 증감률·차이·비율·top/bottom·정렬·간단 집계·reshape / 금지: 회귀·군집·검정·인과·예측·외부데이터
  · 게이트가 거부하니 쓰지 마라: comprehension·lambda·query·eval·apply·merge·파일IO. df·pd·np 만, 단일 표현식만.
  · 조건 필터는 불리언 마스크 + df.loc[...]. '그룹 N건 이상' 필터는
    df.loc[df.groupby('그룹컬럼')['컬럼'].transform('size')>=N] 패턴을 쓰라.
  · 'X 이상 비율' 류는 lambda 없이 df['수치'].ge(X).groupby(df['그룹컬럼']).mean() 패턴을 쓰라.
- chart: 답을 뒷받침하는 차트 주문(렌더는 코드가 함). args={{"expression":"차트 데이터 표현식","kind":"line|bar|table","title":"제목"}}
- finish: 답 제출. args={{"answer":"질문 직답 1~3문장","key_insights":["핵심 인사이트"],"action_plan":["근거 있는 권고(없으면 빈 배열)"],"limitations":["해석 한계"]}}

[정책 — 어기면 finish 가 거부된다]
- 위 증거 요약·look·compute 결과에 등장한 숫자만 써라. 새 숫자가 필요하면 compute 로 계산하라.
  (예: df.groupby('범주컬럼')['수치컬럼'].mean().nlargest(10) — look 을 반복하는 대신 계산하라)
- 상관을 인과로 단정하지 마라("~때문에" 대신 "~와 연관").
- action_plan 은 증거로 뒷받침될 때만 채워라. 억지로 만들지 마라.
- answer 가 비교·순위·추세라면 finish 전에 그것을 증명하는 chart 를 1개 만들어라. 그 외엔 생략 가능.
- 남은 라운드: {MAX_ROUNDS - round_idx}. 증거가 충분하면 바로 finish 하라.{pressure}

JSON 만 출력하라: {{"tool":"look|compute|chart|finish","args":{{...}},"reason":"한 문장"}}"""


def _parse_tool_call(raw: str) -> ToolCall | None:
    """LLM 출력에서 첫 JSON 오브젝트를 뽑아 ToolCall 로. 실패 시 None(라운드 소비)."""
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    if start < 0:
        return None
    for end in range(len(text), start, -1):            # 뒤에서부터 닫는 지점 탐색
        try:
            parsed = json.loads(text[start:end])
            break
        except json.JSONDecodeError:
            continue
    else:
        return None
    if not isinstance(parsed, dict) or "tool" not in parsed:
        return None
    try:
        return ToolCall.model_validate(parsed)
    except Exception:  # noqa: BLE001
        return None
