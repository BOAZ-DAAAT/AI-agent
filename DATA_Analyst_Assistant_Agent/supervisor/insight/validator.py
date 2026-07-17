"""내부 validator — "역할을 잘 수행했나"를 심사하는 품질 게이트 (팀 컨벤션의 서브에이전트 validator).

verify.py(결정론 숫자 게이트)와 역할이 다르다:
  verify   = "답변 속 숫자가 증거에 실존하는가" (코드 대조 — 환각 차단)
  validator = "질문에 직답했는가 · 차트가 답을 뒷받침하는가 · 인과 과장은 없는가" (LLM 심사 — 품질)

둘 다 finish 뒤에 순서대로 걸린다: 숫자 게이트 통과 → 품질 심사 → 통과해야 종료.
심사 실패 시 feedback 이 루프 관찰로 돌아가 재작성된다(최대 1회 — 배회 방지).
LLM 오류·파싱 실패는 fail-open(통과) — 숫자 검증까지 끝난 답을 심사기 장애로 버리지 않는다.
"""

from __future__ import annotations

import json
import re
from typing import Any


def validate_result(llm: Any, question: str, answer: str, key_insights: list[str],
                    charts: list, action_plan: list[str]) -> tuple[bool, str]:
    """finish 결과의 품질 심사. 반환: (통과, 재작성 피드백)."""
    chart_desc = [f"[{c.kind}] {c.title}" for c in charts] or ["(차트 없음)"]
    prompt = f"""너는 데이터 분석 답변의 품질 검증자다. 아래 답변이 '역할을 잘 수행했는지'만 심사하라.
새 사실·숫자를 추가하지 마라(그건 네 일이 아니다).

[사용자 질문] {question}
[답변] {answer}
[핵심 인사이트] {json.dumps(key_insights, ensure_ascii=False)}
[차트] {json.dumps(chart_desc, ensure_ascii=False)}
[액션 플랜] {json.dumps(action_plan, ensure_ascii=False)}

심사 기준 (하나라도 심각하게 어기면 retry):
1. 직답성 — 질문이 물은 것에 정면으로 답했는가(동문서답 아닌가).
2. 차트-답변 대응 — 차트 제목/종류가 답변의 핵심 주장과 일치하는가
   (예: 답은 '금액'인데 차트가 '건수'면 불일치). 답이 계산 불가 선언이면 차트 없음이 정상.
3. 인과 과장 — 상관을 "~때문"으로 단정하지 않았는가.
4. 액션 근거 — 액션 플랜이 답변 내용에서 유도 가능한가(뜬금없지 않은가).

반드시 JSON만 출력: {{"verdict": "pass" 또는 "retry", "feedback": "retry면 무엇을 어떻게 고칠지 한두 문장"}}"""
    try:
        raw = llm.invoke(prompt).content
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(m.group()) if m else {}
    except Exception:  # noqa: BLE001
        return True, ""                                # fail-open — 심사기 장애로 검증된 답을 버리지 않음
    if str(parsed.get("verdict", "pass")).lower() == "retry":
        return False, str(parsed.get("feedback", "품질 기준 미달 — 질문에 더 직접적으로 답하라"))
    return True, ""
