"""codegen 안전 게이트 — LLM이 생성한 pandas 표현식을 실행 전에 정적 검사한다.

원칙: LLM은 표현식을 '제안'만 하고, 실행 여부는 이 게이트(코드)가 결정한다.
- 단일 표현식만(ast.parse mode='eval') → 다중문·반복문·무한루프 차단
- 노출 이름은 df/pd/np 뿐 → builtins·os·open·__import__ 등 접근 차단
- dunder 속성 접근 차단 → 샌드박스 이스케이프(df.__class__...) 차단
- IO·eval·행폭발 메서드 denylist → 파일접근·코드실행·리소스폭발 차단
- df[...] 문자열 첨자 컬럼 실존 대조

⚠️ 하드보장은 아님(denylist의 구조적 한계):
   - 리소스 폭발: 흔한 메서드만 막고 우회 가능. OOM 계측 쌓이면 v2 서브프로세스 승격.
   - IO/코드실행: 직접 IO 함수명(pandas·numpy) + 서브모듈 진입점(lib/core/io 등)까지 막지만,
     알려지지 않은 introspection 경로를 정적으로 100% 보장하진 못함 → 완전 격리는 v2 서브프로세스.
   v1은 최후수단만 발동(빈도 낮음) + 화이트리스트 이름제한 + builtins 제거 심층방어로 잔여 리스크를 낮춘다.
"""

from __future__ import annotations

import ast

from pydantic import BaseModel, Field, field_validator

# 노출 허용 이름(이 외의 bare name은 전부 거부 → builtins/os/open 접근 차단)
_ALLOWED_NAMES = {"df", "pd", "np"}

# 행/컬럼 폭발 가능 메서드 (AST상 안전해도 N² 폭발)
_DENY_EXPLODE = {"merge", "join", "explode", "repeat", "get_dummies",
                 "pivot", "pivot_table", "unstack", "reindex", "combine"}
# 파일/DB IO 함수 (read_* 접두는 별도 처리)
# ⚠️ np는 허용 이름이라 np.load/fromfile/loadtxt 등 numpy IO가 이름검사를 통과한다.
#    이름 패턴이 pandas의 to_*/read_*와 완전히 달라 반드시 명시 차단해야 한다.
#    특히 np.load는 pickle 역직렬화(allow_pickle) → 임의 코드 실행 경로라 최우선 차단.
_DENY_IO = {
    # pandas 파일/DB IO
    "to_csv", "to_pickle", "to_parquet", "to_sql", "to_excel",
    "to_json", "to_hdf", "to_feather", "to_clipboard", "to_gbq", "to_stata",
    "ExcelFile", "HDFStore",
    # numpy 파일 IO
    "load", "save", "savez", "savez_compressed", "fromfile", "tofile",
    "loadtxt", "savetxt", "genfromtxt", "fromregex", "memmap", "DataSource",
}
# 문자열을 코드로 실행하는 경로
_DENY_EXEC = {"eval", "query"}
# 서브모듈 순회로 IO/시스템에 도달하는 진입점 차단 (예: np.lib.npyio.*, pd.io.*, ctypeslib)
# 계산용 표현식은 이 속성들을 쓸 일이 없다(df/pd/np의 계산 함수만 필요).
_DENY_TRAVERSAL = {"lib", "core", "io", "ctypeslib", "testing", "distutils"}
# 대형 배열 생성 (크기 인자로 메모리 폭발 — np.ones((10**9,)) 등). 집계 표현식은 쓸 일이 없다.
# ⚠️ "empty"는 df.empty(빈 DF 여부 체크)와 충돌하므로 제외한다.
_DENY_ALLOC = {"ones", "zeros", "full", "arange", "linspace", "logspace", "geomspace",
               "eye", "identity", "tile", "broadcast_to", "meshgrid",
               "ones_like", "zeros_like", "full_like", "fromiter", "frombuffer"}
_DENY_ATTRS = _DENY_EXPLODE | _DENY_IO | _DENY_EXEC | _DENY_TRAVERSAL | _DENY_ALLOC

_VALID_SHAPES = {"scalar", "series", "frame"}


_VALID_HINTS = {"bar", "line", "grouped_bar"}            # None = 차트 부적합(LLM 판단)


class CodegenRequest(BaseModel):
    """LLM이 발행하는 구조적 신호(자유 문자열 실행 아님)."""
    intent: str                                          # 자연어: 뭘 계산하려는지
    target_columns: list[str] = Field(default_factory=list)
    expression: str                                      # df 위 단일 pandas 표현식
    expected_shape: str = "scalar"                       # scalar | series | frame
    chart_hint: str | None = None                        # bar | line | grouped_bar | None (LLM이 차트 종류 판단)

    @field_validator("chart_hint", mode="before")
    @classmethod
    def _normalize_chart_hint(cls, v):
        # LLM이 '차트 없음'을 JSON null 대신 문자열 "null"/"none"/""로 뱉는 경우를 None으로 정규화한다.
        # (스칼라 답처럼 차트가 부적합한 정당한 케이스가 invalid_chart_hint로 잘못 거부되는 걸 막음.)
        if isinstance(v, str) and v.strip().lower() in {"null", "none", ""}:
            return None
        return v


class GateResult(BaseModel):
    ok: bool
    reason: str = ""


def _subscript_strings(node: ast.Subscript) -> list[str]:
    """df["col"] / df[["a","b"]] 처럼 첨자로 쓰인 문자열만 뽑는다(메서드 인자는 제외)."""
    out: list[str] = []
    sl = node.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        out.append(sl.value)
    elif isinstance(sl, ast.List):
        for e in sl.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                out.append(e.value)
    return out


def validate_expression(expression: str, allowed_columns: list[str]) -> GateResult:
    """표현식을 정적 검사한다. 통과 시 GateResult(ok=True)."""
    # 1) 단일 표현식만 — 다중문/할당/반복문이면 eval 모드 파싱이 SyntaxError
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return GateResult(ok=False, reason=f"not_single_expression: {e.msg}")

    cols = set(allowed_columns)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return GateResult(ok=False, reason="import_not_allowed")
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp, ast.Lambda)):
            return GateResult(ok=False, reason="comprehension_or_lambda_not_allowed")
        if isinstance(node, ast.NamedExpr):                      # 왈러스(:=) 대입
            return GateResult(ok=False, reason="assignment_not_allowed")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            return GateResult(ok=False, reason=f"name_not_allowed: {node.id}")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") or attr.endswith("__"):
                return GateResult(ok=False, reason=f"dunder_attr_not_allowed: {attr}")
            if attr in _DENY_ATTRS or attr.startswith("read_"):
                return GateResult(ok=False, reason=f"denied_method: {attr}")
        if isinstance(node, ast.Subscript):
            # 원본 df에서 직접 꺼내는 첨자만 검증한다(df["col"] / df[["a","b"]]).
            # 중간·파생 결과의 첨자(df.groupby(...).agg(prop=...)["prop"])는 원본에 없는
            # '계산 컬럼'이라 검증 대상이 아니다 — 파생 컬럼 계산이 codegen의 본질이므로 막으면 안 된다.
            # (진짜 오타난 원본컬럼은 실행 시 KeyError→out_of_domain으로 걸려 안전은 그대로.)
            if isinstance(node.value, ast.Name) and node.value.id == "df":
                for s in _subscript_strings(node):
                    if s not in cols:
                        return GateResult(ok=False, reason=f"unknown_column: {s}")

    return GateResult(ok=True)


def validate_request(req: CodegenRequest, allowed_columns: list[str]) -> GateResult:
    """CodegenRequest 전체 검증: expected_shape·선언 컬럼 실존 + 표현식 게이트."""
    if req.expected_shape not in _VALID_SHAPES:
        return GateResult(ok=False, reason=f"invalid_expected_shape: {req.expected_shape}")
    if req.chart_hint is not None and req.chart_hint not in _VALID_HINTS:
        return GateResult(ok=False, reason=f"invalid_chart_hint: {req.chart_hint}")
    # target_columns는 LLM의 '선언'일 뿐 실행되지 않으므로(오직 expression만 eval) 보안과 무관하다.
    # 여기엔 파생·중간 컬럼이 섞여 들어올 수 있어(codegen의 본질) 하드 거부하지 않는다.
    # 실제 컬럼 실존 검증은 expression의 df 직접 첨자에서 수행한다(validate_expression).
    return validate_expression(req.expression, allowed_columns)
