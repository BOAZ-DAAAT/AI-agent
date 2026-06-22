# Test Spec: SQL Agent Architecture

## Backend Non-modification Guard

Pass if the implementation introduces no source edits under `data_agent_backend`. Test automation should inspect git diff paths and fail when any changed path starts with `data_agent_backend/`.

## Adapter Contract Tests

Pass criteria:

- `create_run()` creates a backend run record.
- `append_run_event()` records node events.
- `register_artifact()` returns an `ArtifactRef`.
- `request_approval()` creates a pending approval.
- `check_policy()` returns a policy decision for known actions.
- `run_sql_preview()` registers SQL and result artifacts for read-only SQL.

## LangGraph State and Transition Tests

Pass criteria:

- Success path terminates as `completed`.
- Mart request path terminates as `needs_user_approval`.
- Validation warning path records warning and continues.
- Retryable failure path records retry recommendation.
- Terminal failure path preserves run context and error state.

## SQL Read-only and Policy Tests

Pass criteria:

- `SELECT` and `WITH ... SELECT` are allowed.
- `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, and multi-statement SQL are blocked.
- Source DB writes are never performed by SQL preview.
- Row limit policy is evaluated before execution.

## Artifact Lineage Tests

Pass criteria:

- SQL artifact exists before preview/result artifact.
- Preview/result artifact is linked to SQL artifact.
- GE validation artifact references the result artifact it validates.
- Report artifact references upstream evidence artifacts.

## Approval-gated Mart Persistence Tests

Pass criteria:

- Permanent mart write is not attempted before approval.
- Mart candidate creates an approval request.
- Approved request allows materialization path to continue.
- Rejected request keeps only temp/artifact outputs.

## End-to-end Smoke Scenario

Query:

```text
월별 매출 추이를 분석하고, 반복 조회가 필요하면 데이터마트 저장을 제안해줘.
```

Pass criteria:

- Run is created.
- SQL artifact and preview artifact are created.
- GE-style JSON validation artifact is created.
- Central validation verdict is created.
- Mart storage pauses at `needs_user_approval`.
- Without approval, no permanent mart metadata is registered.

Non-mart query pass criteria:

- Run terminates as `completed`.
- Final report artifact is registered.
- Source DB remains read-only.
