# Orchestration State Variables

This document explains the state and artifact contracts used by the SQL agent orchestration flow. The supervisor may branch by route, retry after recoverable failures, or stop for approval, but every specialist agent communicates through artifact ids and backend adapter reads rather than direct in-memory object passing.

## Core Flow

The intended logical loop is:

1. Load run context and datasource context.
2. Plan the user question and decide the route.
3. Generate safe analytical SQL.
4. Execute SQL through the backend adapter.
5. Validate SQL safety, result shape, and downstream artifacts.
6. Continue to EDA, analysis, visualization, report, approval, retry, or finalization based on validation output.

Routes are not forced into one fixed chain. A simple route may go `sql_agent -> report_agent`; an EDA route may go `sql_agent -> eda_agent -> report_agent`; a trend route may go `sql_agent -> analysis_agent -> visualization_agent -> report_agent`.

## Variables

### datasource_id

Backend-owned datasource identifier selected for the run. Orchestration passes this id to backend methods but must not read or store raw credentials such as passwords. When present, SQL execution should use `BackendAdapter.run_sql_preview(..., datasource_id=datasource_id)`.

### catalog_summary

Credential-free schema summary for the selected datasource. It should contain only analytical planning metadata such as table names, column names, data types, and nullable flags. It is safe for planner prompts because it does not include secrets.

### user_query

The original natural-language request from the user. It is the primary intent signal used by the supervisor route planner and SQL planner.

### goal

Supervisor-normalized objective for the run. This is broader than `user_query` and describes the route outcome, such as a simple SQL answer, EDA profile, trend analysis, or mart review.

### route_kind

The supervisor's route classification. Current values include `simple`, `eda`, `trend`, `mart`, and `comprehensive`. The route controls which specialist agents are scheduled, but validation can still stop, warn, or request approval.

### generated_sql

The final SQL string produced by the planner. The SQL agent stores this in state and also registers the SQL plan as an artifact. Report and validation agents should prefer artifact reads when available so the generated SQL remains auditable.

### artifact_ids

Dictionary keyed by agent or artifact group name. Values are artifact id lists registered in the backend. Example keys include `sql_agent`, `eda_agent`, `analysis_agent`, `visualization_agent`, `report_agent`, `validation_agent`, and `mart_metadata`.

Agents should pass evidence by artifact id only. Downstream agents read artifact content with `BackendAdapter.read_artifact_text(artifact_id)` and metadata with `BackendAdapter.get_artifact(artifact_id)`.

### retry_context

Recoverable failure context from a prior attempt. It should include the failing step, error code, message, query, datasource id, and a concise hint for the next planner call. It should not include credentials or raw connection strings.

### error_state

Terminal or recoverable error details for the current run. It is used by the supervisor and validation layer to decide whether to retry, ask for clarification, request approval, or fail the run.

## Shared Models

### AgentEnvelope

Standard return object for every specialist agent. It contains agent status, a summary, artifact refs, validation checks, retry hints, approval requirements, context refs, and the next handoff target. The supervisor records `AgentEnvelope.artifact_refs` into `artifact_ids`.

### ValidationBlock

Structured local validation result attached to an `AgentEnvelope`. It contains local checks, integrity artifact refs, and business flags. Central validation turns this into a run-level verdict: pass, warn, approval required, or fail.

### ArtifactRef

Lightweight reference to a backend artifact. It carries the artifact id, type, URI, content hash, and preview metadata. It should be enough for routing and UI previews; full content should be read through the backend adapter.

## Artifact Contracts

### SQL Agent

Registers:

- `sql_result`: CSV content from SQL execution.
- `sql_plan`: JSON payload with selected tables, columns, reasoning, and generated SQL.
- GE-style validation JSON for result integrity.

### EDA Agent

Reads SQL result CSV artifacts from `artifact_ids["sql_agent"]`, loads them into pandas DataFrames, and writes a `data_profile` artifact with row counts, columns, dtypes, null counts, unique counts, numeric summaries, categorical top values, quality status, and key issues.

### Analysis Agent

Reads SQL result CSV artifacts and EDA profile artifacts. It writes `analysis_result.json` with method summary, key findings, limitations, source artifact ids, and data quality notes.

### Visualization Agent

Reads SQL result CSV artifacts and analysis artifacts. It writes a Vega-Lite JSON artifact. Version 1 does not render an image; the chart artifact is the JSON spec.

### Report Agent

Reads SQL plan, EDA, analysis, and visualization artifacts. It writes a Markdown report containing the user question, generated SQL, EDA summary, analysis findings, visualization artifact, evidence ids, limitations, and next actions.

### Validation Agent

Checks upstream artifact existence, SQL safety, SQL result row counts, EDA quality status, and report evidence coverage. Findings are marked retryable or non-retryable so the supervisor can branch without assuming every issue is terminal.
