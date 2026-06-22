# PRD: SQL Agent Architecture

## Target Users

- Analysts who ask natural-language business questions and expect SQL-backed answers.
- Data operators who need governed, approval-gated reusable datamarts.
- Engineers who need a buildable orchestration contract over the fixed `data_agent_backend` platform.

## Core User Journeys

- Query to report: user asks a business question, receives SQL-backed analysis and a final report.
- Query to reusable mart: user asks for repeatable analysis, receives an approval prompt before permanent mart storage.
- Review and audit: user or engineer traces report claims back to SQL, result, validation, and chart artifacts.

## MVP User Stories

- As an analyst, I can ask a natural-language question and get a textual report backed by SQL preview artifacts.
- As an analyst, I can see when permanent mart storage requires approval before it happens.
- As an engineer, I can run a smoke scenario without modifying `data_agent_backend`.
- As a reviewer, I can inspect artifact lineage from SQL to preview, validation, and report.

## Explicit Non-goals

- Full deployment and infrastructure automation.
- Full migration of legacy SQL/EDA agents.
- Advanced visualization rendering libraries.
- Personalized long-term memory behavior.
- Full mart refresh scheduler.
- Any schema or service changes under `data_agent_backend`.

## Acceptance Mapping

- LangGraph owns orchestration through Main Supervisor state, nodes, and terminal states.
- Backend remains a fixed dependency accessed only through Backend Adapter.
- Agent handoffs use the common result envelope.
- Validation is layered into GE, self-check, central business validation, and supervisor gate.
- Mart persistence is approval-gated and separated from source DB writes.
- State stores only IDs and small metadata.

## Release Gates

- Documentation artifacts exist: architecture, PRD, test spec, ADR.
- Backend Adapter contract tests pass.
- Supervisor transition tests pass for success, warning, approval, retryable failure, and terminal failure.
- SQL safety tests prove write SQL is blocked.
- E2E smoke creates SQL, preview, validation, report artifacts.
- Non-modification guard proves `data_agent_backend` files are not changed by the implementation.

## Definition of Done

- `data_agent_backend` remains unchanged.
- A minimal query-to-report flow runs through the new orchestration layer.
- Mart persistence pauses on approval before any permanent write.
- Artifacts are registered through existing backend artifact contracts.
- Tests document the contract and pass locally.
