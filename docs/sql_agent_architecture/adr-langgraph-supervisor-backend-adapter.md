# ADR: LangGraph Supervisor with Fixed Backend Adapter

## Status

Accepted.

## Decision

Use LangGraph Main Supervisor as the top-level orchestrator. Keep `data_agent_backend` as a fixed platform dependency. Add a new Backend Adapter in the orchestration layer to consume backend runs, artifacts, approvals, policy, execution, workspace, and export capabilities without backend edits.

## Decision Drivers

- Main orchestration must belong to LangGraph.
- `data_agent_backend` must not be modified.
- Source DB must remain read-only.
- temp/reusable mart layers must be separate persistence boundaries.
- Agent handoff, validation, retry, and approval gates need explicit contracts.
- Data integrity validation and business-context validation must be separate.

## Considered Options

### Option A: LangGraph Supervisor + Backend Adapter + Redesigned Specialist Agents

Chosen target architecture. It satisfies orchestration ownership, backend non-modification, explicit contracts, and phased implementation.

### Option B: Minimal wrappers around existing SQL/EDA agents

Rejected as target architecture. It is faster for demo work but risks inheriting inconsistent state, storage, and validation boundaries.

### Option C: Backend-centric orchestration

Rejected. It violates the requirement that LangGraph owns orchestration.

### Option D: Hybrid Supervisor over wrapped existing SQL/EDA capability

Allowed only as a phase-1 delivery tactic. It is not the target architecture because compatibility shims can become long-term coupling.

## Consequences

- New orchestration/adapter layer becomes the main integration surface.
- Existing non-backend agents are references, not architectural constraints.
- More upfront contract design is required.
- Tests can focus on adapter contracts, graph transitions, policy checks, approval gates, artifact lineage, and smoke runs.
- Backend ingress/bootstrap must not be described or implemented as workflow ownership.
