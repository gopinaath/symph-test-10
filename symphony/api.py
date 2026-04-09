"""FastAPI REST API for the Symphony dashboard and external integrations."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from symphony.observability import PubSub
from symphony.orchestrator import Orchestrator, OrchestratorSnapshot, RunningEntry, RetryEntry


def create_app(
    orchestrator: Orchestrator | None = None,
    pubsub: PubSub | None = None,
) -> FastAPI:
    """Create the FastAPI app with orchestrator dependency."""

    app = FastAPI(title="Symphony", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.orchestrator = orchestrator
    app.state.pubsub = pubsub

    @app.get("/api/v1/state")
    async def get_state() -> dict[str, Any]:
        """Get full orchestrator state snapshot."""
        orch: Orchestrator | None = app.state.orchestrator
        if orch is None:
            raise HTTPException(status_code=503, detail="Orchestrator unavailable")
        try:
            snapshot = orch.snapshot()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Snapshot unavailable") from exc
        return _serialize_snapshot(snapshot)

    @app.get("/api/v1/{issue_identifier}")
    async def get_issue(issue_identifier: str) -> dict[str, Any]:
        """Get details for a specific running issue."""
        orch: Orchestrator | None = app.state.orchestrator
        if orch is None:
            raise HTTPException(status_code=503, detail="Orchestrator unavailable")
        try:
            snapshot = orch.snapshot()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Snapshot unavailable") from exc

        # Search running entries
        for _issue_id, entry in snapshot.running.items():
            if entry.issue.identifier == issue_identifier:
                return _serialize_running_entry(entry)

        # Search retry queue
        for _issue_id, retry in snapshot.retry_queue.items():
            if retry.issue.identifier == issue_identifier:
                return _serialize_retry_entry(retry)

        raise HTTPException(status_code=404, detail=f"Issue {issue_identifier} not found")

    @app.post("/api/v1/refresh")
    async def refresh() -> dict[str, str]:
        """Trigger an immediate poll cycle."""
        orch: Orchestrator | None = app.state.orchestrator
        if orch is None:
            raise HTTPException(status_code=503, detail="Orchestrator unavailable")
        orch.request_refresh()
        return {"status": "refresh_requested"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _serialize_snapshot(snapshot: OrchestratorSnapshot) -> dict[str, Any]:
    """Serialize an OrchestratorSnapshot to JSON-safe dict."""
    return {
        "running": {
            k: _serialize_running_entry(v)
            for k, v in snapshot.running.items()
        },
        "retry_queue": {
            k: _serialize_retry_entry(v)
            for k, v in snapshot.retry_queue.items()
        },
        "completed_count": len(snapshot.completed),
        "codex_totals": snapshot.codex_totals,
        "codex_rate_limits": snapshot.codex_rate_limits,
        "poll_countdown_ms": snapshot.poll_countdown_ms,
        "poll_checking": snapshot.poll_checking,
    }


def _serialize_running_entry(entry: RunningEntry) -> dict[str, Any]:
    return {
        "issue_id": entry.issue.id,
        "identifier": entry.issue.identifier,
        "title": entry.issue.title,
        "state": entry.issue.state,
        "session_id": entry.session_id,
        "turn_count": entry.turn_count,
        "last_event": entry.last_event,
        "last_event_at": entry.last_event_at.isoformat() if entry.last_event_at else None,
        "started_at": entry.started_at.isoformat() if entry.started_at else None,
        "input_tokens": entry.input_tokens,
        "output_tokens": entry.output_tokens,
        "total_tokens": entry.total_tokens,
        "worker_host": entry.worker_host,
        "workspace_path": entry.workspace_path,
    }


def _serialize_retry_entry(entry: RetryEntry) -> dict[str, Any]:
    return {
        "issue_id": entry.issue.id,
        "identifier": entry.issue.identifier,
        "title": entry.issue.title,
        "attempt": entry.attempt,
        "due_at": entry.due_at.isoformat() if entry.due_at else None,
        "error": entry.error,
        "preferred_host": entry.preferred_host,
    }
