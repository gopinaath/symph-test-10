"""Tests for the REST API."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from symphony.api import create_app


@dataclass
class MockIssue:
    id: str = "issue-1"
    identifier: str = "TEST-1"
    title: str = "Test issue"
    state: str = "in progress"


@dataclass
class MockRunningEntry:
    issue: MockIssue = field(default_factory=MockIssue)
    session_id: str = "sess-1"
    turn_count: int = 2
    last_event: str = "turn completed"
    last_event_at: datetime = field(default_factory=lambda: datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc))
    started_at: datetime = field(default_factory=lambda: datetime(2026, 3, 28, 11, 0, 0, tzinfo=timezone.utc))
    input_tokens: int = 1000
    output_tokens: int = 500
    total_tokens: int = 1500
    worker_host: str = None
    workspace_path: str = "/tmp/test"


@dataclass
class MockRetryEntry:
    issue: MockIssue = field(default_factory=MockIssue)
    attempt: int = 1
    due_at: datetime = field(default_factory=lambda: datetime(2026, 3, 28, 12, 1, 0, tzinfo=timezone.utc))
    error: str = "timeout"
    preferred_host: str = None


@dataclass
class MockSnapshot:
    running: dict = field(default_factory=dict)
    retry_queue: dict = field(default_factory=dict)
    completed: set = field(default_factory=set)
    codex_totals: dict = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "seconds_running": 0}
    )
    codex_rate_limits: dict = None
    poll_countdown_ms: int = 25000
    poll_checking: bool = False


class MockOrchestrator:
    def __init__(self, snapshot=None):
        self._snapshot = snapshot or MockSnapshot()
        self._refresh_called = False

    def snapshot(self):
        return self._snapshot

    def request_refresh(self):
        self._refresh_called = True


class TestAPI:
    def test_health(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_state_returns_snapshot(self):
        orch = MockOrchestrator(
            MockSnapshot(
                running={"issue-1": MockRunningEntry()},
            )
        )
        app = create_app(orchestrator=orch)
        client = TestClient(app)
        resp = client.get("/api/v1/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert "issue-1" in data["running"]
        assert data["running"]["issue-1"]["identifier"] == "TEST-1"

    def test_state_unavailable_without_orchestrator(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/v1/state")
        assert resp.status_code == 503

    def test_get_issue_found(self):
        orch = MockOrchestrator(
            MockSnapshot(
                running={"issue-1": MockRunningEntry()},
            )
        )
        app = create_app(orchestrator=orch)
        client = TestClient(app)
        resp = client.get("/api/v1/TEST-1")
        assert resp.status_code == 200
        assert resp.json()["identifier"] == "TEST-1"

    def test_get_issue_not_found(self):
        orch = MockOrchestrator()
        app = create_app(orchestrator=orch)
        client = TestClient(app)
        resp = client.get("/api/v1/UNKNOWN-1")
        assert resp.status_code == 404

    def test_get_issue_from_retry_queue(self):
        orch = MockOrchestrator(
            MockSnapshot(
                retry_queue={"issue-1": MockRetryEntry()},
            )
        )
        app = create_app(orchestrator=orch)
        client = TestClient(app)
        resp = client.get("/api/v1/TEST-1")
        assert resp.status_code == 200
        assert resp.json()["attempt"] == 1

    def test_refresh_triggers_poll(self):
        orch = MockOrchestrator()
        app = create_app(orchestrator=orch)
        client = TestClient(app)
        resp = client.post("/api/v1/refresh")
        assert resp.status_code == 200
        assert orch._refresh_called

    def test_refresh_unavailable_without_orchestrator(self):
        app = create_app()
        client = TestClient(app)
        resp = client.post("/api/v1/refresh")
        assert resp.status_code == 503
