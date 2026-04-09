"""Integration tests ported from Elixir extensions_test.exs.

Covers:
- WorkflowStore lifecycle (hot-reload, broken YAML, stopped store, path change)
- MemoryTracker delegation (add/fetch/update, comment, state)
- REST API responses (GET /state, GET /:id, POST /refresh, 404, 405, 503, timeout)
- Dashboard payload field verification
"""

import asyncio
import os
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from symphony.api import create_app
from symphony.models import Issue
from symphony.tracker.memory import MemoryTracker
from symphony.workflow import Workflow, WorkflowStore


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VALID_WORKFLOW = textwrap.dedent("""\
    ---
    tracker:
      kind: memory
    polling:
      interval_ms: 5000
    ---
    You are an agent working on {{ issue.identifier }}.
""")

UPDATED_WORKFLOW = textwrap.dedent("""\
    ---
    tracker:
      kind: memory
    polling:
      interval_ms: 9999
    ---
    Updated prompt for {{ issue.identifier }}.
""")

BAD_YAML_WORKFLOW = textwrap.dedent("""\
    ---
    - this is a list not a mapping
    ---
    Bad prompt.
""")


def _write_workflow(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    # Bump mtime to ensure the store sees the change even on fast filesystems.
    future = time.time() + 2
    os.utime(path, (future, future))


def _make_issue(
    identifier: str = "TEST-1",
    title: str = "Test issue",
    state: str = "Todo",
) -> Issue:
    return Issue(
        id=f"id-{identifier}",
        identifier=identifier,
        title=title,
        description="A test issue",
        priority=2,
        state=state,
        branch_name=f"symphony/{identifier}",
        url=f"https://example.com/{identifier}",
        assignee_id=None,
    )


# --- Mock orchestrator for API tests ---


@dataclass
class _MockIssue:
    id: str = "issue-1"
    identifier: str = "EXT-1"
    title: str = "Extension test"
    state: str = "in progress"


@dataclass
class _MockRunningEntry:
    issue: _MockIssue = field(default_factory=_MockIssue)
    session_id: str = "sess-ext-1"
    turn_count: int = 3
    last_event: str = "tool_call"
    last_event_at: datetime = field(
        default_factory=lambda: datetime(2026, 4, 5, 10, 0, 0, tzinfo=timezone.utc)
    )
    started_at: datetime = field(
        default_factory=lambda: datetime(2026, 4, 5, 9, 30, 0, tzinfo=timezone.utc)
    )
    input_tokens: int = 2000
    output_tokens: int = 800
    total_tokens: int = 2800
    worker_host: Optional[str] = None
    workspace_path: str = "/tmp/ext-test"


@dataclass
class _MockRetryEntry:
    issue: _MockIssue = field(default_factory=_MockIssue)
    attempt: int = 2
    due_at: datetime = field(
        default_factory=lambda: datetime(2026, 4, 5, 10, 5, 0, tzinfo=timezone.utc)
    )
    error: str = "rate_limited"
    preferred_host: Optional[str] = None


@dataclass
class _MockSnapshot:
    running: dict = field(default_factory=dict)
    retry_queue: dict = field(default_factory=dict)
    completed: set = field(default_factory=set)
    codex_totals: dict = field(
        default_factory=lambda: {
            "input_tokens": 5000,
            "output_tokens": 2000,
            "total_tokens": 7000,
            "seconds_running": 120,
        }
    )
    codex_rate_limits: Optional[dict] = None
    poll_countdown_ms: int = 20000
    poll_checking: bool = False


class _MockOrchestrator:
    def __init__(self, snapshot=None):
        self._snapshot = snapshot or _MockSnapshot()
        self._refresh_called = False

    def snapshot(self):
        return self._snapshot

    def request_refresh(self):
        self._refresh_called = True


class _SlowOrchestrator:
    """Simulates a slow snapshot that times out."""

    def snapshot(self):
        raise Exception("snapshot timed out")

    def request_refresh(self):
        pass


# ===================================================================
# 1. WorkflowStore lifecycle
# ===================================================================


class TestWorkflowStoreLifecycle:
    """Port of extensions_test.exs workflow-store tests."""

    def test_reloads_changes_on_file_modification(self):
        """WorkflowStore picks up changes when the file is modified."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp:
            tmp.write(VALID_WORKFLOW)
            tmp.flush()
            path = tmp.name

        try:
            store = WorkflowStore(path=path, poll_interval=0.1)
            wf = store.init()
            assert wf.config.polling.interval_ms == 5000

            store.start_polling()
            try:
                _write_workflow(path, UPDATED_WORKFLOW)
                # Wait for the poller to pick up the change.
                deadline = time.time() + 5
                while time.time() < deadline:
                    current = store.workflow
                    if current and current.config.polling.interval_ms == 9999:
                        break
                    time.sleep(0.05)

                assert store.workflow is not None
                assert store.workflow.config.polling.interval_ms == 9999
                assert "Updated prompt" in store.workflow.prompt_template
            finally:
                store.stop_polling()
        finally:
            os.unlink(path)

    def test_keeps_last_good_workflow_on_bad_yaml(self):
        """Broken YAML keeps the last successfully loaded workflow."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp:
            tmp.write(VALID_WORKFLOW)
            tmp.flush()
            path = tmp.name

        try:
            store = WorkflowStore(path=path, poll_interval=0.1)
            wf = store.init()
            assert wf.config.polling.interval_ms == 5000

            store.start_polling()
            try:
                _write_workflow(path, BAD_YAML_WORKFLOW)
                # Wait for the poller to attempt the bad reload.
                deadline = time.time() + 5
                while time.time() < deadline:
                    if store.last_error is not None:
                        break
                    time.sleep(0.05)

                # last_error should be set, but the good workflow persists.
                assert store.last_error is not None
                assert store.workflow is not None
                assert store.workflow.config.polling.interval_ms == 5000
            finally:
                store.stop_polling()
        finally:
            os.unlink(path)

    def test_falls_back_to_cached_when_store_is_stopped(self):
        """After stop_polling the cached workflow is still accessible."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp:
            tmp.write(VALID_WORKFLOW)
            tmp.flush()
            path = tmp.name

        try:
            store = WorkflowStore(path=path, poll_interval=0.1)
            store.init()
            store.start_polling()
            store.stop_polling()

            # Workflow should still be returned from cache.
            assert store.workflow is not None
            assert store.workflow.config.polling.interval_ms == 5000
            assert "agent working on" in store.workflow.prompt_template
        finally:
            os.unlink(path)

    def test_path_change_triggers_reload(self):
        """Swapping to a new path and re-init loads the new file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp1:
            tmp1.write(VALID_WORKFLOW)
            tmp1.flush()
            path1 = tmp1.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp2:
            tmp2.write(UPDATED_WORKFLOW)
            tmp2.flush()
            path2 = tmp2.name

        try:
            store = WorkflowStore(path=path1)
            wf1 = store.init()
            assert wf1.config.polling.interval_ms == 5000

            # Simulate path change by creating a new store with the second path.
            store2 = WorkflowStore(path=path2)
            wf2 = store2.init()
            assert wf2.config.polling.interval_ms == 9999
            assert "Updated prompt" in wf2.prompt_template
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ===================================================================
# 2. Tracker delegation
# ===================================================================


class TestMemoryTrackerDelegation:
    """Port of extensions_test.exs tracker delegation tests."""

    @pytest.fixture()
    def tracker(self):
        return MemoryTracker(
            candidate_states={"Todo", "InProgress"},
            active_states={"InProgress"},
            terminal_states={"Done", "Cancelled"},
        )

    async def test_add_and_fetch_candidates(self, tracker):
        """add_issue + fetch_candidate_issues round-trips correctly."""
        issue = _make_issue("EXT-10", state="Todo")
        tracker.add_issue(issue)

        candidates = await tracker.fetch_candidate_issues()
        assert len(candidates) == 1
        assert candidates[0].identifier == "EXT-10"
        assert candidates[0].state == "Todo"

    async def test_fetch_returns_empty_when_no_match(self, tracker):
        """fetch_candidate_issues returns [] when no issues match."""
        issue = _make_issue("EXT-11", state="Done")
        tracker.add_issue(issue)

        candidates = await tracker.fetch_candidate_issues()
        assert candidates == []

    async def test_update_state_delegates(self, tracker):
        """update_issue_state changes the stored state."""
        tracker.add_issue(_make_issue("EXT-20", state="Todo"))

        await tracker.update_issue_state("EXT-20", "InProgress")
        states = await tracker.fetch_issue_states_by_ids(["EXT-20"])
        assert states["EXT-20"] == "InProgress"

    async def test_create_comment_delegates(self, tracker):
        """create_comment stores the comment text."""
        tracker.add_issue(_make_issue("EXT-30"))

        await tracker.create_comment("EXT-30", "Hello from test")
        assert "Hello from test" in tracker.comments["EXT-30"]

    async def test_fetch_issues_by_states(self, tracker):
        """fetch_issues_by_states filters correctly."""
        tracker.add_issue(_make_issue("EXT-40", state="Todo"))
        tracker.add_issue(_make_issue("EXT-41", state="InProgress"))
        tracker.add_issue(_make_issue("EXT-42", state="Done"))

        result = await tracker.fetch_issues_by_states(["InProgress"])
        assert len(result) == 1
        assert result[0].identifier == "EXT-41"

    async def test_fetch_issue_states_by_ids_missing(self, tracker):
        """Missing identifiers return None."""
        states = await tracker.fetch_issue_states_by_ids(["NOPE-1"])
        assert states["NOPE-1"] is None

    async def test_state_change_callback(self):
        """on_state_change callback fires on state transitions."""
        calls = []

        async def cb(ident, old, new):
            calls.append((ident, old, new))

        tracker = MemoryTracker(on_state_change=cb)
        tracker.add_issue(_make_issue("EXT-50", state="Todo"))

        await tracker.update_issue_state("EXT-50", "InProgress")
        assert len(calls) == 1
        assert calls[0] == ("EXT-50", "Todo", "InProgress")

    async def test_is_active_and_terminal(self, tracker):
        """Helper methods report correct boolean for state strings."""
        assert tracker.is_active_state("InProgress") is True
        assert tracker.is_active_state("Todo") is False
        assert tracker.is_terminal_state("Done") is True
        assert tracker.is_terminal_state("InProgress") is False


# ===================================================================
# 3. REST API responses
# ===================================================================


class TestRESTAPIResponses:
    """Port of extensions_test.exs Phoenix API response tests."""

    def test_get_state_returns_full_snapshot(self):
        """GET /api/v1/state returns all expected top-level keys."""
        orch = _MockOrchestrator(
            _MockSnapshot(running={"issue-1": _MockRunningEntry()})
        )
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.get("/api/v1/state")
        assert resp.status_code == 200
        data = resp.json()

        expected_keys = {
            "running",
            "retry_queue",
            "completed_count",
            "codex_totals",
            "codex_rate_limits",
            "poll_countdown_ms",
            "poll_checking",
        }
        assert expected_keys == set(data.keys())
        assert "issue-1" in data["running"]

    def test_get_specific_issue_returns_entry(self):
        """GET /api/v1/<identifier> returns the running entry."""
        orch = _MockOrchestrator(
            _MockSnapshot(running={"issue-1": _MockRunningEntry()})
        )
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.get("/api/v1/EXT-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["identifier"] == "EXT-1"
        assert data["session_id"] == "sess-ext-1"

    def test_post_refresh_triggers_refresh(self):
        """POST /api/v1/refresh calls request_refresh on the orchestrator."""
        orch = _MockOrchestrator()
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.post("/api/v1/refresh")
        assert resp.status_code == 200
        assert resp.json()["status"] == "refresh_requested"
        assert orch._refresh_called is True

    def test_404_for_unknown_issue(self):
        """GET /api/v1/<unknown> returns 404."""
        orch = _MockOrchestrator()
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.get("/api/v1/UNKNOWN-999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_405_for_unsupported_method(self):
        """PUT /api/v1/state should be rejected (405)."""
        orch = _MockOrchestrator()
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.put("/api/v1/state")
        assert resp.status_code == 405

    def test_503_when_orchestrator_unavailable(self):
        """GET /api/v1/state without orchestrator returns 503."""
        app = create_app(orchestrator=None)
        client = TestClient(app)

        resp = client.get("/api/v1/state")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    def test_503_on_snapshot_timeout(self):
        """Slow/erroring orchestrator returns 503."""
        app = create_app(orchestrator=_SlowOrchestrator())
        client = TestClient(app)

        resp = client.get("/api/v1/state")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    def test_refresh_503_when_no_orchestrator(self):
        """POST /api/v1/refresh without orchestrator returns 503."""
        app = create_app(orchestrator=None)
        client = TestClient(app)

        resp = client.post("/api/v1/refresh")
        assert resp.status_code == 503

    def test_get_issue_from_retry_queue(self):
        """GET /api/v1/<identifier> finds issue in the retry queue."""
        retry_issue = _MockIssue(
            id="issue-r1", identifier="RETRY-1", title="Retry me", state="backoff"
        )
        orch = _MockOrchestrator(
            _MockSnapshot(
                retry_queue={
                    "issue-r1": _MockRetryEntry(issue=retry_issue)
                }
            )
        )
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.get("/api/v1/RETRY-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["identifier"] == "RETRY-1"
        assert data["attempt"] == 2


# ===================================================================
# 4. Dashboard payload verification
# ===================================================================


class TestDashboardPayload:
    """Verify the API payloads contain expected fields for dashboard consumption.

    Replaces the Elixir LiveView rendering tests with API-side field checks.
    """

    def test_running_entry_has_dashboard_fields(self):
        """Running entry payload has all fields the dashboard needs."""
        orch = _MockOrchestrator(
            _MockSnapshot(running={"issue-1": _MockRunningEntry()})
        )
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.get("/api/v1/state")
        entry = resp.json()["running"]["issue-1"]

        required_fields = {
            "issue_id",
            "identifier",
            "title",
            "state",
            "session_id",
            "turn_count",
            "last_event",
            "last_event_at",
            "started_at",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "worker_host",
            "workspace_path",
        }
        assert required_fields.issubset(set(entry.keys()))

    def test_retry_entry_has_dashboard_fields(self):
        """Retry entry payload has all fields the dashboard needs."""
        orch = _MockOrchestrator(
            _MockSnapshot(retry_queue={"issue-1": _MockRetryEntry()})
        )
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.get("/api/v1/state")
        entry = resp.json()["retry_queue"]["issue-1"]

        required_fields = {
            "issue_id",
            "identifier",
            "title",
            "attempt",
            "due_at",
            "error",
            "preferred_host",
        }
        assert required_fields.issubset(set(entry.keys()))

    def test_snapshot_codex_totals_present(self):
        """Snapshot includes codex_totals with token accounting fields."""
        orch = _MockOrchestrator()
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.get("/api/v1/state")
        data = resp.json()

        totals = data["codex_totals"]
        assert "input_tokens" in totals
        assert "output_tokens" in totals
        assert "total_tokens" in totals
        assert "seconds_running" in totals

    def test_snapshot_poll_fields_present(self):
        """Snapshot includes polling status fields for dashboard countdown."""
        orch = _MockOrchestrator()
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.get("/api/v1/state")
        data = resp.json()

        assert "poll_countdown_ms" in data
        assert isinstance(data["poll_countdown_ms"], int)
        assert "poll_checking" in data
        assert isinstance(data["poll_checking"], bool)

    def test_datetime_fields_are_iso_strings(self):
        """Datetime fields are serialized as ISO 8601 strings."""
        orch = _MockOrchestrator(
            _MockSnapshot(running={"issue-1": _MockRunningEntry()})
        )
        app = create_app(orchestrator=orch)
        client = TestClient(app)

        resp = client.get("/api/v1/state")
        entry = resp.json()["running"]["issue-1"]

        # Both datetime fields should be ISO strings (contain 'T').
        assert "T" in entry["last_event_at"]
        assert "T" in entry["started_at"]
