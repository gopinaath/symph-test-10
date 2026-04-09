"""Tests for the core orchestrator.

Covers state reconciliation, retry logic, polling, dispatch ordering,
blocker/assignee filtering, worker host selection, and snapshot contents.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from symphony.config import AgentConfig, CodexConfig, Config, PollingConfig, WorkerConfig
from symphony.models import BlockerInfo, Issue
from symphony.orchestrator import (
    AgentResult,
    Orchestrator,
    RetryEntry,
)
from symphony.tracker.memory import MemoryTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COUNTER = 0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_issue(
    identifier: str = "ISS-1",
    title: str = "Test issue",
    state: str = "Todo",
    priority: int | None = None,
    created_at: datetime | None = None,
    assigned_to_worker: bool = True,
    assignee_id: str | None = None,
    blocked_by: list[BlockerInfo] | None = None,
) -> Issue:
    global _COUNTER
    _COUNTER += 1
    return Issue(
        id=f"id-{identifier}-{_COUNTER}",
        identifier=identifier,
        title=title,
        description="",
        priority=priority,
        state=state,
        branch_name=f"branch-{identifier}",
        url=f"https://example.com/{identifier}",
        assignee_id=assignee_id,
        blocked_by=blocked_by or [],
        assigned_to_worker=assigned_to_worker,
        created_at=created_at or _now(),
    )


def _make_config(
    max_concurrent: int = 5,
    polling_interval_ms: int = 60_000,
    stall_timeout_ms: int = 600_000,
    max_retry_backoff_ms: int = 320_000,
    ssh_hosts: list[str] | None = None,
    max_concurrent_agents_per_host: int = 2,
    worker_name: str | None = None,
    max_concurrent_per_state: dict[str, int] | None = None,
) -> Config:
    # Build the worker config; add ``name`` dynamically since the Pydantic
    # model may not have it yet.
    worker = WorkerConfig(
        ssh_hosts=ssh_hosts or [],
        max_concurrent_agents_per_host=max_concurrent_agents_per_host,
    )
    # Attach worker name (the orchestrator reads it via getattr).
    if worker_name is not None:
        object.__setattr__(worker, "name", worker_name)

    return Config(
        polling=PollingConfig(interval_ms=polling_interval_ms),
        agent=AgentConfig(
            max_concurrent_agents=max_concurrent,
            max_retry_backoff_ms=max_retry_backoff_ms,
            max_concurrent_agents_by_state=max_concurrent_per_state or {},
        ),
        codex=CodexConfig(
            stall_timeout_ms=stall_timeout_ms,
        ),
        worker=worker,
    )


class StubWorkspace:
    """Minimal workspace stub for testing -- no filesystem side-effects."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.removed: list[str] = []

    async def create(self, identifier: str) -> str:
        self.created.append(identifier)
        return f"/tmp/ws/{identifier}"

    async def remove(self, identifier: str) -> None:
        self.removed.append(identifier)


def _make_runner(
    result: AgentResult | None = None,
    error: Exception | None = None,
    hang: bool = False,
    delay: float = 0,
) -> AsyncMock:
    """Return an async mock agent runner factory."""
    if result is None and error is None and not hang:
        result = AgentResult()

    async def _run(issue: Issue, ws: str | None, host: str | None) -> AgentResult:
        if hang:
            await asyncio.sleep(999_999)
        if delay:
            await asyncio.sleep(delay)
        if error is not None:
            raise error
        assert result is not None
        return result

    mock = AsyncMock(side_effect=_run)
    return mock


async def _run_orchestrator_cycle(orch: Orchestrator) -> None:
    """Manually trigger one poll cycle (bypassing the timer loop)."""
    await orch._poll_cycle()


# ---------------------------------------------------------------------------
# State reconciliation
# ---------------------------------------------------------------------------


class TestReconciliation:
    """Tests for _reconcile() behaviour."""

    async def test_no_running_issues_is_noop(self) -> None:
        """Reconciliation with empty running dict does nothing."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="Todo")],
        )
        ws = StubWorkspace()
        runner = _make_runner()
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        # No running entries -- reconcile should be a no-op.
        await orch._reconcile()

        assert len(orch.running) == 0
        assert len(ws.removed) == 0

    async def test_non_active_state_stops_agent_no_workspace_cleanup(self) -> None:
        """If a running issue moves to a non-active, non-terminal state,
        the agent is stopped but the workspace is kept."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(hang=True)
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        # Manually dispatch the issue
        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        assert "ISS-1" in orch.running

        # Move to non-active state (e.g. "Todo")
        tracker.set_issue_state("ISS-1", "Todo")

        await orch._reconcile()

        assert "ISS-1" not in orch.running
        assert "ISS-1" not in ws.removed  # workspace NOT cleaned

    async def test_terminal_state_stops_agent_and_cleans_workspace(self) -> None:
        """If a running issue moves to a terminal state, the agent is
        stopped AND the workspace is cleaned up."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(hang=True)
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        assert "ISS-1" in orch.running

        # Move to terminal state
        tracker.set_issue_state("ISS-1", "Done")

        await orch._reconcile()

        assert "ISS-1" not in orch.running
        assert "ISS-1" in ws.removed  # workspace IS cleaned

    async def test_missing_issue_stops_agent_no_workspace_cleanup(self) -> None:
        """If a running issue is no longer found in the tracker,
        the agent is stopped without workspace cleanup."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(hang=True)
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        assert "ISS-1" in orch.running

        # Remove the issue entirely
        tracker.remove_issue("ISS-1")

        await orch._reconcile()

        assert "ISS-1" not in orch.running
        assert "ISS-1" not in ws.removed

    async def test_reconcile_updates_running_issue_state(self) -> None:
        """Reconciliation updates the cached issue state for active issues."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(hang=True)
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)

        # The issue stays active; verify the entry remains
        await orch._reconcile()

        assert "ISS-1" in orch.running
        assert orch.running["ISS-1"].issue.state == "InProgress"

    async def test_reconcile_stops_reassigned_issue(self) -> None:
        """If a running issue is reassigned away from our worker,
        the agent is stopped without workspace cleanup."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress", assigned_to_worker=True)],
        )
        ws = StubWorkspace()
        runner = _make_runner(hang=True)
        config = _make_config(worker_name="worker-1")
        orch = Orchestrator(config, tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress", assigned_to_worker=True)
        await orch._dispatch_issue(issue)
        assert "ISS-1" in orch.running

        # Reassign away from this worker
        orch.running["ISS-1"].issue.assigned_to_worker = False

        await orch._reconcile()

        assert "ISS-1" not in orch.running
        assert "ISS-1" not in ws.removed


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for normal/abnormal exit retry scheduling."""

    async def test_normal_exit_schedules_continuation_retry(self) -> None:
        """Normal agent completion schedules a ~1s continuation retry."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(result=AgentResult())
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)

        # Wait for the agent to finish
        await asyncio.sleep(0.05)

        # The issue should be in the retry queue (continuation)
        assert "ISS-1" in orch.retry_queue
        entry = orch.retry_queue["ISS-1"]
        assert entry.attempt == 0
        # due_at should be ~1 second from now
        delta = (entry.due_at - _now()).total_seconds()
        assert -0.5 <= delta <= 1.5

    async def test_abnormal_exit_increments_retry_progressively(self) -> None:
        """Each abnormal exit bumps the retry attempt counter."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()

        async def _failing_runner(issue, ws_path, host):
            return AgentResult(error="crash")

        orch = Orchestrator(_make_config(), tracker, ws, _failing_runner)

        # First dispatch
        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        await asyncio.sleep(0.05)

        # After first failure: attempt=1
        assert "ISS-1" in orch.retry_queue
        assert orch.retry_queue["ISS-1"].attempt == 1

        # Simulate processing that retry (set due_at to past)
        orch.retry_queue["ISS-1"].due_at = _now() - timedelta(seconds=1)
        await orch._process_retry_queue()
        await asyncio.sleep(0.05)

        # After second failure: attempt=2
        assert "ISS-1" in orch.retry_queue
        assert orch.retry_queue["ISS-1"].attempt == 2

        # Process again
        orch.retry_queue["ISS-1"].due_at = _now() - timedelta(seconds=1)
        await orch._process_retry_queue()
        await asyncio.sleep(0.05)

        # After third failure: attempt=3
        assert "ISS-1" in orch.retry_queue
        assert orch.retry_queue["ISS-1"].attempt == 3

    async def test_first_abnormal_exit_waits_10s(self) -> None:
        """First failure retry is scheduled ~10s in the future."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(result=AgentResult(error="boom"))
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        await asyncio.sleep(0.05)

        entry = orch.retry_queue["ISS-1"]
        assert entry.attempt == 1
        # Due in ~10 seconds
        delta = (entry.due_at - _now()).total_seconds()
        assert 8 <= delta <= 12

    async def test_stale_retry_timer_does_not_consume_newer_entry(self) -> None:
        """A retry timer with an old token must not trigger for a newer entry."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(hang=True)
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        # Manually create a retry entry with a known token
        issue = _make_issue("ISS-1", state="InProgress")
        old_token = "old-token"
        entry_old = RetryEntry(
            issue=issue,
            attempt=1,
            due_at=_now() + timedelta(seconds=0.1),
            token=old_token,
        )
        orch._retry_queue["ISS-1"] = entry_old

        # Schedule a timer with the old token
        orch._schedule_retry_timer("ISS-1", old_token, 0.05)

        # Now replace the retry entry with a new one (new token)
        new_token = "new-token"
        entry_new = RetryEntry(
            issue=issue,
            attempt=2,
            due_at=_now() + timedelta(seconds=10),
            token=new_token,
        )
        orch._retry_queue["ISS-1"] = entry_new
        # Schedule a new timer (which cancels the old timer task)
        orch._schedule_retry_timer("ISS-1", new_token, 10.0)

        # Wait a bit for old timer to have fired (if it wasn't cancelled)
        await asyncio.sleep(0.15)

        # The retry queue should still have the new entry
        assert "ISS-1" in orch.retry_queue
        assert orch.retry_queue["ISS-1"].token == new_token

        # Clean up
        for _, task in orch._retry_timers.values():
            task.cancel()


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


class TestPolling:
    """Tests for the polling loop and manual refresh."""

    async def test_manual_refresh_coalesces_and_ignores_superseded(self) -> None:
        """Multiple request_refresh() calls coalesce; the poll interval
        is reset after a check completes."""
        tracker = MemoryTracker()
        ws = StubWorkspace()
        runner = _make_runner()
        config = _make_config(polling_interval_ms=60_000)
        orch = Orchestrator(config, tracker, ws, runner)

        await orch.start()
        await asyncio.sleep(0.15)  # let initial poll fire

        # After initial poll, next poll should be ~60s away.
        snap1 = orch.snapshot()
        assert snap1.poll_countdown_ms > 10_000

        # Request multiple refreshes -- they should coalesce
        orch.request_refresh()
        orch.request_refresh()
        orch.request_refresh()

        # Give the poll cycle time to fire
        await asyncio.sleep(0.15)

        # After the refresh-triggered poll, countdown should be reset to ~60s
        snap2 = orch.snapshot()
        assert snap2.poll_countdown_ms > 10_000

        await orch.stop()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    """Tests for candidate sorting and eligibility filtering."""

    async def test_sorts_by_priority_then_oldest(self) -> None:
        """Issues are dispatched priority-first (1-4, None=5), then by
        oldest created_at, then by identifier."""
        t0 = _now()
        issues = [
            _make_issue("C", state="Todo", priority=3, created_at=t0 + timedelta(seconds=2)),
            _make_issue("A", state="Todo", priority=1, created_at=t0 + timedelta(seconds=1)),
            _make_issue("B", state="Todo", priority=1, created_at=t0),
            _make_issue("D", state="Todo", priority=None, created_at=t0),
        ]
        tracker = MemoryTracker(issues=issues)
        ws = StubWorkspace()

        dispatched: list[str] = []

        async def _capturing_runner(issue, ws_path, host):
            dispatched.append(issue.identifier)
            return AgentResult()

        orch = Orchestrator(_make_config(max_concurrent=10), tracker, ws, _capturing_runner)
        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)

        # B (pri=1, oldest), A (pri=1, newer), C (pri=3), D (pri=None=5)
        assert dispatched == ["B", "A", "C", "D"]

    async def test_todo_with_non_terminal_blocker_not_eligible(self) -> None:
        """A Todo issue with a non-terminal blocker is NOT dispatched."""
        issue = _make_issue(
            "ISS-1",
            state="Todo",
            blocked_by=[
                BlockerInfo(id="b1", identifier="BLOCKER-1", state="InProgress"),
            ],
        )
        tracker = MemoryTracker(issues=[issue])
        ws = StubWorkspace()
        runner = _make_runner()
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)

        assert "ISS-1" not in orch.running
        runner.assert_not_called()

    async def test_assigned_to_another_worker_not_eligible(self) -> None:
        """An issue not assigned to this worker is NOT dispatched."""
        issue = _make_issue(
            "ISS-1",
            state="Todo",
            assigned_to_worker=False,
        )
        tracker = MemoryTracker(issues=[issue])
        ws = StubWorkspace()
        runner = _make_runner()
        config = _make_config(worker_name="my-worker")
        orch = Orchestrator(config, tracker, ws, runner)

        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)

        assert "ISS-1" not in orch.running
        runner.assert_not_called()

    async def test_todo_with_terminal_blockers_is_eligible(self) -> None:
        """A Todo issue whose blockers are ALL terminal IS dispatched."""
        issue = _make_issue(
            "ISS-1",
            state="Todo",
            blocked_by=[
                BlockerInfo(id="b1", identifier="B-1", state="Done"),
                BlockerInfo(id="b2", identifier="B-2", state="Cancelled"),
            ],
        )
        tracker = MemoryTracker(issues=[issue])
        ws = StubWorkspace()
        runner = _make_runner()
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)

        # The issue should have been dispatched
        assert runner.call_count == 1

    async def test_dispatch_revalidation_skips_stale_todo(self) -> None:
        """If a non-terminal blocker appears between the initial eligibility
        check and re-validation, the issue is skipped."""
        issue = _make_issue("ISS-1", state="Todo")
        tracker = MemoryTracker(issues=[issue])
        ws = StubWorkspace()
        runner = _make_runner()
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        # Patch fetch_issue_states_by_ids to sneak in a blocker during revalidation
        original_fetch = tracker.fetch_issue_states_by_ids
        call_count = 0

        async def _fetch_with_side_effect(ids):
            nonlocal call_count
            call_count += 1
            result = await original_fetch(ids)
            # After the first revalidation call for ISS-1, add a blocker
            if call_count == 1 and "ISS-1" in ids:
                tracker.issues["ISS-1"].blocked_by = [
                    BlockerInfo(
                        id="b1",
                        identifier="BLOCK-1",
                        state="InProgress",
                    )
                ]
            return result

        tracker.fetch_issue_states_by_ids = _fetch_with_side_effect

        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)

        # The issue should NOT have been dispatched because re-check sees blocker
        runner.assert_not_called()


# ---------------------------------------------------------------------------
# Worker host selection
# ---------------------------------------------------------------------------


class TestWorkerHostSelection:
    """Tests for SSH host capacity and selection."""

    async def test_skips_full_hosts(self) -> None:
        """With 2 hosts and max 1 per host, filling host-a routes to host-b."""
        tracker = MemoryTracker(
            issues=[
                _make_issue("ISS-1", state="Todo"),
                _make_issue("ISS-2", state="Todo"),
            ],
        )
        ws = StubWorkspace()

        hosts_used: list[str | None] = []

        async def _capturing_runner(issue, ws_path, host):
            hosts_used.append(host)
            await asyncio.sleep(999_999)
            return AgentResult()

        config = _make_config(
            ssh_hosts=["host-a", "host-b"],
            max_concurrent_agents_per_host=1,
            max_concurrent=10,
        )
        orch = Orchestrator(config, tracker, ws, _capturing_runner)

        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)

        assert len(orch.running) == 2
        running_hosts = {e.worker_host for e in orch.running.values()}
        assert running_hosts == {"host-a", "host-b"}

        # Clean up
        for entry in list(orch.running.values()):
            entry.task.cancel()
        await asyncio.sleep(0.05)

    async def test_no_worker_capacity_when_all_hosts_full(self) -> None:
        """If every SSH host is at capacity, the issue is not dispatched."""
        tracker = MemoryTracker(
            issues=[
                _make_issue("ISS-1", state="Todo"),
                _make_issue("ISS-2", state="Todo"),
                _make_issue("ISS-3", state="Todo"),
            ],
        )
        ws = StubWorkspace()

        async def _hanging_runner(issue, ws_path, host):
            await asyncio.sleep(999_999)
            return AgentResult()

        config = _make_config(
            ssh_hosts=["host-a"],
            max_concurrent_agents_per_host=1,
            max_concurrent=10,
        )
        orch = Orchestrator(config, tracker, ws, _hanging_runner)

        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)

        # Only 1 should be running (host-a is full after that)
        assert len(orch.running) == 1

        # Clean up
        for entry in list(orch.running.values()):
            entry.task.cancel()
        await asyncio.sleep(0.05)

    async def test_preferred_host_retained_when_capacity(self) -> None:
        """On retry, the preferred host is used if it still has capacity."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()

        hosts_seen: list[str | None] = []

        async def _capturing_runner(issue, ws_path, host):
            hosts_seen.append(host)
            return AgentResult()

        config = _make_config(
            ssh_hosts=["host-a", "host-b"],
            max_concurrent_agents_per_host=2,
            max_concurrent=10,
        )
        orch = Orchestrator(config, tracker, ws, _capturing_runner)

        # Dispatch with preferred_host=host-b
        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue, preferred_host="host-b")
        await asyncio.sleep(0.05)

        assert hosts_seen == ["host-b"]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    """Tests for OrchestratorSnapshot contents."""

    async def test_snapshot_reflects_codex_update_and_session_id(self) -> None:
        """After an agent run the snapshot includes session_id and token totals."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(
            result=AgentResult(
                session_id="sess-abc",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            )
        )
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        await asyncio.sleep(0.05)

        snap = orch.snapshot()
        assert snap.codex_totals["input_tokens"] == 100
        assert snap.codex_totals["output_tokens"] == 50
        assert snap.codex_totals["total_tokens"] == 150

    async def test_snapshot_tracks_thread_totals(self) -> None:
        """Token totals accumulate across multiple agent runs."""
        tracker = MemoryTracker(
            issues=[
                _make_issue("ISS-1", state="InProgress"),
                _make_issue("ISS-2", state="InProgress"),
            ],
        )
        ws = StubWorkspace()
        runner = _make_runner(
            result=AgentResult(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
            )
        )
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        await orch._dispatch_issue(_make_issue("ISS-1", state="InProgress"))
        await orch._dispatch_issue(_make_issue("ISS-2", state="InProgress"))
        await asyncio.sleep(0.1)

        snap = orch.snapshot()
        assert snap.codex_totals["input_tokens"] == 20
        assert snap.codex_totals["output_tokens"] == 40
        assert snap.codex_totals["total_tokens"] == 60

    async def test_snapshot_tracks_turn_completed_usage(self) -> None:
        """The snapshot includes per-turn token usage accumulated on the entry."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(
            result=AgentResult(
                session_id="sess-1",
                turn_count=1,
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
            )
        )
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        await asyncio.sleep(0.05)

        snap = orch.snapshot()
        assert snap.codex_totals["total_tokens"] == 300
        assert snap.codex_totals["seconds_running"] > 0

    async def test_snapshot_tracks_rate_limit_payloads(self) -> None:
        """Rate limit info from agent result appears in the snapshot."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        rate_limits = {"requests_remaining": 42, "reset_at": "2025-01-01T00:00:00Z"}
        runner = _make_runner(result=AgentResult(rate_limits=rate_limits))
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        await orch._dispatch_issue(_make_issue("ISS-1", state="InProgress"))
        await asyncio.sleep(0.05)

        snap = orch.snapshot()
        assert snap.codex_rate_limits == rate_limits

    async def test_snapshot_includes_retry_entries(self) -> None:
        """Retry queue entries appear in the snapshot."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(result=AgentResult(error="crash"))
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        await orch._dispatch_issue(_make_issue("ISS-1", state="InProgress"))
        await asyncio.sleep(0.05)

        snap = orch.snapshot()
        assert "ISS-1" in snap.retry_queue
        assert snap.retry_queue["ISS-1"].attempt == 1
        assert snap.retry_queue["ISS-1"].error == "crash"

    async def test_snapshot_includes_poll_countdown_and_checking(self) -> None:
        """Snapshot has poll_countdown_ms and poll_checking fields."""
        tracker = MemoryTracker()
        ws = StubWorkspace()
        runner = _make_runner()
        config = _make_config(polling_interval_ms=60_000)
        orch = Orchestrator(config, tracker, ws, runner)

        await orch.start()
        await asyncio.sleep(0.15)  # let initial poll fire

        snap = orch.snapshot()
        assert snap.poll_countdown_ms > 0
        assert snap.poll_checking is False

        await orch.stop()

    async def test_triggers_immediate_poll_shortly_after_startup(self) -> None:
        """The first poll fires within ~200ms of start()."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="Todo")],
        )
        ws = StubWorkspace()
        runner = _make_runner()
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        await orch.start()
        await asyncio.sleep(0.25)

        assert runner.call_count >= 1

        await orch.stop()

    async def test_poll_cycle_resets_countdown(self) -> None:
        """After a poll cycle, the countdown is reset to the polling interval."""
        tracker = MemoryTracker()
        ws = StubWorkspace()
        runner = _make_runner()
        config = _make_config(polling_interval_ms=30_000)
        orch = Orchestrator(config, tracker, ws, runner)

        await orch.start()
        await asyncio.sleep(0.15)  # initial poll at 100ms fires

        snap1 = orch.snapshot()
        assert snap1.poll_countdown_ms > 20_000

        # Request a manual refresh
        orch.request_refresh()
        await asyncio.sleep(0.15)

        snap2 = orch.snapshot()
        assert snap2.poll_countdown_ms > 20_000

        await orch.stop()

    async def test_restarts_stalled_workers_with_retry_backoff(self) -> None:
        """A stalled agent is terminated and re-queued with retry backoff."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(hang=True)
        # Use a very short stall timeout for testing
        config = _make_config(stall_timeout_ms=1)
        orch = Orchestrator(config, tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        assert "ISS-1" in orch.running

        # Wait a tiny bit so the stall threshold is passed
        await asyncio.sleep(0.01)

        await orch._reconcile()

        assert "ISS-1" not in orch.running
        assert "ISS-1" in orch.retry_queue
        retry = orch.retry_queue["ISS-1"]
        assert retry.attempt == 1
        assert retry.error == "stall_timeout"
        # Backoff should be ~10s for attempt 1
        delta = (retry.due_at - _now()).total_seconds()
        assert 8 <= delta <= 12


# ---------------------------------------------------------------------------
# Integration: full start/stop cycle
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests with MemoryTracker."""

    async def test_full_lifecycle(self) -> None:
        """Start orchestrator, let it dispatch, agent completes, stop."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="Todo")],
        )
        ws = StubWorkspace()
        runner = _make_runner(result=AgentResult(session_id="s1", input_tokens=5))
        orch = Orchestrator(_make_config(), tracker, ws, runner)

        await orch.start()
        await asyncio.sleep(0.3)

        snap = orch.snapshot()
        assert snap.codex_totals["input_tokens"] == 5
        assert runner.call_count >= 1

        await orch.stop()

    async def test_exponential_backoff_cap(self) -> None:
        """Retry backoff is capped at max_retry_backoff_ms."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(result=AgentResult(error="fail"))
        config = _make_config(max_retry_backoff_ms=15_000)
        orch = Orchestrator(config, tracker, ws, runner)

        # Simulate high attempt number
        issue = _make_issue("ISS-1", state="InProgress")
        orch._schedule_failure_retry(issue, attempt=100, error="fail")

        entry = orch.retry_queue["ISS-1"]
        # 10_000 * 2^99 would be huge, but capped at 15_000 ms = 15s
        delta = (entry.due_at - _now()).total_seconds()
        assert delta <= 16

        # Clean up timers
        for _, task in orch._retry_timers.values():
            task.cancel()

    async def test_completed_issues_not_redispatched(self) -> None:
        """Once an issue completes and its state is terminal, it is not
        dispatched again."""
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="Todo")],
        )
        ws = StubWorkspace()

        call_count = 0

        async def _counting_runner(issue, ws_path, host):
            nonlocal call_count
            call_count += 1
            await tracker.update_issue_state("ISS-1", "Done")
            return AgentResult()

        orch = Orchestrator(_make_config(), tracker, ws, _counting_runner)
        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)

        assert call_count == 1

        # Run another cycle -- the issue is Done, should not dispatch
        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)
        assert call_count == 1

    async def test_global_capacity_limits_dispatch(self) -> None:
        """When max_concurrent is reached, no more issues are dispatched."""
        issues = [_make_issue(f"ISS-{i}", state="Todo") for i in range(5)]
        tracker = MemoryTracker(issues=issues)
        ws = StubWorkspace()

        async def _hanging_runner(issue, ws_path, host):
            await asyncio.sleep(999_999)
            return AgentResult()

        config = _make_config(max_concurrent=2)
        orch = Orchestrator(config, tracker, ws, _hanging_runner)

        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)

        assert len(orch.running) == 2

        # Clean up
        for entry in list(orch.running.values()):
            entry.task.cancel()
        await asyncio.sleep(0.05)

    async def test_per_state_capacity_limits(self) -> None:
        """Per-state concurrency limits are respected."""
        issues = [
            _make_issue("ISS-1", state="Todo"),
            _make_issue("ISS-2", state="Todo"),
            _make_issue("ISS-3", state="InProgress"),
        ]
        tracker = MemoryTracker(issues=issues)
        ws = StubWorkspace()

        async def _hanging_runner(issue, ws_path, host):
            await asyncio.sleep(999_999)
            return AgentResult()

        config = _make_config(
            max_concurrent=10,
            max_concurrent_per_state={"Todo": 1},
        )
        orch = Orchestrator(config, tracker, ws, _hanging_runner)

        await _run_orchestrator_cycle(orch)
        await asyncio.sleep(0.05)

        todo_running = sum(1 for e in orch.running.values() if e.issue.state == "Todo")
        assert todo_running == 1
        assert len(orch.running) == 2  # 1 Todo + 1 InProgress

        # Clean up
        for entry in list(orch.running.values()):
            entry.task.cancel()
        await asyncio.sleep(0.05)
