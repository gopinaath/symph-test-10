"""Core orchestrator — asyncio-based polling loop for Symphony."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from symphony.config import Config
from symphony.models import Issue
from symphony.tracker.base import Tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunningEntry:
    """Tracks a currently-executing agent task."""

    issue: Issue
    task: asyncio.Task[None]
    started_at: datetime
    session_id: str | None = None
    turn_count: int = 0
    last_event: str | None = None
    last_event_at: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    worker_host: str | None = None
    workspace_path: str | None = None
    attempt: int = 0  # which retry attempt this dispatch originated from


@dataclass
class RetryEntry:
    """An issue queued for a retry dispatch."""

    issue: Issue
    attempt: int
    due_at: datetime
    error: str | None = None
    preferred_host: str | None = None
    token: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class OrchestratorSnapshot:
    """A point-in-time snapshot of orchestrator state."""

    running: dict[str, RunningEntry]
    retry_queue: dict[str, RetryEntry]
    completed: set[str]
    codex_totals: dict[str, Any]  # {input_tokens, output_tokens, total_tokens, seconds_running}
    codex_rate_limits: dict[str, Any] | None
    poll_countdown_ms: int
    poll_checking: bool


@dataclass
class AgentResult:
    """Result returned by an agent runner."""

    session_id: str | None = None
    turn_count: int = 0
    last_event: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None
    rate_limits: dict[str, Any] | None = None
    validation_passed: bool = False
    validation_output: str = ""


# The runner receives (issue, workspace_path, worker_host) and returns an
# ``AgentResult``.
AgentRunnerFactory = Callable[
    [Issue, str | None, str | None],
    Coroutine[Any, Any, AgentResult],
]


# ---------------------------------------------------------------------------
# Config access helpers
# ---------------------------------------------------------------------------
# The Config model was built by another agent using Pydantic and the field
# names differ from the original spec.  These helpers abstract the
# differences so the orchestrator logic reads cleanly.


def _get_max_concurrent(config: Config) -> int:
    """Return the global agent concurrency limit."""
    return config.agent.max_concurrent_agents


def _get_max_concurrent_by_state(config: Config) -> dict[str, int]:
    agent = config.agent
    result: dict[str, int] = (
        getattr(agent, "max_concurrent_agents_by_state", None) or getattr(agent, "max_concurrent_per_state", {}) or {}
    )
    return result


def _get_stall_timeout_ms(config: Config) -> int:
    """Return stall timeout in ms (may live on codex or agent section)."""
    codex = config.codex
    stall: int = getattr(codex, "stall_timeout_ms", 0)
    if stall:
        return stall
    agent = config.agent
    result: int = getattr(agent, "stall_timeout_ms", 600_000)
    return result


def _get_max_retry_backoff_ms(config: Config) -> int:
    result: int = getattr(config.agent, "max_retry_backoff_ms", 320_000)
    return result


def _get_polling_interval_ms(config: Config) -> int:
    return config.polling.interval_ms


def _get_ssh_hosts(config: Config) -> list[str]:
    result: list[str] = getattr(config.worker, "ssh_hosts", []) or []
    return result


def _get_max_per_host(config: Config) -> int:
    result: int = getattr(config.worker, "max_concurrent_agents_per_host", 2)
    return result


def _get_worker_name(config: Config) -> str | None:
    return getattr(config.worker, "name", None)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Core polling-loop orchestrator for Symphony."""

    # Grace period for agent tasks to finish before force-cancellation.
    AGENT_GRACE_PERIOD: float = 5.0  # seconds

    def __init__(
        self,
        config: Config,
        tracker: Tracker,
        workspace: Any,  # Workspace or compatible duck-type
        agent_runner_factory: AgentRunnerFactory,
        status_dashboard: Any | None = None,
    ) -> None:
        self._config = config
        self._tracker = tracker
        self._workspace = workspace
        self._agent_runner_factory = agent_runner_factory
        self._status_dashboard = status_dashboard

        # --- internal state ---
        self._running: dict[str, RunningEntry] = {}
        self._retry_queue: dict[str, RetryEntry] = {}
        self._completed: set[str] = set()

        # Codex totals across all tasks
        self._codex_totals: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "seconds_running": 0.0,
        }
        self._codex_rate_limits: dict[str, Any] | None = None

        # Polling bookkeeping
        self._poll_task: asyncio.Task[None] | None = None
        self._refresh_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._poll_checking = False
        self._next_poll_at: float | None = None  # monotonic clock
        self._started_at: float | None = None  # monotonic clock

        # Retry timer tasks keyed by issue identifier -> (token, Task)
        self._retry_timers: dict[str, tuple[str, asyncio.Task[None]]] = {}

        # Whether stop() has been called (for idempotency).
        self._is_stopped = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin the polling loop."""
        self._started_at = time.monotonic()
        # First poll fires quickly (100 ms after startup).
        self._next_poll_at = self._started_at + 0.1
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Gracefully shut down the orchestrator.

        1. Cancels the polling loop task.
        2. Cancels all retry timer tasks.
        3. Cancels all running agent tasks with a grace period
           (``AGENT_GRACE_PERIOD`` seconds) — waits for agents to finish
           or forces cancellation on timeout.
        4. Marks all agents as stopped and cleans up internal state.
        5. Renders offline status on the ``status_dashboard`` if available.

        This method is idempotent: calling it multiple times is safe.
        """
        if self._is_stopped:
            return
        self._is_stopped = True

        self._stop_event.set()
        self._refresh_event.set()  # wake any sleeping poll

        # 1. Cancel the polling loop task.
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
        self._poll_task = None

        # 2. Cancel all retry timers.
        for _token, timer_task in self._retry_timers.values():
            timer_task.cancel()
        self._retry_timers.clear()
        self._retry_queue.clear()

        # 3. Cancel running agent tasks with a grace period.
        agent_tasks = [entry.task for entry in self._running.values()]
        if agent_tasks:
            # Request cancellation for every agent.
            for entry in list(self._running.values()):
                entry.task.cancel()

            # Wait up to AGENT_GRACE_PERIOD for agents to finish gracefully.
            _done, pending = await asyncio.wait(
                agent_tasks, timeout=self.AGENT_GRACE_PERIOD
            )

            # Force-cancel any that did not finish within the grace period.
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        # 4. Mark agents as stopped / clean up state.
        self._running.clear()

        # 5. Render offline status if a dashboard is available.
        if self._status_dashboard is not None:
            try:
                render = getattr(self._status_dashboard, "render_offline", None)
                if render is not None:
                    if asyncio.iscoroutinefunction(render):
                        await render()
                    else:
                        render()
            except Exception:
                logger.exception("Failed to render offline status on dashboard")

    def snapshot(self) -> OrchestratorSnapshot:
        """Return a point-in-time snapshot of the orchestrator state."""
        now_mono = time.monotonic()
        countdown_ms = max(0, int((self._next_poll_at - now_mono) * 1000)) if self._next_poll_at is not None else 0

        return OrchestratorSnapshot(
            running=dict(self._running),
            retry_queue=dict(self._retry_queue),
            completed=set(self._completed),
            codex_totals=dict(self._codex_totals),
            codex_rate_limits=self._codex_rate_limits,
            poll_countdown_ms=countdown_ms,
            poll_checking=self._poll_checking,
        )

    def request_refresh(self) -> None:
        """Request an immediate poll cycle (coalesces repeated calls)."""
        self._next_poll_at = time.monotonic()
        self._refresh_event.set()

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Main polling loop: reconcile, retry, dispatch."""
        interval_s = _get_polling_interval_ms(self._config) / 1000.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            delay = max(0.0, self._next_poll_at - now) if self._next_poll_at is not None else interval_s

            if delay > 0:
                self._refresh_event.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._refresh_event.wait(), timeout=delay)

            if self._stop_event.is_set():
                break

            try:
                self._poll_checking = True
                await self._poll_cycle()
            except Exception:
                logger.exception("Error in poll cycle")
            finally:
                self._poll_checking = False

            # Schedule next regular poll
            self._next_poll_at = time.monotonic() + interval_s
            self._refresh_event.clear()

    async def _poll_cycle(self) -> None:
        """Execute one complete poll cycle."""
        await self._reconcile()
        await self._process_retry_queue()
        await self._dispatch_candidates()

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def _reconcile(self) -> None:
        """Re-check states of running issues; stop agents as needed."""
        if not self._running:
            return

        identifiers = list(self._running.keys())
        current_states = await self._tracker.fetch_issue_states_by_ids(identifiers)

        for ident in identifiers:
            if ident not in self._running:
                continue  # removed by an earlier iteration
            entry = self._running[ident]
            new_state = current_states.get(ident)

            if new_state is None:
                await self._stop_agent(ident, clean_workspace=False)
            elif self._tracker.is_terminal_state(new_state):
                await self._stop_agent(ident, clean_workspace=True)
            elif not self._tracker.is_active_state(new_state) or self._is_reassigned(entry):
                await self._stop_agent(ident, clean_workspace=False)
            else:
                # Still active — update cached state
                entry.issue.state = new_state
                if self._is_stalled(entry):
                    worker_host = entry.worker_host
                    issue = entry.issue
                    attempt = entry.attempt + 1
                    await self._stop_agent(ident, clean_workspace=False)
                    self._schedule_failure_retry(
                        issue,
                        attempt=attempt,
                        error="stall_timeout",
                        preferred_host=worker_host,
                    )

    def _is_reassigned(self, entry: RunningEntry) -> bool:
        """True if the issue is assigned to a different worker."""
        worker_name = _get_worker_name(self._config)
        if not worker_name:
            return False

        issue = entry.issue
        # New-style model: ``assigned_to_worker`` bool
        if hasattr(issue, "assigned_to_worker"):
            return not issue.assigned_to_worker

        # Legacy model: ``assignee`` string compared against worker name
        assignee: str | None = getattr(issue, "assignee", None) or getattr(issue, "assignee_id", None)
        if assignee is None:
            return False
        return assignee != worker_name

    def _is_stalled(self, entry: RunningEntry) -> bool:
        """True if the agent has exceeded the stall timeout."""
        timeout_ms = _get_stall_timeout_ms(self._config)
        if timeout_ms <= 0:
            return False
        now = datetime.now(timezone.utc)
        check_time = entry.last_event_at or entry.started_at
        elapsed_ms = (now - check_time).total_seconds() * 1000
        return elapsed_ms >= timeout_ms

    async def _stop_agent(self, identifier: str, *, clean_workspace: bool) -> None:
        """Cancel an agent task and optionally remove its workspace."""
        entry = self._running.pop(identifier, None)
        if entry is None:
            return

        entry.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await entry.task

        if clean_workspace:
            try:
                await self._workspace.remove(identifier)
            except Exception:
                logger.exception("Failed to clean workspace for %s", identifier)

    # ------------------------------------------------------------------
    # Retry queue
    # ------------------------------------------------------------------

    async def _process_retry_queue(self) -> None:
        """Dispatch retries whose due_at has passed."""
        now = datetime.now(timezone.utc)
        due = [
            (ident, entry)
            for ident, entry in list(self._retry_queue.items())
            if entry.due_at <= now and ident not in self._running
        ]
        for ident, entry in due:
            states = await self._tracker.fetch_issue_states_by_ids([ident])
            current_state = states.get(ident)
            if current_state is None or self._tracker.is_terminal_state(current_state):
                self._retry_queue.pop(ident, None)
                continue
            entry.issue.state = current_state
            attempt = entry.attempt
            self._retry_queue.pop(ident, None)
            await self._dispatch_issue(
                entry.issue,
                preferred_host=entry.preferred_host,
                attempt=attempt,
            )

    def _schedule_continuation_retry(self, issue: Issue, *, preferred_host: str | None = None) -> None:
        """Schedule a ~1 s retry for an active issue (normal completion)."""
        due_at = datetime.now(timezone.utc) + timedelta(seconds=1)
        entry = RetryEntry(
            issue=issue,
            attempt=0,
            due_at=due_at,
            preferred_host=preferred_host,
        )
        self._retry_queue[issue.identifier] = entry
        self._schedule_retry_timer(issue.identifier, entry.token, 1.0)

    def _schedule_failure_retry(
        self,
        issue: Issue,
        attempt: int,
        error: str | None = None,
        preferred_host: str | None = None,
    ) -> None:
        """Schedule a retry with exponential backoff on failure."""
        max_backoff_ms = _get_max_retry_backoff_ms(self._config)
        backoff_ms = min(10_000 * (2 ** max(attempt - 1, 0)), max_backoff_ms)
        due_at = datetime.now(timezone.utc) + timedelta(milliseconds=backoff_ms)
        entry = RetryEntry(
            issue=issue,
            attempt=attempt,
            due_at=due_at,
            error=error,
            preferred_host=preferred_host,
        )
        self._retry_queue[issue.identifier] = entry
        self._schedule_retry_timer(issue.identifier, entry.token, backoff_ms / 1000.0)

    def _schedule_retry_timer(self, identifier: str, token: str, delay_seconds: float) -> None:
        """Fire an asyncio timer that triggers a poll after *delay_seconds*."""
        if identifier in self._retry_timers:
            _old_token, old_task = self._retry_timers[identifier]
            old_task.cancel()

        async def _timer() -> None:
            try:
                await asyncio.sleep(delay_seconds)
            except asyncio.CancelledError:
                return
            entry = self._retry_queue.get(identifier)
            if entry is not None and entry.token == token:
                self.request_refresh()

        task = asyncio.create_task(_timer())
        self._retry_timers[identifier] = (token, task)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch_candidates(self) -> None:
        """Fetch candidates, sort by priority, dispatch eligible."""
        candidates = await self._tracker.fetch_candidate_issues()
        if not candidates:
            return

        def sort_key(issue: Issue) -> tuple[int, datetime, str]:
            pri = issue.priority if issue.priority is not None else 5
            created = issue.created_at or datetime.min.replace(tzinfo=timezone.utc)
            return (pri, created, issue.identifier)

        candidates.sort(key=sort_key)

        for issue in candidates:
            if not self._is_dispatch_eligible(issue):
                continue
            if not self._has_global_capacity():
                break
            if not self._has_state_capacity(issue.state):
                continue
            if not self._has_host_capacity():
                continue

            # Re-validate: re-fetch the issue to pick up any state or blocker
            # changes that occurred since the initial fetch.
            fresh = await self._revalidate_issue(issue)
            if fresh is None:
                continue

            if not self._is_dispatch_eligible(fresh):
                continue

            await self._dispatch_issue(fresh)

    async def _revalidate_issue(self, issue: Issue) -> Issue | None:
        """Re-fetch the latest issue data before dispatch.

        Returns the updated issue, or ``None`` if it should be skipped
        (disappeared or moved to terminal state).
        """
        # First check state is still valid
        states = await self._tracker.fetch_issue_states_by_ids([issue.identifier])
        current_state = states.get(issue.identifier)
        if current_state is None or self._tracker.is_terminal_state(current_state):
            return None

        # Re-fetch the full issue from the tracker to pick up any blocker
        # or assignment changes.
        refreshed = await self._tracker.fetch_candidate_issues()
        refreshed_map = {i.identifier: i for i in refreshed}
        if issue.identifier in refreshed_map:
            fresh_issue = refreshed_map[issue.identifier]
            fresh_issue.state = current_state
            return fresh_issue

        # If the issue is no longer a candidate, use the original with
        # updated state.
        issue.state = current_state
        return issue

    def _is_dispatch_eligible(self, issue: Issue) -> bool:
        """True if the issue can be dispatched right now."""
        ident = issue.identifier

        if ident in self._running:
            return False
        if ident in self._completed:
            return False
        if ident in self._retry_queue:
            return False

        # Todo with a non-terminal blocker => skip
        blockers = getattr(issue, "blocked_by", None) or getattr(issue, "blockers", None) or []
        if issue.state == "Todo" and blockers:
            has_non_terminal = False
            for b in blockers:
                if hasattr(b, "is_terminal"):
                    if not b.is_terminal:
                        has_non_terminal = True
                        break
                else:
                    if not self._tracker.is_terminal_state(b.state):
                        has_non_terminal = True
                        break
            if has_non_terminal:
                return False

        # Assigned to a different worker
        worker_name = _get_worker_name(self._config)
        if worker_name:
            if hasattr(issue, "assigned_to_worker"):
                if not issue.assigned_to_worker:
                    return False
            else:
                assignee = getattr(issue, "assignee", None) or getattr(issue, "assignee_id", None)
                if assignee and assignee != worker_name:
                    return False

        return True

    def _has_global_capacity(self) -> bool:
        return len(self._running) < _get_max_concurrent(self._config)

    def _has_state_capacity(self, state: str) -> bool:
        limits = _get_max_concurrent_by_state(self._config)
        if not limits:
            return True
        # Keys may be normalized to lowercase by the config validator.
        limit = limits.get(state) or limits.get(state.lower())
        if limit is None:
            return True
        count = sum(
            1 for e in self._running.values() if e.issue.state == state or e.issue.state.lower() == state.lower()
        )
        return count < limit

    def _has_host_capacity(self, host: str | None = None) -> bool:
        """True if at least one SSH host has capacity (or no hosts configured)."""
        hosts = _get_ssh_hosts(self._config)
        if not hosts:
            return True
        if host:
            count = sum(1 for e in self._running.values() if e.worker_host == host)
            return count < _get_max_per_host(self._config)
        return self._pick_host() is not None

    def _pick_host(self, preferred: str | None = None) -> str | None:
        """Return the least-loaded SSH host with available capacity."""
        hosts = _get_ssh_hosts(self._config)
        if not hosts:
            return None

        max_per = _get_max_per_host(self._config)
        load: dict[str, int] = {h: 0 for h in hosts}
        for entry in self._running.values():
            if entry.worker_host and entry.worker_host in load:
                load[entry.worker_host] += 1

        if preferred and preferred in load and load[preferred] < max_per:
            return preferred

        available = [(cnt, h) for h, cnt in load.items() if cnt < max_per]
        if not available:
            return None
        available.sort()
        return available[0][1]

    async def _dispatch_issue(
        self,
        issue: Issue,
        *,
        preferred_host: str | None = None,
        attempt: int = 0,
    ) -> None:
        """Create workspace and launch an agent task for *issue*."""
        hosts = _get_ssh_hosts(self._config)
        worker_host: str | None = None
        if hosts:
            worker_host = self._pick_host(preferred=preferred_host)
            if worker_host is None:
                logger.warning("no_worker_capacity: cannot dispatch %s", issue.identifier)
                return

        ws_result = await self._workspace.create(issue.identifier)
        if isinstance(ws_result, (str, Path)):
            workspace_path = str(ws_result)
        else:
            logger.error(
                "Workspace creation failed for %s: %s",
                issue.identifier,
                ws_result,
            )
            return

        task = asyncio.create_task(self._run_agent(issue, workspace_path, worker_host, attempt))

        entry = RunningEntry(
            issue=issue,
            task=task,
            started_at=datetime.now(timezone.utc),
            worker_host=worker_host,
            workspace_path=workspace_path,
            attempt=attempt,
        )
        self._running[issue.identifier] = entry

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        issue: Issue,
        workspace_path: str | None,
        worker_host: str | None,
        attempt: int,
    ) -> None:
        """Execute an agent runner and handle the outcome."""
        identifier = issue.identifier
        result: AgentResult | None = None
        try:
            result = await self._agent_runner_factory(issue, workspace_path, worker_host)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            result = AgentResult(error=str(exc))

        entry = self._running.pop(identifier, None)
        if entry is None:
            return

        if result:
            self._update_codex_totals(entry, result)

        if result and result.error:
            next_attempt = attempt + 1
            self._schedule_failure_retry(
                issue,
                attempt=next_attempt,
                error=result.error,
                preferred_host=worker_host,
            )
        else:
            # Check if validation gating should block completion.
            validation_cfg = getattr(self._config, "validation", None)
            validation_required = (
                validation_cfg is not None
                and getattr(validation_cfg, "enabled", False)
                and getattr(validation_cfg, "required_for_completion", True)
            )

            if validation_required and result and not result.validation_passed:
                # Validation was required but did not pass — treat as failure.
                next_attempt = attempt + 1
                truncated_output = (result.validation_output or "")[:500]
                self._schedule_failure_retry(
                    issue,
                    attempt=next_attempt,
                    error=f"validation_failed: {truncated_output}",
                    preferred_host=worker_host,
                )
            else:
                self._completed.add(identifier)
                states = await self._tracker.fetch_issue_states_by_ids([identifier])
                current_state = states.get(identifier)
                if current_state and self._tracker.is_active_state(current_state):
                    self._completed.discard(identifier)
                    self._schedule_continuation_retry(issue, preferred_host=worker_host)

    def _update_codex_totals(self, entry: RunningEntry, result: AgentResult) -> None:
        """Merge agent result into the running entry and global totals."""
        if result.session_id:
            entry.session_id = result.session_id
        entry.turn_count += 1
        entry.last_event = result.last_event
        entry.last_event_at = datetime.now(timezone.utc)
        entry.input_tokens += result.input_tokens
        entry.output_tokens += result.output_tokens
        entry.total_tokens += result.total_tokens

        self._codex_totals["input_tokens"] += result.input_tokens
        self._codex_totals["output_tokens"] += result.output_tokens
        self._codex_totals["total_tokens"] += result.total_tokens

        elapsed = (datetime.now(timezone.utc) - entry.started_at).total_seconds()
        self._codex_totals["seconds_running"] += elapsed

        if result.rate_limits is not None:
            self._codex_rate_limits = result.rate_limits

    # ------------------------------------------------------------------
    # Internal helpers exposed for testing
    # ------------------------------------------------------------------

    @property
    def running(self) -> dict[str, RunningEntry]:
        return self._running

    @property
    def retry_queue(self) -> dict[str, RetryEntry]:
        return self._retry_queue

    @property
    def completed(self) -> set[str]:
        return self._completed
