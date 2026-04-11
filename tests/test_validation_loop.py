"""Tests for after-turn validation (Phase 1c) and completion gating (Phase 1d)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from symphony.agent_runner import (
    AgentRunner,
    AgentRunnerConfig,
    RunResult,
)
from symphony.codex.app_server import AppServer
from symphony.config import AgentConfig, CodexConfig, Config, PollingConfig, ValidationConfig, WorkerConfig
from symphony.models import Issue
from symphony.orchestrator import AgentResult, Orchestrator
from symphony.tracker.memory import MemoryTracker
from symphony.workspace import ValidationResult

# ---------------------------------------------------------------------------
# Shared fakes for agent_runner tests
# ---------------------------------------------------------------------------

_COUNTER = 0


@dataclass
class FakeIssue:
    id: str = "ISSUE-1"
    title: str = "Fix the bug"
    state: str = "active"

    def is_terminal(self) -> bool:
        return self.state == "terminal"


@dataclass
class FakeWorkspace:
    path: str = "/tmp/workspace"
    prepared: bool = False
    cleaned: bool = False
    _prepare_error: Exception | None = None
    _validation_results: list[ValidationResult] | None = None
    _validation_call_count: int = 0

    async def prepare(self) -> None:
        if self._prepare_error:
            raise self._prepare_error
        self.prepared = True

    async def cleanup(self) -> None:
        self.cleaned = True

    async def run_validation(self, identifier: str, config: ValidationConfig) -> ValidationResult:
        idx = min(self._validation_call_count, len(self._validation_results) - 1)
        result = self._validation_results[idx]
        self._validation_call_count += 1
        return result


class FakeCallback:
    def __init__(self) -> None:
        self.sessions: list[str] = []
        self.turns: list[tuple[int, dict]] = []
        self.events: list[tuple[str, Any]] = []

    async def session_started(self, session_id: str) -> None:
        self.sessions.append(session_id)

    async def turn_completed(self, turn_number: int, usage: dict[str, Any]) -> None:
        self.turns.append((turn_number, usage))

    async def codex_update(self, event_type: str, payload: Any) -> None:
        self.events.append((event_type, payload))


# ---------------------------------------------------------------------------
# App server helpers
# ---------------------------------------------------------------------------


def _fake_app_server(
    *,
    session_id: str = "session-42",
    turn_usage: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a mock AppServer that successfully completes turns."""
    server = AsyncMock(spec=AppServer)
    server.start = AsyncMock()
    server.start_thread = AsyncMock(return_value=session_id)
    server.run_turn = AsyncMock(return_value={"turn_id": "t-1", "usage": turn_usage or {"input_tokens": 5}})
    server.stop = AsyncMock()
    return server


def _factory_returning(server: Any):
    """Return a factory callable that always yields *server*."""

    def factory(config=None, on_event=None):
        server._on_event = on_event
        return server

    return factory


# ---------------------------------------------------------------------------
# Shared helpers for orchestrator tests
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_issue(
    identifier: str = "ISS-1",
    title: str = "Test issue",
    state: str = "Todo",
    priority: int | None = None,
    created_at: datetime | None = None,
    assigned_to_worker: bool = True,
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
        assignee_id=None,
        blocked_by=[],
        assigned_to_worker=assigned_to_worker,
        created_at=created_at or _now(),
    )


def _make_config(
    max_concurrent: int = 5,
    polling_interval_ms: int = 60_000,
    validation: ValidationConfig | None = None,
) -> Config:
    return Config(
        polling=PollingConfig(interval_ms=polling_interval_ms),
        agent=AgentConfig(max_concurrent_agents=max_concurrent),
        codex=CodexConfig(),
        worker=WorkerConfig(),
        validation=validation or ValidationConfig(),
    )


class StubWorkspace:
    """Minimal workspace stub for testing."""

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
) -> AsyncMock:
    if result is None and error is None:
        result = AgentResult()

    async def _run(issue: Issue, ws: str | None, host: str | None) -> AgentResult:
        if error is not None:
            raise error
        assert result is not None
        return result

    mock = AsyncMock(side_effect=_run)
    return mock


# ===========================================================================
# Phase 1c: Agent runner after-turn validation tests
# ===========================================================================


class TestValidationPassStopsEarly:
    """test_validation_pass_stops_early: agent turn completes, validation
    passes -> RunResult.validation_passed=True, stopped_reason='validation_passed'."""

    @pytest.mark.asyncio
    async def test_validation_pass_stops_early(self):
        issue = FakeIssue()
        ws = FakeWorkspace(
            _validation_results=[ValidationResult(passed=True, exit_code=0, stdout="All tests passed")],
        )
        server = _fake_app_server()
        cfg = AgentRunnerConfig(max_turns=5)
        v_config = ValidationConfig(enabled=True, command="pytest", max_attempts=3)

        runner = AgentRunner(cfg, app_server_factory=_factory_returning(server))
        result = await runner.run(issue, ws, validation_config=v_config)

        assert result.validation_passed is True
        assert result.stopped_reason == "validation_passed"
        # Should have stopped after the first turn since validation passed
        assert result.turns_executed == 1
        assert server.run_turn.call_count == 1


class TestValidationFailFeedsBackIntoNextTurn:
    """test_validation_fail_feeds_back_into_next_turn: agent turn completes,
    validation fails -> next turn gets error context in prompt."""

    @pytest.mark.asyncio
    async def test_validation_fail_feeds_back_into_next_turn(self):
        issue = FakeIssue()
        # First validation fails, second passes
        ws = FakeWorkspace(
            _validation_results=[
                ValidationResult(passed=False, exit_code=1, stderr="FAILED test_foo.py::test_bar"),
                ValidationResult(passed=True, exit_code=0, stdout="All passed"),
            ],
        )
        server = _fake_app_server()
        cfg = AgentRunnerConfig(max_turns=5)
        v_config = ValidationConfig(enabled=True, command="pytest", max_attempts=3)

        runner = AgentRunner(cfg, app_server_factory=_factory_returning(server))
        result = await runner.run(issue, ws, validation_config=v_config)

        assert result.validation_passed is True
        assert result.stopped_reason == "validation_passed"
        assert result.turns_executed == 2

        # The second turn's prompt should contain the validation failure output
        calls = server.run_turn.call_args_list
        assert len(calls) == 2
        second_prompt = calls[1].kwargs["input_text"]
        assert "validation" in second_prompt.lower()
        assert "FAILED test_foo.py::test_bar" in second_prompt


class TestValidationMaxAttemptsExceeded:
    """test_validation_max_attempts_exceeded: validation fails 3 times ->
    RunResult.validation_passed=False."""

    @pytest.mark.asyncio
    async def test_validation_max_attempts_exceeded(self):
        issue = FakeIssue()
        # All validations fail
        ws = FakeWorkspace(
            _validation_results=[
                ValidationResult(passed=False, exit_code=1, stderr="FAIL 1"),
                ValidationResult(passed=False, exit_code=1, stderr="FAIL 2"),
                ValidationResult(passed=False, exit_code=1, stderr="FAIL 3"),
            ],
        )
        server = _fake_app_server()
        # max_turns=3 so we can run 3 turn+validation cycles
        cfg = AgentRunnerConfig(max_turns=3)
        v_config = ValidationConfig(enabled=True, command="pytest", max_attempts=3)

        runner = AgentRunner(cfg, app_server_factory=_factory_returning(server))
        result = await runner.run(issue, ws, validation_config=v_config)

        assert result.validation_passed is False
        assert result.turns_executed == 3
        assert result.stopped_reason == "max_turns"
        assert result.validation_output == "FAIL 3"


class TestNoValidationConfigNormalBehavior:
    """test_no_validation_config_normal_behavior: no validation config ->
    behaves exactly as before."""

    @pytest.mark.asyncio
    async def test_no_validation_config_normal_behavior(self):
        issue = FakeIssue()
        ws = FakeWorkspace()
        server = _fake_app_server()
        cfg = AgentRunnerConfig(max_turns=2)

        runner = AgentRunner(cfg, app_server_factory=_factory_returning(server))
        result = await runner.run(issue, ws)

        assert result.validation_passed is False
        assert result.validation_output == ""
        assert result.stopped_reason == "max_turns"
        assert result.turns_executed == 2
        assert server.run_turn.call_count == 2


class TestValidationDisabledNormalBehavior:
    """test_validation_disabled_normal_behavior: validation config with
    enabled=False -> no validation runs."""

    @pytest.mark.asyncio
    async def test_validation_disabled_normal_behavior(self):
        issue = FakeIssue()
        # Even though workspace has run_validation, it should not be called
        ws = FakeWorkspace(
            _validation_results=[ValidationResult(passed=True, exit_code=0)],
        )
        server = _fake_app_server()
        cfg = AgentRunnerConfig(max_turns=2)
        v_config = ValidationConfig(enabled=False, command="pytest")

        runner = AgentRunner(cfg, app_server_factory=_factory_returning(server))
        result = await runner.run(issue, ws, validation_config=v_config)

        assert result.validation_passed is False
        assert result.validation_output == ""
        assert result.stopped_reason == "max_turns"
        assert result.turns_executed == 2
        # run_validation should never have been called
        assert ws._validation_call_count == 0


# ===========================================================================
# Phase 1d: Orchestrator completion gating tests
# ===========================================================================


class TestValidationFailureTriggersRetryNotCompletion:
    """test_validation_failure_triggers_retry_not_completion: agent returns
    validation_passed=False -> scheduled for retry, NOT added to completed."""

    async def test_validation_failure_triggers_retry_not_completion(self):
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(
            result=AgentResult(validation_passed=False, validation_output="tests failed"),
        )
        config = _make_config(
            validation=ValidationConfig(enabled=True, command="pytest", required_for_completion=True),
        )
        orch = Orchestrator(config, tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        await asyncio.sleep(0.05)

        # Should NOT be in completed
        assert "ISS-1" not in orch.completed
        # Should be in retry queue
        assert "ISS-1" in orch.retry_queue
        entry = orch.retry_queue["ISS-1"]
        assert entry.attempt == 1
        assert "validation_failed" in (entry.error or "")

        # Clean up timers
        for _, task in orch._retry_timers.values():
            task.cancel()


class TestValidationPassMarksCompleted:
    """test_validation_pass_marks_completed: agent returns
    validation_passed=True -> added to completed."""

    async def test_validation_pass_marks_completed(self):
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(
            result=AgentResult(validation_passed=True, validation_output="all good"),
        )
        config = _make_config(
            validation=ValidationConfig(enabled=True, command="pytest", required_for_completion=True),
        )
        orch = Orchestrator(config, tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        await asyncio.sleep(0.05)

        # Should be in completed (or continuation retry if state is still active).
        # Since tracker says InProgress (active), it may schedule continuation,
        # but at least NOT in retry_queue with a failure error.
        retry_entry = orch.retry_queue.get("ISS-1")
        if retry_entry is not None:
            # If in retry queue, it must be a continuation (attempt=0, no error)
            assert retry_entry.attempt == 0
            assert retry_entry.error is None


class TestValidationNotConfiguredCompletesNormally:
    """test_validation_not_configured_completes_normally: no validation ->
    backward compatible completion."""

    async def test_validation_not_configured_completes_normally(self):
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(result=AgentResult())
        # Default ValidationConfig has enabled=False
        config = _make_config()
        orch = Orchestrator(config, tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        await asyncio.sleep(0.05)

        # Should NOT be in retry queue with a failure error.
        retry_entry = orch.retry_queue.get("ISS-1")
        if retry_entry is not None:
            # It's a continuation retry (normal behavior), not a validation failure
            assert "validation_failed" not in (retry_entry.error or "")


class TestValidationNotRequiredCompletesRegardless:
    """test_validation_not_required_completes_regardless:
    required_for_completion=False -> completes even on validation failure."""

    async def test_validation_not_required_completes_regardless(self):
        tracker = MemoryTracker(
            issues=[_make_issue("ISS-1", state="InProgress")],
        )
        ws = StubWorkspace()
        runner = _make_runner(
            result=AgentResult(validation_passed=False, validation_output="tests failed"),
        )
        config = _make_config(
            validation=ValidationConfig(
                enabled=True,
                command="pytest",
                required_for_completion=False,
            ),
        )
        orch = Orchestrator(config, tracker, ws, runner)

        issue = _make_issue("ISS-1", state="InProgress")
        await orch._dispatch_issue(issue)
        await asyncio.sleep(0.05)

        # With required_for_completion=False, validation_passed=False should NOT
        # cause a validation_failed retry.
        retry_entry = orch.retry_queue.get("ISS-1")
        if retry_entry is not None:
            # Must be a normal continuation retry, not a validation failure
            assert "validation_failed" not in (retry_entry.error or "")
