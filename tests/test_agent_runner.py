"""Tests for symphony.agent_runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from symphony.agent_runner import (
    AgentRunner,
    AgentRunnerConfig,
    RunResult,
    WorkspacePrepareFailedError,
)
from symphony.codex.app_server import AppServer

# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------


@dataclass
class FakeIssue:
    id: str = "ISSUE-1"
    title: str = "Fix the bug"
    state: str = "active"
    _terminal_after_turn: int | None = None

    def is_terminal(self) -> bool:
        return self.state == "terminal"


@dataclass
class FakeWorkspace:
    path: str = "/tmp/workspace"
    prepared: bool = False
    cleaned: bool = False
    _prepare_error: Exception | None = None

    async def prepare(self) -> None:
        if self._prepare_error:
            raise self._prepare_error
        self.prepared = True

    async def cleanup(self) -> None:
        self.cleaned = True


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


class FakePromptBuilder:
    def initial_prompt(self, issue: Any) -> str:
        return f"Initial: {issue.title}"

    def continuation_prompt(self, issue: Any, turn: int) -> str:
        return f"Continue (turn {turn}): {issue.title}"


# ---------------------------------------------------------------------------
# Helpers
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
        # Wire up on_event so the runner can push events through.
        server._on_event = on_event
        return server

    return factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentRunnerKeepsWorkspace:
    """agent runner keeps workspace after successful codex run."""

    @pytest.mark.asyncio
    async def test_keeps_workspace_after_successful_run(self):
        issue = FakeIssue()
        ws = FakeWorkspace()
        server = _fake_app_server()
        cfg = AgentRunnerConfig(max_turns=1)

        runner = AgentRunner(cfg, app_server_factory=_factory_returning(server))
        result = await runner.run(issue, ws)

        assert isinstance(result, RunResult)
        assert result.issue_id == "ISSUE-1"
        assert result.turns_executed == 1
        # Workspace should NOT have been cleaned up by the runner itself.
        assert ws.cleaned is False


class TestAgentRunnerForwardsUpdates:
    """agent runner forwards timestamped codex updates to recipient."""

    @pytest.mark.asyncio
    async def test_forwards_timestamped_codex_updates(self):
        issue = FakeIssue()
        ws = FakeWorkspace()
        callback = FakeCallback()
        server = _fake_app_server(session_id="sess-99")
        cfg = AgentRunnerConfig(max_turns=1)

        runner = AgentRunner(cfg, app_server_factory=_factory_returning(server))
        await runner.run(issue, ws, callback=callback)

        assert "sess-99" in callback.sessions
        assert len(callback.turns) == 1
        assert callback.turns[0][0] == 1  # turn number


class TestAgentRunnerSSHFailure:
    """agent runner surfaces ssh startup failures."""

    @pytest.mark.asyncio
    async def test_surfaces_ssh_startup_failures(self):
        issue = FakeIssue()
        ws = FakeWorkspace(_prepare_error=OSError("ssh: Connection refused"))
        cfg = AgentRunnerConfig(max_turns=1)

        runner = AgentRunner(cfg)
        with pytest.raises(WorkspacePrepareFailedError, match="Connection refused"):
            await runner.run(issue, ws)


class TestAgentRunnerContinuation:
    """agent runner continues with follow-up turn while issue remains active."""

    @pytest.mark.asyncio
    async def test_continues_with_follow_up_turns_while_active(self):
        issue = FakeIssue(state="active")
        ws = FakeWorkspace()
        callback = FakeCallback()
        server = _fake_app_server()
        cfg = AgentRunnerConfig(max_turns=3)

        runner = AgentRunner(
            cfg,
            prompt_builder=FakePromptBuilder(),
            app_server_factory=_factory_returning(server),
        )
        result = await runner.run(issue, ws, callback=callback)

        assert result.turns_executed == 3
        assert len(callback.turns) == 3

        # Verify the first call used initial prompt and subsequent ones used
        # continuation prompt.
        calls = server.run_turn.call_args_list
        assert "Initial:" in calls[0].kwargs["input_text"]
        assert "Continue (turn 2):" in calls[1].kwargs["input_text"]
        assert "Continue (turn 3):" in calls[2].kwargs["input_text"]


class TestAgentRunnerMaxTurns:
    """agent runner stops continuing once max_turns reached."""

    @pytest.mark.asyncio
    async def test_stops_once_max_turns_reached(self):
        issue = FakeIssue(state="active")
        ws = FakeWorkspace()
        server = _fake_app_server()
        cfg = AgentRunnerConfig(max_turns=2)

        runner = AgentRunner(cfg, app_server_factory=_factory_returning(server))
        result = await runner.run(issue, ws)

        assert result.turns_executed == 2
        assert result.stopped_reason == "max_turns"
        assert server.run_turn.call_count == 2

    @pytest.mark.asyncio
    async def test_stops_early_when_issue_becomes_terminal(self):
        """Variant: issue transitions to terminal state between turns."""
        issue = FakeIssue(state="active")
        ws = FakeWorkspace()
        server = _fake_app_server()
        cfg = AgentRunnerConfig(max_turns=5)

        # After the first turn completes, mark issue terminal.
        original_run_turn = server.run_turn

        call_count = 0

        async def _run_turn_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            result = await original_run_turn(**kwargs)
            if call_count >= 1:
                issue.state = "terminal"
            return result

        server.run_turn = AsyncMock(side_effect=_run_turn_side_effect)

        runner = AgentRunner(cfg, app_server_factory=_factory_returning(server))
        result = await runner.run(issue, ws)

        # Only turn 1 should have executed — turn 2 checks is_terminal() first.
        assert result.turns_executed == 1
        assert result.stopped_reason == "terminal_state"
