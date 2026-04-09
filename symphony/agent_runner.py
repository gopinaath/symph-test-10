"""Executes a single issue autonomously through the Codex app-server."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from symphony.codex.app_server import AppServer, AppServerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward-reference type stubs (real implementations live elsewhere)
# ---------------------------------------------------------------------------


class Issue(Protocol):  # symphony.models.Issue
    id: str
    title: str
    state: str

    def is_terminal(self) -> bool: ...


class Workspace(Protocol):  # symphony.workspace.Workspace
    path: str

    async def prepare(self) -> None: ...

    async def cleanup(self) -> None: ...


class PromptBuilder(Protocol):  # symphony.prompt_builder.PromptBuilder
    def initial_prompt(self, issue: Any) -> str: ...

    def continuation_prompt(self, issue: Any, turn: int) -> str: ...


# ---------------------------------------------------------------------------
# Update callback protocol
# ---------------------------------------------------------------------------


class RunUpdateCallback(Protocol):
    """Callback the orchestrator passes in to receive progress updates."""

    async def session_started(self, session_id: str) -> None: ...

    async def turn_completed(self, turn_number: int, usage: dict[str, Any]) -> None: ...

    async def codex_update(self, event_type: str, payload: Any) -> None: ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkspacePrepareFailedError(Exception):
    """SSH or local workspace preparation failed."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentRunnerConfig:
    max_turns: int = 5
    app_server_config: AppServerConfig = field(default_factory=AppServerConfig)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    issue_id: str
    turns_executed: int
    total_usage: dict[str, Any] = field(default_factory=dict)
    stopped_reason: str = ""


# ---------------------------------------------------------------------------
# Hook protocol
# ---------------------------------------------------------------------------


class RunHook(Protocol):
    async def before_run(self, issue: Any, workspace: Any) -> None: ...

    async def after_run(self, issue: Any, workspace: Any, result: RunResult) -> None: ...


class _NullHook:
    async def before_run(self, issue: Any, workspace: Any) -> None:
        pass

    async def after_run(self, issue: Any, workspace: Any, result: RunResult) -> None:
        pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AgentRunner:
    """Runs a single issue through one or more Codex turns."""

    def __init__(
        self,
        config: AgentRunnerConfig | None = None,
        *,
        prompt_builder: PromptBuilder | None = None,
        hook: RunHook | None = None,
        app_server_factory: Callable[..., AppServer] | None = None,
    ) -> None:
        self._config = config or AgentRunnerConfig()
        self._prompt_builder = prompt_builder
        self._hook: RunHook = hook or _NullHook()
        self._app_server_factory = app_server_factory or self._default_app_server

    # -- public API ----------------------------------------------------------

    async def run(
        self,
        issue: Any,
        workspace: Any,
        *,
        callback: RunUpdateCallback | None = None,
        reuse_workspace: bool = False,
    ) -> RunResult:
        """Execute the issue end-to-end and return the result."""

        # 1. Prepare workspace.
        if not reuse_workspace:
            try:
                await workspace.prepare()
            except Exception as exc:
                raise WorkspacePrepareFailedError(str(exc)) from exc

        # 2. Before-run hook.
        await self._hook.before_run(issue, workspace)

        # 3. Start app-server session.
        events: list[tuple[str, Any]] = []

        async def _on_event(etype: str, payload: Any) -> None:
            ts = time.time()
            events.append((etype, payload))
            if callback:
                await callback.codex_update(etype, {"ts": ts, **({} if not isinstance(payload, dict) else payload)})

        server = self._app_server_factory(
            config=self._config.app_server_config,
            on_event=_on_event,
        )
        try:
            await server.start(workspace.path)
            session_id = await server.start_thread(workspace.path)
            if callback:
                await callback.session_started(session_id)

            # 4. Turn loop.
            total_usage: dict[str, Any] = {}
            turns_executed = 0
            stopped_reason = "max_turns"

            for turn_number in range(1, self._config.max_turns + 1):
                # Check if the issue has moved to a terminal state.
                if turn_number > 1 and issue.is_terminal():
                    stopped_reason = "terminal_state"
                    break

                # Build prompt.
                if self._prompt_builder is not None:
                    if turn_number == 1:
                        prompt = self._prompt_builder.initial_prompt(issue)
                    else:
                        prompt = self._prompt_builder.continuation_prompt(issue, turn_number)
                else:
                    prompt = issue.title if turn_number == 1 else f"Continue working on: {issue.title}"

                # Execute turn.
                result = await server.run_turn(
                    input_text=prompt,
                    cwd=workspace.path,
                    title=issue.title,
                )
                turn_usage = result.get("usage", {})
                turns_executed = turn_number
                _merge_usage(total_usage, turn_usage)

                if callback:
                    await callback.turn_completed(turn_number, turn_usage)

            result_obj = RunResult(
                issue_id=issue.id,
                turns_executed=turns_executed,
                total_usage=total_usage,
                stopped_reason=stopped_reason,
            )
        finally:
            await server.stop()

        # 8. After-run hook.
        await self._hook.after_run(issue, workspace, result_obj)

        return result_obj

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _default_app_server(
        config: AppServerConfig | None = None,
        on_event: Any = None,
    ) -> AppServer:
        return AppServer(config, on_event=on_event)


def _merge_usage(total: dict[str, Any], turn: dict[str, Any]) -> None:
    for k, v in turn.items():
        if isinstance(v, (int, float)):
            total[k] = total.get(k, 0) + v
        else:
            total[k] = v
