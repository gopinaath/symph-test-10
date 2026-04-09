"""In-memory tracker for testing."""

from __future__ import annotations

from copy import deepcopy
from typing import Awaitable, Callable, Optional

from symphony.models import Issue
from symphony.tracker.base import Tracker

# Callback signature: (issue_id, old_state, new_state) -> Awaitable[None]
StateChangeCallback = Callable[[str, str, str], Awaitable[None]]


class MemoryTracker(Tracker):
    """An in-memory issue tracker for unit / integration tests."""

    def __init__(
        self,
        issues: Optional[list[Issue]] = None,
        active_states: Optional[set[str]] = None,
        terminal_states: Optional[set[str]] = None,
        candidate_states: Optional[set[str]] = None,
        worker_name: Optional[str] = None,
        on_state_change: Optional[StateChangeCallback] = None,
    ) -> None:
        self.issues: dict[str, Issue] = {}
        self.comments: dict[str, list[str]] = {}
        self._active_states = active_states or {"InProgress", "in_progress"}
        self._terminal_states = terminal_states or {"Done", "Cancelled", "Canceled"}
        self._candidate_states = candidate_states or {
            "Todo",
            "InProgress",
            "in_progress",
        }
        self._worker_name = worker_name
        self._on_state_change = on_state_change
        if issues:
            for issue in issues:
                self.issues[issue.identifier] = deepcopy(issue)

    def add_issue(self, issue: Issue) -> None:
        self.issues[issue.identifier] = deepcopy(issue)

    def remove_issue(self, identifier: str) -> None:
        self.issues.pop(identifier, None)

    def set_issue_state(self, identifier: str, state: str) -> None:
        if identifier in self.issues:
            self.issues[identifier].state = state

    def set_issue_assignee(self, identifier: str, assignee: Optional[str]) -> None:
        if identifier in self.issues:
            issue = self.issues[identifier]
            if hasattr(issue, "assignee_id"):
                issue.assignee_id = assignee
            if hasattr(issue, "assigned_to_worker"):
                # When assignee is set to a value, mark as assigned; None = unassigned
                issue.assigned_to_worker = assignee is not None
            if hasattr(issue, "assignee"):
                issue.assignee = assignee

    # --- Tracker interface ---

    async def fetch_candidate_issues(self) -> list[Issue]:
        return [
            deepcopy(issue)
            for issue in self.issues.values()
            if issue.state in self._candidate_states
        ]

    async def fetch_issues_by_states(self, states: list[str]) -> list[Issue]:
        if not states:
            return []
        state_set = set(states)
        return [
            deepcopy(issue)
            for issue in self.issues.values()
            if issue.state in state_set
        ]

    async def fetch_issue_states_by_ids(
        self, identifiers: list[str]
    ) -> dict[str, Optional[str]]:
        result: dict[str, Optional[str]] = {}
        for ident in identifiers:
            issue = self.issues.get(ident)
            result[ident] = issue.state if issue else None
        return result

    async def create_comment(self, identifier: str, body: str) -> None:
        self.comments.setdefault(identifier, []).append(body)

    async def update_issue_state(self, identifier: str, state: str) -> None:
        if identifier in self.issues:
            old_state = self.issues[identifier].state
            self.issues[identifier].state = state
            if self._on_state_change is not None and old_state != state:
                await self._on_state_change(identifier, old_state, state)

    # --- helpers ---

    def is_active_state(self, state: str) -> bool:
        return state in self._active_states

    def is_terminal_state(self, state: str) -> bool:
        return state in self._terminal_states
