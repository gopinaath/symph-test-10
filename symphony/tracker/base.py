"""Abstract base class for issue trackers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from symphony.models import Issue


class Tracker(ABC):
    """Interface for issue tracking systems (Linear, GitHub, etc.)."""

    @abstractmethod
    async def fetch_candidate_issues(self) -> list[Issue]:
        """Fetch issues eligible for processing (typically Todo/InProgress)."""
        ...

    @abstractmethod
    async def fetch_issues_by_states(self, states: list[str]) -> list[Issue]:
        """Fetch issues filtered by given state names."""
        ...

    @abstractmethod
    async def fetch_issue_states_by_ids(
        self, identifiers: list[str]
    ) -> dict[str, Optional[str]]:
        """Return {identifier: current_state} for the given issue IDs.

        If an issue is not found, its value should be None.
        """
        ...

    @abstractmethod
    async def create_comment(self, identifier: str, body: str) -> None:
        """Post a comment on the issue."""
        ...

    @abstractmethod
    async def update_issue_state(self, identifier: str, state: str) -> None:
        """Transition the issue to a new state."""
        ...

    # --- helpers used by orchestrator ---

    def is_active_state(self, state: str) -> bool:
        """Return True if the state means the issue is being worked on."""
        return state.lower() in ("inprogress", "in_progress", "in progress")

    def is_terminal_state(self, state: str) -> bool:
        """Return True if the state means the issue is finished."""
        return state.lower() in ("done", "cancelled", "canceled", "closed")
