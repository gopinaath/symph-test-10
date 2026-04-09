"""Tests for Symphony tracker implementations."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from symphony.models import Issue
from symphony.tracker.github import GitHubTracker
from symphony.tracker.memory import MemoryTracker

# =========================================================================
# MemoryTracker tests
# =========================================================================


def _make_issue(identifier: str, state: str = "Todo", **kw: Any) -> Issue:
    defaults = dict(
        id=kw.pop("id", identifier),
        description=kw.pop("description", ""),
        priority=kw.pop("priority", None),
        branch_name=kw.pop("branch_name", ""),
        url=kw.pop("url", ""),
        assignee_id=kw.pop("assignee_id", None),
    )
    return Issue(identifier=identifier, title=f"Issue {identifier}", state=state, **defaults, **kw)


class TestMemoryTracker:
    @pytest.mark.asyncio
    async def test_add_and_fetch_candidate(self) -> None:
        t = MemoryTracker()
        t.add_issue(_make_issue("1", state="Todo"))
        t.add_issue(_make_issue("2", state="Done"))
        candidates = await t.fetch_candidate_issues()
        assert [c.identifier for c in candidates] == ["1"]

    @pytest.mark.asyncio
    async def test_fetch_by_states(self) -> None:
        t = MemoryTracker()
        t.add_issue(_make_issue("1", state="Todo"))
        t.add_issue(_make_issue("2", state="InProgress"))
        t.add_issue(_make_issue("3", state="Done"))

        result = await t.fetch_issues_by_states(["Todo", "Done"])
        ids = {iss.identifier for iss in result}
        assert ids == {"1", "3"}

    @pytest.mark.asyncio
    async def test_fetch_by_states_empty_is_noop(self) -> None:
        t = MemoryTracker()
        t.add_issue(_make_issue("1"))
        result = await t.fetch_issues_by_states([])
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_issue_states_by_ids(self) -> None:
        t = MemoryTracker()
        t.add_issue(_make_issue("a", state="Todo"))
        t.add_issue(_make_issue("b", state="Done"))
        states = await t.fetch_issue_states_by_ids(["a", "b", "missing"])
        assert states == {"a": "Todo", "b": "Done", "missing": None}

    @pytest.mark.asyncio
    async def test_update_state(self) -> None:
        t = MemoryTracker()
        t.add_issue(_make_issue("x", state="Todo"))
        await t.update_issue_state("x", "InProgress")
        assert t.issues["x"].state == "InProgress"

    @pytest.mark.asyncio
    async def test_create_comment(self) -> None:
        t = MemoryTracker()
        t.add_issue(_make_issue("c"))
        await t.create_comment("c", "hello")
        await t.create_comment("c", "world")
        assert t.comments["c"] == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_state_change_callback(self) -> None:
        events: list[tuple[str, str, str]] = []

        async def callback(ident: str, old: str, new: str) -> None:
            events.append((ident, old, new))

        t = MemoryTracker(on_state_change=callback)
        t.add_issue(_make_issue("e", state="Todo"))
        await t.update_issue_state("e", "InProgress")
        await t.update_issue_state("e", "InProgress")  # same state, no event
        await t.update_issue_state("e", "Done")
        assert events == [("e", "Todo", "InProgress"), ("e", "InProgress", "Done")]


# =========================================================================
# GitHubTracker tests (mocked HTTP via respx)
# =========================================================================


def _gh_issue(
    number: int,
    title: str = "Test",
    state: str = "open",
    labels: list[str] | None = None,
    body: str = "",
    assignee: str | None = None,
) -> dict[str, Any]:
    """Build a minimal GitHub issue JSON payload."""
    return {
        "number": number,
        "title": title,
        "state": state,
        "labels": [{"name": lbl} for lbl in (labels or [])],
        "body": body,
        "assignee": {"login": assignee} if assignee else None,
        "html_url": f"https://github.com/o/r/issues/{number}",
    }


@pytest.fixture
def gh_tracker() -> GitHubTracker:
    client = httpx.AsyncClient(
        base_url="https://api.github.com",
        headers={"Authorization": "Bearer fake"},
    )
    return GitHubTracker(
        owner="o",
        repo="r",
        token="fake",
        candidate_states=["open", "todo"],
        state_labels={"todo": "Todo", "in_progress": "In Progress"},
        client=client,
    )


class TestGitHubTrackerFetch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_candidate_issues(self, gh_tracker: GitHubTracker) -> None:
        respx.get(
            "https://api.github.com/repos/o/r/issues",
            params__contains={"labels": "open"},
        ).respond(json=[_gh_issue(1, labels=["open"])])

        respx.get(
            "https://api.github.com/repos/o/r/issues",
            params__contains={"labels": "Todo"},
        ).respond(json=[_gh_issue(2, labels=["Todo"])])

        issues = await gh_tracker.fetch_candidate_issues()
        ids = {i.identifier for i in issues}
        assert "1" in ids
        assert "2" in ids

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_issues_by_states_empty(self, gh_tracker: GitHubTracker) -> None:
        result = await gh_tracker.fetch_issues_by_states([])
        assert result == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_extract_priority_and_blocked(self, gh_tracker: GitHubTracker) -> None:
        respx.get(
            "https://api.github.com/repos/o/r/issues",
            params__contains={"labels": "open"},
        ).respond(
            json=[
                _gh_issue(
                    5,
                    labels=["open", "priority:2"],
                    body="Blocked by #3 and blocked by #7",
                )
            ]
        )

        respx.get(
            "https://api.github.com/repos/o/r/issues",
            params__contains={"labels": "Todo"},
        ).respond(json=[])

        issues = await gh_tracker.fetch_candidate_issues()
        assert len(issues) == 1
        assert issues[0].priority == 2
        assert [b.identifier for b in issues[0].blocked_by] == ["3", "7"]


class TestGitHubTrackerComment:
    @pytest.mark.asyncio
    @respx.mock
    async def test_create_comment(self, gh_tracker: GitHubTracker) -> None:
        respx.post("https://api.github.com/repos/o/r/issues/10/comments").respond(status_code=201)

        await gh_tracker.create_comment("10", "nice work")


class TestGitHubTrackerState:
    @pytest.mark.asyncio
    @respx.mock
    async def test_update_issue_state(self, gh_tracker: GitHubTracker) -> None:
        respx.get("https://api.github.com/repos/o/r/issues/10").respond(json=_gh_issue(10, labels=["open", "bug"]))
        respx.patch("https://api.github.com/repos/o/r/issues/10").respond(
            status_code=200, json=_gh_issue(10, labels=["In Progress", "bug"])
        )

        await gh_tracker.update_issue_state("10", "in_progress")

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_issue_states_by_ids(self, gh_tracker: GitHubTracker) -> None:
        respx.get("https://api.github.com/repos/o/r/issues/1").respond(json=_gh_issue(1, labels=["open"]))
        respx.get("https://api.github.com/repos/o/r/issues/999").respond(status_code=404)

        states = await gh_tracker.fetch_issue_states_by_ids(["1", "999"])
        assert states["1"] == "open"
        assert states["999"] is None


class TestGitHubTrackerPagination:
    @pytest.mark.asyncio
    @respx.mock
    async def test_pagination(self, gh_tracker: GitHubTracker) -> None:
        """Two pages of results followed by an empty third page."""
        page1 = [_gh_issue(i, labels=["open"]) for i in range(1, 101)]
        page2 = [_gh_issue(101, labels=["open"])]

        route = respx.get(
            "https://api.github.com/repos/o/r/issues",
            params__contains={"labels": "open"},
        )

        # respx returns responses in order when side_effect is used.
        route.side_effect = [
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ]

        # The second candidate state "todo" returns nothing.
        respx.get(
            "https://api.github.com/repos/o/r/issues",
            params__contains={"labels": "Todo"},
        ).respond(json=[])

        issues = await gh_tracker.fetch_candidate_issues()
        assert len(issues) == 101
