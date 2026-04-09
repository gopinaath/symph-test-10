"""Tests for Symphony tracker implementations."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

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


# =========================================================================
# GitHubTracker retry and rate-limit tests
# =========================================================================


class TestGitHubTrackerRetry:
    @pytest.mark.asyncio
    @respx.mock
    async def test_retries_on_500_error(self, gh_tracker: GitHubTracker) -> None:
        """Mock 500 then 200 -- verify retry succeeds."""
        route = respx.get("https://api.github.com/repos/o/r/issues/42")
        route.side_effect = [
            httpx.Response(500, json={"message": "Internal Server Error"}),
            httpx.Response(200, json=_gh_issue(42, labels=["open"])),
        ]

        with patch("symphony.tracker.github.asyncio.sleep", return_value=None):
            resp = await gh_tracker._request_with_retry(
                "GET", "/repos/o/r/issues/42"
            )

        assert resp.status_code == 200
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_retries_on_429_with_retry_after(self, gh_tracker: GitHubTracker) -> None:
        """Mock 429 with Retry-After header, verify delay is respected."""
        route = respx.get("https://api.github.com/repos/o/r/issues/42")
        route.side_effect = [
            httpx.Response(
                429,
                json={"message": "rate limited"},
                headers={"Retry-After": "2"},
            ),
            httpx.Response(200, json=_gh_issue(42, labels=["open"])),
        ]

        sleep_durations: list[float] = []
        original_sleep = None

        async def mock_sleep(duration: float) -> None:
            sleep_durations.append(duration)

        with patch("symphony.tracker.github.asyncio.sleep", side_effect=mock_sleep):
            resp = await gh_tracker._request_with_retry(
                "GET", "/repos/o/r/issues/42"
            )

        assert resp.status_code == 200
        assert route.call_count == 2
        # The retry delay should be exactly 2.0 (from Retry-After header).
        assert any(d == 2.0 for d in sleep_durations)

    @pytest.mark.asyncio
    @respx.mock
    async def test_respects_rate_limit_remaining_zero(self, gh_tracker: GitHubTracker) -> None:
        """Mock response with X-RateLimit-Remaining=0, verify wait."""
        reset_time = str(int(time.time()) + 2)

        route = respx.get("https://api.github.com/repos/o/r/issues/42")
        route.side_effect = [
            httpx.Response(
                200,
                json=_gh_issue(42, labels=["open"]),
                headers={
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Limit": "5000",
                    "X-RateLimit-Reset": reset_time,
                },
            ),
        ]

        sleep_durations: list[float] = []

        async def mock_sleep(duration: float) -> None:
            sleep_durations.append(duration)

        with patch("symphony.tracker.github.asyncio.sleep", side_effect=mock_sleep):
            resp = await gh_tracker._request_with_retry(
                "GET", "/repos/o/r/issues/42"
            )

        assert resp.status_code == 200
        # Should have slept until rate limit reset.
        assert len(sleep_durations) >= 1
        assert sleep_durations[0] > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_gives_up_after_max_retries(self, gh_tracker: GitHubTracker) -> None:
        """Mock 3x 500, verify raises HTTPStatusError."""
        route = respx.get("https://api.github.com/repos/o/r/issues/42")
        route.side_effect = [
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(500, json={"message": "error"}),
            httpx.Response(500, json={"message": "error"}),
        ]

        with patch("symphony.tracker.github.asyncio.sleep", return_value=None):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await gh_tracker._request_with_retry(
                    "GET", "/repos/o/r/issues/42"
                )

        assert exc_info.value.response.status_code == 500
        assert route.call_count == 3

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_retry_on_4xx(self, gh_tracker: GitHubTracker) -> None:
        """Mock 404, verify immediate return (no retry)."""
        route = respx.get("https://api.github.com/repos/o/r/issues/999")
        route.respond(status_code=404, json={"message": "Not Found"})

        resp = await gh_tracker._request_with_retry(
            "GET", "/repos/o/r/issues/999"
        )

        assert resp.status_code == 404
        assert route.call_count == 1


# =========================================================================
# TEST-006: Pagination ordering, batched state fetch, unassigned issues
# =========================================================================


class TestGitHubTrackerPaginationEdgeCases:
    @pytest.mark.asyncio
    @respx.mock
    async def test_pagination_preserves_ordering(self, gh_tracker: GitHubTracker) -> None:
        """Multi-page results maintain insertion order across pages."""
        # Page 1: issues 1-100 (full page), Page 2: issues 101-110 (partial).
        page1 = [_gh_issue(i, title=f"Issue {i}", labels=["open"]) for i in range(1, 101)]
        page2 = [_gh_issue(i, title=f"Issue {i}", labels=["open"]) for i in range(101, 111)]

        route = respx.get(
            "https://api.github.com/repos/o/r/issues",
            params__contains={"labels": "open"},
        )
        route.side_effect = [
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ]

        respx.get(
            "https://api.github.com/repos/o/r/issues",
            params__contains={"labels": "Todo"},
        ).respond(json=[])

        issues = await gh_tracker.fetch_candidate_issues()
        identifiers = [iss.identifier for iss in issues]
        # Verify total count.
        assert len(identifiers) == 110
        # Verify ordering is preserved: first page items come before second page.
        assert identifiers == [str(i) for i in range(1, 111)]

    @pytest.mark.asyncio
    @respx.mock
    async def test_state_fetch_paginates_beyond_one_page(self, gh_tracker: GitHubTracker) -> None:
        """55 IDs are fetched individually, each producing one request."""
        ids = [str(i) for i in range(1, 56)]

        for i in range(1, 56):
            respx.get(f"https://api.github.com/repos/o/r/issues/{i}").respond(
                json=_gh_issue(i, labels=["open"]),
            )

        states = await gh_tracker.fetch_issue_states_by_ids(ids)
        assert len(states) == 55
        # Every ID should have a non-None state.
        for ident in ids:
            assert states[ident] is not None

    @pytest.mark.asyncio
    @respx.mock
    async def test_unassigned_issues_marked_not_routed(self, gh_tracker: GitHubTracker) -> None:
        """Issue with a different assignee should have assignee_id set, not None."""
        respx.get(
            "https://api.github.com/repos/o/r/issues",
            params__contains={"labels": "open"},
        ).respond(
            json=[
                _gh_issue(20, assignee="other-dev", labels=["open"]),
                _gh_issue(21, assignee=None, labels=["open"]),
            ]
        )

        respx.get(
            "https://api.github.com/repos/o/r/issues",
            params__contains={"labels": "Todo"},
        ).respond(json=[])

        issues = await gh_tracker.fetch_candidate_issues()
        by_id = {iss.identifier: iss for iss in issues}

        # Issue 20 has an assignee "other-dev".
        assert by_id["20"].assignee_id == "other-dev"
        # Issue 21 has no assignee.
        assert by_id["21"].assignee_id is None
