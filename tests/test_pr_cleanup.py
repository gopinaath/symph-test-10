"""Tests for symphony.pr_cleanup — ports workspace_before_remove_test.exs."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from symphony.pr_cleanup import (
    cleanup_workspace_prs,
    close_prs_for_branch,
    get_current_branch,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_process(returncode: int = 0, stdout: str = "", stderr: str = "") -> AsyncMock:
    """Build a mock subprocess whose communicate() returns the given data."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(
        return_value=(stdout.encode(), stderr.encode())
    )
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


def _make_exec_side_effect(mapping: dict[tuple[str, ...], AsyncMock]):
    """Return an async callable that dispatches on the first N args.

    *mapping* keys are tuples of the leading positional args.
    """

    async def _side_effect(*args, **_kwargs):  # type: ignore[no-untyped-def]
        for key, proc in mapping.items():
            if args[: len(key)] == key:
                return proc
        raise AssertionError(f"Unexpected subprocess call: {args}")

    return _side_effect


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("symphony.pr_cleanup._is_on_path", return_value=False)
async def test_noop_when_git_unavailable(mock_path: MagicMock) -> None:
    """No-op when git is not on PATH (branch detection fails)."""
    result = await cleanup_workspace_prs()
    assert result == []


@pytest.mark.asyncio
@patch("symphony.pr_cleanup._is_on_path")
async def test_noop_when_gh_unavailable(mock_path: MagicMock) -> None:
    """No-op when gh is not on PATH."""

    def _which(name: str) -> bool:
        return name != "gh"

    mock_path.side_effect = _which

    # Provide an explicit branch so we skip git entirely.
    result = await close_prs_for_branch("feature/x")
    assert result == []


@pytest.mark.asyncio
@patch("symphony.pr_cleanup._is_on_path", return_value=True)
@patch("asyncio.create_subprocess_exec")
async def test_noop_when_gh_auth_fails(
    mock_exec: AsyncMock,
    mock_path: MagicMock,
) -> None:
    """No-op when gh auth status fails."""
    mock_exec.return_value = _make_process(returncode=1, stderr="not logged in")

    result = await close_prs_for_branch("feature/x")
    assert result == []


@pytest.mark.asyncio
@patch("symphony.pr_cleanup._is_on_path", return_value=True)
@patch("asyncio.create_subprocess_exec")
async def test_uses_current_branch_when_not_specified(
    mock_exec: AsyncMock,
    mock_path: MagicMock,
) -> None:
    """When branch is None, get_current_branch is used."""
    git_proc = _make_process(returncode=0, stdout="feature/auto-detected\n")
    auth_proc = _make_process(returncode=0)
    list_proc = _make_process(returncode=0, stdout="[]")

    mock_exec.side_effect = _make_exec_side_effect({
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): git_proc,
        ("gh", "auth", "status"): auth_proc,
        ("gh", "pr", "list"): list_proc,
    })

    result = await cleanup_workspace_prs()
    assert result == []

    # Verify git rev-parse was called
    calls = mock_exec.call_args_list
    git_calls = [c for c in calls if c[0][0] == "git"]
    assert len(git_calls) == 1
    assert "rev-parse" in git_calls[0][0]


@pytest.mark.asyncio
@patch("symphony.pr_cleanup._is_on_path", return_value=True)
@patch("asyncio.create_subprocess_exec")
async def test_closes_open_prs_for_branch(
    mock_exec: AsyncMock,
    mock_path: MagicMock,
) -> None:
    """Lists open PRs for the branch and closes each one."""
    import json

    prs = [
        {"number": 10, "url": "https://github.com/org/repo/pull/10"},
        {"number": 20, "url": "https://github.com/org/repo/pull/20"},
    ]
    auth_proc = _make_process(returncode=0)
    list_proc = _make_process(returncode=0, stdout=json.dumps(prs))
    close_proc_10 = _make_process(returncode=0)
    close_proc_20 = _make_process(returncode=0)

    mock_exec.side_effect = _make_exec_side_effect({
        ("gh", "auth", "status"): auth_proc,
        ("gh", "pr", "list"): list_proc,
        ("gh", "pr", "close", "10"): close_proc_10,
        ("gh", "pr", "close", "20"): close_proc_20,
    })

    results = await close_prs_for_branch("feature/x")
    assert len(results) == 2
    assert results[0]["number"] == 10
    assert results[0]["success"] is True
    assert results[1]["number"] == 20
    assert results[1]["success"] is True


@pytest.mark.asyncio
@patch("symphony.pr_cleanup._is_on_path", return_value=True)
@patch("asyncio.create_subprocess_exec")
async def test_tolerates_close_failures(
    mock_exec: AsyncMock,
    mock_path: MagicMock,
) -> None:
    """One PR close failure does not prevent closing the others."""
    import json

    prs = [
        {"number": 10, "url": "https://github.com/org/repo/pull/10"},
        {"number": 20, "url": "https://github.com/org/repo/pull/20"},
        {"number": 30, "url": "https://github.com/org/repo/pull/30"},
    ]
    auth_proc = _make_process(returncode=0)
    list_proc = _make_process(returncode=0, stdout=json.dumps(prs))
    close_ok = _make_process(returncode=0)
    close_fail = _make_process(returncode=1, stderr="permission denied")

    mock_exec.side_effect = _make_exec_side_effect({
        ("gh", "auth", "status"): auth_proc,
        ("gh", "pr", "list"): list_proc,
        ("gh", "pr", "close", "10"): close_ok,
        ("gh", "pr", "close", "20"): close_fail,
        ("gh", "pr", "close", "30"): close_ok,
    })

    results = await close_prs_for_branch("feature/x")
    assert len(results) == 3

    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert "error" in results[1]
    assert results[2]["success"] is True


@pytest.mark.asyncio
@patch("symphony.pr_cleanup._is_on_path", return_value=True)
@patch("asyncio.create_subprocess_exec")
async def test_noop_when_pr_list_fails(
    mock_exec: AsyncMock,
    mock_path: MagicMock,
) -> None:
    """No close attempts when pr list itself fails."""
    auth_proc = _make_process(returncode=0)
    list_proc = _make_process(returncode=1, stderr="API error")

    mock_exec.side_effect = _make_exec_side_effect({
        ("gh", "auth", "status"): auth_proc,
        ("gh", "pr", "list"): list_proc,
    })

    results = await close_prs_for_branch("feature/x")
    assert results == []

    # Verify no close calls were made
    calls = mock_exec.call_args_list
    close_calls = [c for c in calls if len(c[0]) >= 3 and c[0][:3] == ("gh", "pr", "close")]
    assert len(close_calls) == 0


@pytest.mark.asyncio
async def test_noop_when_branch_is_blank() -> None:
    """Empty string branch results in no-op."""
    result = await cleanup_workspace_prs(branch="")
    assert result == []

    result2 = await cleanup_workspace_prs(branch="   ")
    assert result2 == []


@pytest.mark.asyncio
@patch("symphony.pr_cleanup._is_on_path", return_value=True)
@patch("asyncio.create_subprocess_exec")
async def test_formats_close_failures_without_stderr(
    mock_exec: AsyncMock,
    mock_path: MagicMock,
) -> None:
    """Error message is formatted correctly even when stderr is empty."""
    import json

    prs = [{"number": 42, "url": "https://github.com/org/repo/pull/42"}]
    auth_proc = _make_process(returncode=0)
    list_proc = _make_process(returncode=0, stdout=json.dumps(prs))
    close_fail = _make_process(returncode=1, stderr="")

    mock_exec.side_effect = _make_exec_side_effect({
        ("gh", "auth", "status"): auth_proc,
        ("gh", "pr", "list"): list_proc,
        ("gh", "pr", "close", "42"): close_fail,
    })

    results = await close_prs_for_branch("feature/x")
    assert len(results) == 1
    assert results[0]["success"] is False
    error_msg = results[0]["error"]
    assert isinstance(error_msg, str)
    assert "42" in error_msg
    # Should not end with a dangling colon or extra whitespace
    assert not error_msg.endswith(":")
    assert not error_msg.endswith(": ")
