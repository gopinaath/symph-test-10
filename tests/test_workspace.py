"""Tests for symphony.workspace — covers all Elixir-equivalent test cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from symphony.workspace import Workspace, WorkspaceConfig, WorkspaceError

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ws(tmp_path: Path, **overrides) -> Workspace:
    """Build a Workspace with *tmp_path* as root plus any config overrides."""
    cfg = WorkspaceConfig(root=str(tmp_path), **overrides)
    return Workspace(cfg)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_via_after_create_hook(tmp_path: Path) -> None:
    """Workspace bootstrap via after_create hook (e.g. git clone)."""
    ws = _ws(tmp_path, after_create="touch $WORKSPACE/bootstrapped")
    result = await ws.create("issue-1")
    assert isinstance(result, Path)
    assert (result / "bootstrapped").exists()


@pytest.mark.asyncio
async def test_path_deterministic_sanitized(tmp_path: Path) -> None:
    """Workspace path is deterministic per identifier; / becomes _."""
    ws = _ws(tmp_path)
    result = await ws.create("org/repo#42")
    assert isinstance(result, Path)
    assert result.name == "org_repo_42"

    # Same identifier produces same path.
    result2 = await ws.create("org/repo#42")
    assert result == result2


@pytest.mark.asyncio
async def test_reuses_existing_directory(tmp_path: Path) -> None:
    """Workspace reuses existing directory without deleting local changes."""
    ws = _ws(tmp_path)
    result = await ws.create("issue-2")
    assert isinstance(result, Path)
    (result / "local-change.txt").write_text("precious")

    result2 = await ws.create("issue-2")
    assert isinstance(result2, Path)
    assert (result2 / "local-change.txt").read_text() == "precious"


@pytest.mark.asyncio
async def test_replaces_stale_non_directory(tmp_path: Path) -> None:
    """Workspace replaces stale non-directory paths (file -> dir)."""
    ws = _ws(tmp_path)
    stale_path = tmp_path / "stale-issue"
    stale_path.write_text("I am a file, not a dir")

    result = await ws.create("stale-issue")
    assert isinstance(result, Path)
    assert result.is_dir()


@pytest.mark.asyncio
async def test_rejects_symlink_escape(tmp_path: Path) -> None:
    """Workspace rejects symlink escapes under configured root."""
    ws = _ws(tmp_path)
    # Create a symlink inside root that points outside root.
    escape_target = tmp_path.parent / "escape_target"
    escape_target.mkdir(exist_ok=True)
    link_path = tmp_path / "evil-link"
    link_path.symlink_to(escape_target)

    result = await ws.create("evil-link")
    assert isinstance(result, WorkspaceError)
    assert "escapes root" in result.message


@pytest.mark.asyncio
async def test_canonicalizes_symlinked_root(tmp_path: Path) -> None:
    """Workspace canonicalizes symlinked workspace roots."""
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    link_root = tmp_path / "link-root"
    link_root.symlink_to(real_root)

    ws = _ws(link_root)
    result = await ws.create("issue-3")
    assert isinstance(result, Path)
    # The resolved path must be under the real root.
    assert str(result).startswith(str(real_root))


@pytest.mark.asyncio
async def test_remove_rejects_workspace_root(tmp_path: Path) -> None:
    """Workspace remove rejects the workspace root itself."""
    ws = _ws(tmp_path)
    # "." would resolve to the root itself.
    err = await ws.remove(".")
    assert isinstance(err, WorkspaceError)
    assert "root" in err.message.lower()


@pytest.mark.asyncio
async def test_surfaces_after_create_hook_failure(tmp_path: Path) -> None:
    """Workspace surfaces after_create hook failures (non-zero exit)."""
    ws = _ws(tmp_path, after_create="exit 1")
    result = await ws.create("fail-issue")
    assert isinstance(result, WorkspaceError)
    assert "exit" in result.message.lower() or "code" in result.message.lower()


@pytest.mark.asyncio
async def test_surfaces_after_create_hook_timeout(tmp_path: Path) -> None:
    """Workspace surfaces after_create hook timeouts."""
    ws = _ws(tmp_path, after_create="sleep 60", hook_timeout=0.5)
    result = await ws.create("timeout-issue")
    assert isinstance(result, WorkspaceError)
    assert "timed out" in result.message.lower()


@pytest.mark.asyncio
async def test_creates_empty_dir_when_no_hook(tmp_path: Path) -> None:
    """Workspace creates empty directory when no hook configured."""
    ws = _ws(tmp_path)
    result = await ws.create("empty-issue")
    assert isinstance(result, Path)
    assert result.is_dir()
    assert list(result.iterdir()) == []


@pytest.mark.asyncio
async def test_removes_workspace_for_closed_issue(tmp_path: Path) -> None:
    """Workspace removes all workspaces for a closed issue identifier."""
    ws = _ws(tmp_path)
    result = await ws.create("closed-42")
    assert isinstance(result, Path)
    assert result.exists()

    err = await ws.remove("closed-42")
    assert err is None
    assert not result.exists()


@pytest.mark.asyncio
async def test_cleanup_handles_missing_root(tmp_path: Path) -> None:
    """Workspace cleanup handles missing root gracefully."""
    missing = tmp_path / "does-not-exist"
    ws = _ws(missing)
    errors = await ws.cleanup(["a", "b"])
    assert errors == []


@pytest.mark.asyncio
async def test_cleanup_ignores_none_identifier(tmp_path: Path) -> None:
    """Workspace cleanup ignores non-binary (None) identifier."""
    ws = _ws(tmp_path)
    errors = await ws.cleanup([None, None])
    assert errors == []


@pytest.mark.asyncio
async def test_remove_returns_error_for_missing_directory(tmp_path: Path) -> None:
    """Workspace remove returns error info for missing directory."""
    ws = _ws(tmp_path)
    err = await ws.remove("ghost")
    assert isinstance(err, WorkspaceError)
    assert "does not exist" in err.message


@pytest.mark.asyncio
async def test_hooks_support_multiline_scripts(tmp_path: Path) -> None:
    """Workspace hooks support multiline scripts."""
    script = "touch $WORKSPACE/line1\ntouch $WORKSPACE/line2"
    ws = _ws(tmp_path, after_create=script)
    result = await ws.create("multi-hook")
    assert isinstance(result, Path)
    assert (result / "line1").exists()
    assert (result / "line2").exists()


@pytest.mark.asyncio
async def test_remove_continues_when_before_remove_hook_fails(
    tmp_path: Path,
) -> None:
    """Workspace remove continues when before_remove hook fails."""
    ws = _ws(tmp_path, before_remove="exit 1")
    result = await ws.create("rm-hook-fail")
    assert isinstance(result, Path)
    assert result.exists()

    err = await ws.remove("rm-hook-fail")
    # Removal should still succeed despite hook failure.
    assert err is None
    assert not result.exists()


@pytest.mark.asyncio
async def test_remove_continues_when_before_remove_hook_times_out(
    tmp_path: Path,
) -> None:
    """Workspace remove continues when before_remove hook times out."""
    ws = _ws(tmp_path, before_remove="sleep 60", hook_timeout=0.5)
    result = await ws.create("rm-hook-timeout")
    assert isinstance(result, Path)
    assert result.exists()

    err = await ws.remove("rm-hook-timeout")
    assert err is None
    assert not result.exists()
