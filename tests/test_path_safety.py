"""Tests for symphony.path_safety."""

from __future__ import annotations

from pathlib import Path

from symphony.path_safety import SafeResolveError, safe_resolve


class TestSafeResolve:
    def test_canonical_path_within_root(self, tmp_path: Path) -> None:
        """safe_resolve returns canonical path within root."""
        (tmp_path / "subdir").mkdir()
        result = safe_resolve("subdir", tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path / "subdir"
        assert result.is_absolute()

    def test_nested_path(self, tmp_path: Path) -> None:
        """safe_resolve handles nested relative paths."""
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b").mkdir()
        result = safe_resolve("a/b", tmp_path)
        assert isinstance(result, Path)
        assert result == tmp_path / "a" / "b"

    def test_rejects_path_outside_root(self, tmp_path: Path) -> None:
        """safe_resolve rejects paths that escape the root via '..'."""
        result = safe_resolve("../escape", tmp_path)
        assert isinstance(result, SafeResolveError)
        assert "escapes root" in result.message

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        """safe_resolve rejects a symlink pointing outside root."""
        outside = tmp_path.parent / "outside_target"
        outside.mkdir(exist_ok=True)
        link = tmp_path / "sneaky"
        link.symlink_to(outside)

        result = safe_resolve("sneaky", tmp_path)
        assert isinstance(result, SafeResolveError)
        assert "escapes root" in result.message

    def test_rejects_long_segment(self, tmp_path: Path) -> None:
        """safe_resolve returns error for path segments > 255 chars."""
        long_name = "x" * 256
        result = safe_resolve(long_name, tmp_path)
        assert isinstance(result, SafeResolveError)
        assert "255" in result.message

    def test_absolute_path_within_root(self, tmp_path: Path) -> None:
        """safe_resolve handles absolute paths that are under root."""
        child = tmp_path / "inside"
        child.mkdir()
        result = safe_resolve(str(child), tmp_path)
        assert isinstance(result, Path)
        assert result == child

    def test_absolute_path_outside_root(self, tmp_path: Path) -> None:
        """safe_resolve rejects absolute paths outside root."""
        result = safe_resolve("/etc/passwd", tmp_path)
        assert isinstance(result, SafeResolveError)
        assert "escapes root" in result.message

    def test_segment_exactly_255_is_ok(self, tmp_path: Path) -> None:
        """A segment of exactly 255 chars is allowed."""
        name = "a" * 255
        # Don't actually create it (filesystem may not allow), just check length validation passes.
        result = safe_resolve(name, tmp_path)
        # Might be a Path (non-existing) or an OS error, but NOT a segment-length error.
        if isinstance(result, SafeResolveError):
            assert "255" not in result.message
