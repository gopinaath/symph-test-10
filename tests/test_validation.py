"""Tests for workspace validation (run_validation + ValidationResult)."""

from __future__ import annotations

import asyncio

import pytest

from symphony.config import ValidationConfig
from symphony.workspace import ValidationResult, Workspace, WorkspaceConfig


def _make_workspace(tmp_path) -> Workspace:
    root = tmp_path / "workspaces"
    root.mkdir()
    return Workspace(WorkspaceConfig(root=str(root)))


@pytest.fixture()
def ws(tmp_path):
    workspace = _make_workspace(tmp_path)
    # Pre-create the workspace directory for the test identifier
    ws_dir = workspace.path_for("test-issue")
    ws_dir.mkdir(parents=True, exist_ok=True)
    return workspace


# ---------------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------------


class TestValidationBasic:
    def test_validation_passes_when_no_command_and_no_assertions(self, ws) -> None:
        config = ValidationConfig(enabled=True)
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.exit_code is None
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.assertion_results == []

    def test_validation_runs_command_and_captures_output(self, ws) -> None:
        config = ValidationConfig(
            enabled=True,
            command="echo hello-world",
        )
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert result.passed is True
        assert result.exit_code == 0
        assert "hello-world" in result.stdout

    def test_validation_fails_on_nonzero_exit_code(self, ws) -> None:
        config = ValidationConfig(
            enabled=True,
            command="exit 1",
        )
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert result.passed is False
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# File exists assertions
# ---------------------------------------------------------------------------


class TestFileExistsAssertion:
    def test_validation_file_exists_assertion_passes(self, ws) -> None:
        # Create the file in the workspace
        ws_dir = ws.path_for("test-issue")
        (ws_dir / "readme.txt").write_text("hello")

        config = ValidationConfig(
            enabled=True,
            assertions=[{"kind": "file_exists", "path": "readme.txt"}],
        )
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert result.passed is True
        assert len(result.assertion_results) == 1
        assert result.assertion_results[0]["passed"] is True

    def test_validation_file_exists_assertion_fails(self, ws) -> None:
        config = ValidationConfig(
            enabled=True,
            assertions=[{"kind": "file_exists", "path": "nonexistent.txt"}],
        )
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert result.passed is False
        assert len(result.assertion_results) == 1
        assert result.assertion_results[0]["passed"] is False


# ---------------------------------------------------------------------------
# Command exit code assertions
# ---------------------------------------------------------------------------


class TestCommandExitCodeAssertion:
    def test_validation_command_exit_code_assertion_passes(self, ws) -> None:
        config = ValidationConfig(
            enabled=True,
            assertions=[
                {"kind": "command_exit_code", "command": "true", "expected": 0},
            ],
        )
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert result.passed is True
        assert len(result.assertion_results) == 1
        assert result.assertion_results[0]["passed"] is True

    def test_validation_command_exit_code_assertion_fails(self, ws) -> None:
        config = ValidationConfig(
            enabled=True,
            assertions=[
                {"kind": "command_exit_code", "command": "false", "expected": 0},
            ],
        )
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert result.passed is False
        assert len(result.assertion_results) == 1
        assert result.assertion_results[0]["passed"] is False
        assert result.assertion_results[0]["actual"] == 1


# ---------------------------------------------------------------------------
# File contains assertions
# ---------------------------------------------------------------------------


class TestFileContainsAssertion:
    def test_validation_file_contains_assertion_passes(self, ws) -> None:
        ws_dir = ws.path_for("test-issue")
        (ws_dir / "main.py").write_text("def main():\n    pass\n")

        config = ValidationConfig(
            enabled=True,
            assertions=[
                {"kind": "file_contains", "path": "main.py", "pattern": "def main"},
            ],
        )
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert result.passed is True
        assert len(result.assertion_results) == 1
        assert result.assertion_results[0]["passed"] is True

    def test_validation_file_contains_assertion_fails(self, ws) -> None:
        ws_dir = ws.path_for("test-issue")
        (ws_dir / "main.py").write_text("print('hello')\n")

        config = ValidationConfig(
            enabled=True,
            assertions=[
                {"kind": "file_contains", "path": "main.py", "pattern": "def main"},
            ],
        )
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert result.passed is False
        assert len(result.assertion_results) == 1
        assert result.assertion_results[0]["passed"] is False


# ---------------------------------------------------------------------------
# Assertion failure skips command
# ---------------------------------------------------------------------------


class TestAssertionFailureSkipsCommand:
    def test_validation_assertion_failure_skips_command(self, ws) -> None:
        # Create a marker file that the command would create if it ran
        ws_dir = ws.path_for("test-issue")

        config = ValidationConfig(
            enabled=True,
            command="touch marker_file",
            assertions=[
                {"kind": "file_exists", "path": "nonexistent.txt"},
            ],
        )
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert result.passed is False
        # The command should not have run, so marker_file should not exist
        assert not (ws_dir / "marker_file").exists()


# ---------------------------------------------------------------------------
# Command timeout
# ---------------------------------------------------------------------------


class TestValidationTimeout:
    def test_validation_command_timeout(self, ws) -> None:
        config = ValidationConfig(
            enabled=True,
            command="sleep 60",
            timeout_ms=500,  # 0.5 second timeout
        )
        result = asyncio.get_event_loop().run_until_complete(
            ws.run_validation("test-issue", config)
        )
        assert result.passed is False
        assert result.exit_code == -1
        assert "timed out" in result.stderr
