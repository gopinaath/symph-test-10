"""Tests for symphony.codex.dynamic_tool."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from symphony.codex.dynamic_tool import (
    GITHUB_GRAPHQL_URL,
    RUN_TESTS_SPEC,
    default_registry,
    github_graphql_handler,
    make_run_tests_handler,
)

# ---------------------------------------------------------------------------
# tool_specs
# ---------------------------------------------------------------------------


class TestToolSpecs:
    def test_tool_specs_advertises_input_contract(self):
        registry = default_registry(github_token="fake-token")
        specs = registry.tool_specs()
        assert len(specs) == 1
        spec = specs[0]
        assert spec["name"] == "github_graphql"
        schema = spec["inputSchema"]
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert "query" in schema["required"]


# ---------------------------------------------------------------------------
# unsupported tools
# ---------------------------------------------------------------------------


class TestUnsupportedTools:
    @pytest.mark.asyncio
    async def test_unsupported_tools_return_failure_with_supported_tool_list(self):
        registry = default_registry(github_token="fake")
        result = await registry.execute("nonexistent_tool", {})
        assert result["success"] is False
        assert "Unsupported tool: nonexistent_tool" in result["output"]
        assert "github_graphql" in result["output"]


# ---------------------------------------------------------------------------
# github_graphql
# ---------------------------------------------------------------------------


class TestGithubGraphql:
    @pytest.mark.asyncio
    @respx.mock
    async def test_github_graphql_returns_successful_responses(self):
        respx.post(GITHUB_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={"data": {"viewer": {"login": "octocat"}}},
            )
        )
        result = await github_graphql_handler(
            {"query": "{ viewer { login } }", "variables": {}},
            token="fake-token",
        )
        assert result["success"] is True
        data = json.loads(result["output"])
        assert data["viewer"]["login"] == "octocat"

    @pytest.mark.asyncio
    @respx.mock
    async def test_github_graphql_accepts_raw_query_string(self):
        respx.post(GITHUB_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={"data": {"repository": {"name": "test"}}},
            )
        )
        result = await github_graphql_handler(
            {"query": '{ repository(owner:"o", name:"r") { name } }'},
            token="fake-token",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_github_graphql_rejects_blank_query_strings(self):
        result = await github_graphql_handler({"query": "   "}, token="fake-token")
        assert result["success"] is False
        assert "Blank query" in result["output"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_github_graphql_marks_error_responses_as_failures(self):
        respx.post(GITHUB_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json={"errors": [{"message": "Field not found"}]},
            )
        )
        result = await github_graphql_handler({"query": "{ bad }"}, token="fake-token")
        assert result["success"] is False
        assert "Field not found" in result["output"]

    @pytest.mark.asyncio
    async def test_github_graphql_validates_required_arguments(self):
        result = await github_graphql_handler({}, token="fake-token")
        assert result["success"] is False
        assert "Missing required argument: query" in result["output"]

    @pytest.mark.asyncio
    async def test_github_graphql_rejects_invalid_argument_types(self):
        result = await github_graphql_handler(
            {"query": 42},
            token="fake-token",  # type: ignore[arg-type]
        )
        assert result["success"] is False
        assert "Invalid argument type for query" in result["output"]

        result2 = await github_graphql_handler(
            {"query": "{ ok }", "variables": "not-a-dict"},
            token="fake-token",
        )
        assert result2["success"] is False
        assert "Invalid argument type for variables" in result2["output"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_github_graphql_formats_transport_failures(self):
        respx.post(GITHUB_GRAPHQL_URL).mock(side_effect=httpx.ConnectError("Connection refused"))
        result = await github_graphql_handler({"query": "{ viewer { login } }"}, token="fake-token")
        assert result["success"] is False
        assert "Transport failure" in result["output"]


# ---------------------------------------------------------------------------
# run_tests
# ---------------------------------------------------------------------------


class TestRunTests:
    def test_run_tests_spec_advertised(self):
        """tool_specs includes run_tests with correct schema after registration."""
        registry = default_registry(github_token="fake-token")
        registry.register_run_tests("/tmp/fake-workspace", default_command="echo ok")
        specs = registry.tool_specs()
        names = [s["name"] for s in specs]
        assert "run_tests" in names
        rt_spec = next(s for s in specs if s["name"] == "run_tests")
        schema = rt_spec["inputSchema"]
        assert "command" in schema["properties"]
        assert schema["properties"]["command"]["type"] == "string"
        assert "timeout_seconds" in schema["properties"]
        assert schema["properties"]["timeout_seconds"]["type"] == "integer"
        assert schema["additionalProperties"] is False

    @pytest.mark.asyncio
    async def test_run_tests_executes_command_and_returns_output(self, tmp_path):
        """Mock subprocess, verify structured result."""
        handler = make_run_tests_handler(str(tmp_path), default_command="echo hello")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"all tests passed\n", b""))
        mock_proc.returncode = 0
        mock_proc.kill = AsyncMock()
        mock_proc.wait = AsyncMock()

        with patch("symphony.codex.dynamic_tool.asyncio.create_subprocess_shell", return_value=mock_proc) as mock_create:
            result = await handler({"command": "pytest -v"})

        mock_create.assert_called_once()
        assert result["success"] is True
        assert "all tests passed" in result["output"]
        assert result["exit_code"] == 0
        assert isinstance(result["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_run_tests_uses_default_command_when_not_specified(self, tmp_path):
        """Verify fallback to configured default."""
        handler = make_run_tests_handler(str(tmp_path), default_command="make test")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
        mock_proc.returncode = 0
        mock_proc.kill = AsyncMock()
        mock_proc.wait = AsyncMock()

        with patch("symphony.codex.dynamic_tool.asyncio.create_subprocess_shell", return_value=mock_proc) as mock_create:
            result = await handler({})

        # Verify it used the default command
        call_args = mock_create.call_args
        assert call_args[0][0] == "make test"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_tests_handles_timeout(self, tmp_path):
        """Mock slow command, verify timeout result."""
        handler = make_run_tests_handler(str(tmp_path), default_command="sleep 999")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = AsyncMock()
        mock_proc.wait = AsyncMock()

        with patch("symphony.codex.dynamic_tool.asyncio.create_subprocess_shell", return_value=mock_proc):
            # We also need to patch wait_for so it raises TimeoutError
            with patch("symphony.codex.dynamic_tool.asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                result = await handler({"timeout_seconds": 1})

        assert result["success"] is False
        assert result["output"] == "Test execution timed out"
        assert result["exit_code"] == -1
        assert isinstance(result["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_run_tests_handles_missing_workspace(self):
        """Verify error when workspace path doesn't exist."""
        handler = make_run_tests_handler("/nonexistent/path/that/does/not/exist", default_command="echo hi")
        result = await handler({})
        assert result["success"] is False
        assert "does not exist" in result["output"]
        assert result["exit_code"] == -1

    @pytest.mark.asyncio
    async def test_run_tests_returns_exit_code(self, tmp_path):
        """Verify non-zero exit code captured."""
        handler = make_run_tests_handler(str(tmp_path), default_command="false")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"FAIL: test_something\n"))
        mock_proc.returncode = 1
        mock_proc.kill = AsyncMock()
        mock_proc.wait = AsyncMock()

        with patch("symphony.codex.dynamic_tool.asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await handler({})

        assert result["success"] is False
        assert result["exit_code"] == 1
        assert "FAIL: test_something" in result["output"]

    @pytest.mark.asyncio
    async def test_run_tests_no_command_configured(self, tmp_path):
        """Error when no command is specified and no default is configured."""
        handler = make_run_tests_handler(str(tmp_path))
        result = await handler({})
        assert result["success"] is False
        assert "No test command" in result["output"]
