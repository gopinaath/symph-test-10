"""Tests for symphony.codex.dynamic_tool."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from symphony.codex.dynamic_tool import (
    GITHUB_GRAPHQL_URL,
    default_registry,
    github_graphql_handler,
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
