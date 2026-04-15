"""Extensible dynamic-tool registry for the Codex app-server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Type alias for tool handler functions.
ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class DynamicToolRegistry:
    """Registry of dynamic tools that can be dispatched during a Codex session."""

    _handlers: dict[str, ToolHandler] = field(default_factory=dict)
    _specs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- public API ----------------------------------------------------------

    def register(
        self,
        name: str,
        spec: dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        """Register a tool by *name* with its JSON-Schema *spec* and async *handler*."""
        self._specs[name] = spec
        self._handlers[name] = handler

    def tool_specs(self) -> list[dict[str, Any]]:
        """Return the list of tool definitions suitable for ``dynamicTools`` in the
        ``thread/start`` payload."""
        return [{"name": name, "inputSchema": spec} for name, spec in self._specs.items()]

    def register_run_tests(self, workspace_path: str, default_command: str | None = None) -> None:
        """Register the run_tests tool with workspace context."""
        handler = make_run_tests_handler(workspace_path, default_command)
        self.register("run_tests", RUN_TESTS_SPEC, handler)

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch *tool_name* with *arguments*.

        Returns ``{success: bool, output: str}``.
        If the tool is not registered, returns a failure listing supported tools.
        """
        handler = self._handlers.get(tool_name)
        if handler is None:
            supported = sorted(self._handlers)
            return {
                "success": False,
                "output": (f"Unsupported tool: {tool_name}. Supported tools: {', '.join(supported)}"),
            }
        try:
            return await handler(arguments)
        except Exception as exc:  # noqa: BLE001
            logger.exception("dynamic tool %s failed", tool_name)
            return {"success": False, "output": f"Tool call failed: {exc}"}


# ---------------------------------------------------------------------------
# Built-in tool: github_graphql
# ---------------------------------------------------------------------------

GITHUB_GRAPHQL_SPEC: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A GitHub GraphQL query or mutation.",
        },
        "variables": {
            "type": "object",
            "description": "Optional variables for the query.",
            "additionalProperties": True,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"


async def github_graphql_handler(
    arguments: dict[str, Any],
    *,
    token: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Execute a GitHub GraphQL query.

    Parameters
    ----------
    arguments:
        Must contain ``query`` (str) and optionally ``variables`` (dict).
    token:
        GitHub personal-access token.  Falls back to the ``GITHUB_TOKEN``
        environment variable when *None*.
    http_client:
        Optional pre-configured *httpx.AsyncClient* (useful for testing).
    """
    # --- validate required arguments ----------------------------------------
    if "query" not in arguments:
        return {
            "success": False,
            "output": "Missing required argument: query",
        }

    query = arguments["query"]

    if not isinstance(query, str):
        return {
            "success": False,
            "output": f"Invalid argument type for query: expected string, got {type(query).__name__}",
        }

    if not query.strip():
        return {
            "success": False,
            "output": "Blank query string is not allowed",
        }

    variables = arguments.get("variables")
    if variables is not None and not isinstance(variables, dict):
        return {
            "success": False,
            "output": f"Invalid argument type for variables: expected object, got {type(variables).__name__}",
        }

    # --- resolve token ------------------------------------------------------
    if token is None:
        import os

        token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return {"success": False, "output": "GITHUB_TOKEN is not set"}

    headers = {
        "Authorization": f"bearer {token}",
        "Content-Type": "application/json",
    }

    body: dict[str, Any] = {"query": query}
    if variables is not None:
        body["variables"] = variables

    # --- execute request ----------------------------------------------------
    own_client = http_client is None
    client = http_client or httpx.AsyncClient()
    try:
        try:
            resp = await client.post(
                GITHUB_GRAPHQL_URL,
                headers=headers,
                json=body,
            )
        except httpx.HTTPError as exc:
            return {"success": False, "output": f"Transport failure: {exc}"}

        data = resp.json()

        if "errors" in data:
            return {"success": False, "output": json.dumps(data["errors"])}

        return {"success": True, "output": json.dumps(data.get("data", data))}
    finally:
        if own_client:
            await client.aclose()


# ---------------------------------------------------------------------------
# Built-in tool: run_tests
# ---------------------------------------------------------------------------

RUN_TESTS_SPEC: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Test command to run. If omitted, uses the configured default from WORKFLOW.md validation.command.",
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Timeout in seconds (default: 120)",
        },
    },
    "additionalProperties": False,
}

_DEFAULT_TEST_TIMEOUT = 120


def make_run_tests_handler(workspace_path: str, default_command: str | None = None) -> ToolHandler:
    """Create a run_tests handler bound to a specific workspace."""

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        command = arguments.get("command") or default_command
        if not command:
            return {
                "success": False,
                "output": "No test command specified and no default command configured.",
                "exit_code": -1,
                "duration_ms": 0,
            }

        timeout = arguments.get("timeout_seconds", _DEFAULT_TEST_TIMEOUT)

        ws = Path(workspace_path)
        if not ws.exists():
            return {
                "success": False,
                "output": f"Workspace path does not exist: {workspace_path}",
                "exit_code": -1,
                "duration_ms": 0,
            }

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
                elapsed = (time.monotonic() - start) * 1000
                return {
                    "success": False,
                    "output": "Test execution timed out",
                    "exit_code": -1,
                    "duration_ms": elapsed,
                }

            stdout_text = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
            stderr_text = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
            output = stdout_text + stderr_text
            exit_code = proc.returncode or 0

            elapsed = (time.monotonic() - start) * 1000
            return {
                "success": exit_code == 0,
                "output": output,
                "exit_code": exit_code,
                "duration_ms": elapsed,
            }
        except OSError as exc:
            elapsed = (time.monotonic() - start) * 1000
            return {
                "success": False,
                "output": str(exc),
                "exit_code": -1,
                "duration_ms": elapsed,
            }

    return handler


def default_registry(
    *,
    github_token: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> DynamicToolRegistry:
    """Return a :class:`DynamicToolRegistry` pre-loaded with the built-in tools."""
    registry = DynamicToolRegistry()

    async def _gh_handler(args: dict[str, Any]) -> dict[str, Any]:
        return await github_graphql_handler(args, token=github_token, http_client=http_client)

    registry.register("github_graphql", GITHUB_GRAPHQL_SPEC, _gh_handler)
    return registry
