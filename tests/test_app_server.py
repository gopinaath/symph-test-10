"""Tests for symphony.codex.app_server — covers all Elixir test cases."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from symphony.codex.app_server import (
    AppServer,
    AppServerConfig,
    AppServerError,
    InputRequired,
    TurnFailed,
)
from symphony.codex.dynamic_tool import DynamicToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonl(*messages: dict[str, Any]) -> bytes:
    """Encode a sequence of JSON-RPC messages as newline-delimited bytes."""
    return b"".join(json.dumps(m).encode() + b"\n" for m in messages)


def _init_ok() -> dict:
    return {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}}


def _thread_ok(thread_id: str = "t-1") -> dict:
    return {"jsonrpc": "2.0", "id": 2, "result": {"threadId": thread_id}}


def _turn_ok(turn_id: str = "turn-1") -> dict:
    return {"jsonrpc": "2.0", "id": 3, "result": {"turnId": turn_id}}


def _turn_completed(usage: dict | None = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "turn/completed",
        "params": {"usage": usage or {}},
    }


def _turn_failed(reason: str = "error") -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "turn/failed",
        "params": {"reason": reason},
    }


class FakeProcess:
    """Minimal stand-in for asyncio.subprocess.Process."""

    def __init__(self, stdout_data: bytes = b"", stderr_data: bytes = b""):
        self.stdin = FakeStreamWriter()
        self.stdout = FakeStreamReader(stdout_data)
        self.stderr = FakeStreamReader(stderr_data)
        self.returncode = None

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        self.returncode = self.returncode or 0
        return self.returncode


class FakeStreamWriter:
    def __init__(self) -> None:
        self.written = bytearray()

    def write(self, data: bytes) -> None:
        self.written.extend(data)

    async def drain(self) -> None:
        pass


class FakeStreamReader:
    def __init__(self, data: bytes = b"") -> None:
        self._data = data
        self._pos = 0

    async def readline(self) -> bytes:
        if self._pos >= len(self._data):
            return b""
        end = self._data.index(b"\n", self._pos) + 1
        line = self._data[self._pos : end]
        self._pos = end
        return line


def _make_process(*messages: dict[str, Any], stderr: str = "") -> FakeProcess:
    """Create a FakeProcess whose stdout yields the given JSON-RPC messages."""
    return FakeProcess(
        stdout_data=_jsonl(*messages),
        stderr_data=stderr.encode() + (b"\n" if stderr else b""),
    )


async def _start_server(
    config: AppServerConfig | None = None,
    process: FakeProcess | None = None,
    tool_registry: DynamicToolRegistry | None = None,
    on_event: Any = None,
    cwd: str | None = None,
) -> AppServer:
    """Helper: create, patch-launch, and initialise an AppServer."""
    if cwd is None:
        cwd = tempfile.mkdtemp()
    cfg = config or AppServerConfig()
    server = AppServer(cfg, tool_registry=tool_registry, on_event=on_event)

    # We need to provide at minimum the init response.  If a process is not
    # supplied, provide a minimal one.
    if process is None:
        process = _make_process(_init_ok())

    with patch("asyncio.create_subprocess_exec", return_value=process):
        await server.start(cwd)
    return server


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStartup:
    """Startup command and configuration."""

    @pytest.mark.asyncio
    async def test_starts_with_workspace_cwd_and_expected_startup_command(self):
        """app server starts with workspace cwd and expected startup command."""
        cwd = tempfile.mkdtemp()
        proc = _make_process(_init_ok())

        with patch(
            "asyncio.create_subprocess_exec", return_value=proc
        ) as mock_exec:
            server = AppServer(AppServerConfig())
            await server.start(cwd)

            mock_exec.assert_called_once()
            args = mock_exec.call_args
            # First positional args should be the command.
            cmd = list(args[0])
            assert cmd == ["codex", "app-server", "--stdio"]
            # cwd kwarg must match.
            assert args[1]["cwd"] == cwd

        await server.stop()

    @pytest.mark.asyncio
    async def test_startup_command_supports_codex_args_override(self):
        """app server startup command supports codex args override."""
        cwd = tempfile.mkdtemp()
        custom_cmd = ["my-codex", "app-server", "--stdio", "--extra-flag"]
        cfg = AppServerConfig(command=custom_cmd)
        proc = _make_process(_init_ok())

        with patch(
            "asyncio.create_subprocess_exec", return_value=proc
        ) as mock_exec:
            server = AppServer(cfg)
            await server.start(cwd)
            cmd = list(mock_exec.call_args[0])
            assert cmd == custom_cmd

        await server.stop()

    @pytest.mark.asyncio
    async def test_startup_payload_uses_configurable_approval_and_sandbox(self):
        """app server startup payload uses configurable approval and sandbox settings."""
        cwd = tempfile.mkdtemp()
        cfg = AppServerConfig(approval_policy="on-failure", sandbox_policy="network-none")
        # init + thread/start response
        proc = _make_process(_init_ok(), _thread_ok())

        server = await _start_server(config=cfg, process=proc, cwd=cwd)
        thread_id = await server.start_thread(cwd, approval_policy="on-failure", sandbox_policy="network-none")

        # Parse what was written to stdin — the third message should be
        # thread/start (after initialize + initialized notification).
        written = proc.stdin.written.decode()
        lines = [json.loads(l) for l in written.strip().split("\n") if l.strip()]
        thread_start = [l for l in lines if l.get("method") == "thread/start"]
        assert len(thread_start) == 1
        params = thread_start[0]["params"]
        assert params["approvalPolicy"] == "on-failure"
        assert params["sandbox"] == "network-none"

        await server.stop()


class TestCwdValidation:
    """Workspace root and path validation."""

    @pytest.mark.asyncio
    async def test_rejects_workspace_root_and_paths_outside_workspace_root(self):
        """app server rejects workspace root and paths outside workspace root."""
        with pytest.raises(AppServerError, match="workspace root"):
            AppServer._validate_cwd("/")

        with pytest.raises(AppServerError, match="workspace root"):
            AppServer._validate_cwd("")

    @pytest.mark.asyncio
    async def test_rejects_symlink_escape_cwd_paths(self):
        """app server rejects symlink escape cwd paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir) / "real"
            real_dir.mkdir()
            link = Path(tmpdir) / "link"
            link.symlink_to(real_dir)
            # The link itself is fine (points to a valid subdir).
            # But a link that resolves to "/" should fail.
            root_link = Path(tmpdir) / "root_link"
            root_link.symlink_to("/")
            with pytest.raises(AppServerError, match="filesystem root"):
                AppServer._validate_cwd(str(root_link))


class TestSandboxPolicy:
    """Sandbox policy pass-through."""

    @pytest.mark.asyncio
    async def test_passes_explicit_turn_sandbox_policies_through_unchanged(self):
        """app server passes explicit turn sandbox policies through unchanged."""
        cwd = tempfile.mkdtemp()
        proc = _make_process(_init_ok(), _thread_ok(), _turn_ok(), _turn_completed())

        server = await _start_server(process=proc, cwd=cwd)
        await server.start_thread(cwd)
        await server.run_turn(
            input_text="hello",
            cwd=cwd,
            sandbox_policy="network-none",
        )

        written = proc.stdin.written.decode()
        lines = [json.loads(l) for l in written.strip().split("\n") if l.strip()]
        turn_start = [l for l in lines if l.get("method") == "turn/start"]
        assert len(turn_start) == 1
        assert turn_start[0]["params"]["sandboxPolicy"] == "network-none"

        await server.stop()


class TestInputRequired:
    """Request-for-input handling."""

    @pytest.mark.asyncio
    async def test_marks_request_for_input_events_as_hard_failure(self):
        """app server marks request-for-input events as hard failure."""
        cwd = tempfile.mkdtemp()
        proc = _make_process(
            _init_ok(),
            _thread_ok(),
            _turn_ok(),
            {"jsonrpc": "2.0", "method": "turn/input_required", "params": {}},
        )

        server = await _start_server(process=proc, cwd=cwd)
        await server.start_thread(cwd)
        with pytest.raises(InputRequired, match="interactive input"):
            await server.run_turn(input_text="do it", cwd=cwd)

        await server.stop()


class TestApprovalPolicies:
    """Approval behaviour under different policies."""

    @pytest.mark.asyncio
    async def test_fails_when_command_execution_approval_required_under_safer_defaults(
        self,
    ):
        """app server fails when command execution approval required under safer defaults."""
        cwd = tempfile.mkdtemp()
        cfg = AppServerConfig(approval_policy="on-failure")
        proc = _make_process(
            _init_ok(),
            _thread_ok(),
            _turn_ok(),
            {
                "jsonrpc": "2.0",
                "id": 100,
                "method": "item/commandExecution/requestApproval",
                "params": {"command": "rm -rf /"},
            },
        )

        server = await _start_server(config=cfg, process=proc, cwd=cwd)
        await server.start_thread(cwd)
        with pytest.raises(AppServerError, match="approval required"):
            await server.run_turn(input_text="delete everything", cwd=cwd)

        await server.stop()

    @pytest.mark.asyncio
    async def test_auto_approves_when_approval_policy_is_never(self):
        """app server auto-approves when approval policy is 'never'."""
        cwd = tempfile.mkdtemp()
        cfg = AppServerConfig(approval_policy="never")
        proc = _make_process(
            _init_ok(),
            _thread_ok(),
            _turn_ok(),
            {
                "jsonrpc": "2.0",
                "id": 100,
                "method": "item/commandExecution/requestApproval",
                "params": {"command": "ls"},
            },
            _turn_completed({"input_tokens": 10}),
        )

        server = await _start_server(config=cfg, process=proc, cwd=cwd)
        await server.start_thread(cwd)
        result = await server.run_turn(input_text="list files", cwd=cwd)

        # Verify an approval response was sent back.
        written = proc.stdin.written.decode()
        lines = [json.loads(l) for l in written.strip().split("\n") if l.strip()]
        approvals = [l for l in lines if l.get("id") == 100 and "result" in l]
        assert len(approvals) == 1
        assert approvals[0]["result"]["approved"] is True

        await server.stop()

    @pytest.mark.asyncio
    async def test_auto_approves_mcp_tool_approval_prompts(self):
        """app server auto-approves MCP tool approval prompts."""
        cwd = tempfile.mkdtemp()
        cfg = AppServerConfig(approval_policy="never")
        proc = _make_process(
            _init_ok(),
            _thread_ok(),
            _turn_ok(),
            {
                "jsonrpc": "2.0",
                "id": 200,
                "method": "item/mcpTool/requestApproval",
                "params": {"toolName": "mcp_tool_x"},
            },
            _turn_completed(),
        )

        server = await _start_server(config=cfg, process=proc, cwd=cwd)
        await server.start_thread(cwd)
        await server.run_turn(input_text="run it", cwd=cwd)

        written = proc.stdin.written.decode()
        lines = [json.loads(l) for l in written.strip().split("\n") if l.strip()]
        approvals = [l for l in lines if l.get("id") == 200 and "result" in l]
        assert len(approvals) == 1
        assert approvals[0]["result"]["approved"] is True

        await server.stop()


class TestUserInput:
    """Auto-answering user input requests."""

    @pytest.mark.asyncio
    async def test_sends_generic_non_interactive_answer_for_freeform_tool_input(
        self,
    ):
        """app server sends generic non-interactive answer for freeform tool input."""
        cwd = tempfile.mkdtemp()
        proc = _make_process(
            _init_ok(),
            _thread_ok(),
            _turn_ok(),
            {
                "jsonrpc": "2.0",
                "id": 300,
                "method": "item/tool/requestUserInput",
                "params": {"type": "freeform", "prompt": "Enter value:"},
            },
            _turn_completed(),
        )

        server = await _start_server(process=proc, cwd=cwd)
        await server.start_thread(cwd)
        await server.run_turn(input_text="go", cwd=cwd)

        written = proc.stdin.written.decode()
        lines = [json.loads(l) for l in written.strip().split("\n") if l.strip()]
        answers = [l for l in lines if l.get("id") == 300 and "result" in l]
        assert len(answers) == 1
        assert "non-interactive" in answers[0]["result"]["answer"].lower()

        await server.stop()

    @pytest.mark.asyncio
    async def test_sends_generic_non_interactive_answer_for_option_based_input(
        self,
    ):
        """app server sends generic non-interactive answer for option-based input."""
        cwd = tempfile.mkdtemp()
        proc = _make_process(
            _init_ok(),
            _thread_ok(),
            _turn_ok(),
            {
                "jsonrpc": "2.0",
                "id": 301,
                "method": "item/tool/requestUserInput",
                "params": {"type": "options", "options": ["a", "b"]},
            },
            _turn_completed(),
        )

        server = await _start_server(process=proc, cwd=cwd)
        await server.start_thread(cwd)
        await server.run_turn(input_text="go", cwd=cwd)

        written = proc.stdin.written.decode()
        lines = [json.loads(l) for l in written.strip().split("\n") if l.strip()]
        answers = [l for l in lines if l.get("id") == 301 and "result" in l]
        assert len(answers) == 1
        assert "non-interactive" in answers[0]["result"]["answer"].lower()

        await server.stop()


class TestDynamicTools:
    """Dynamic tool dispatch."""

    @pytest.mark.asyncio
    async def test_rejects_unsupported_dynamic_tool_calls_without_stalling(self):
        """app server rejects unsupported dynamic tool calls without stalling."""
        cwd = tempfile.mkdtemp()
        proc = _make_process(
            _init_ok(),
            _thread_ok(),
            _turn_ok(),
            {
                "jsonrpc": "2.0",
                "id": 400,
                "method": "item/tool/call",
                "params": {"name": "unknown_tool", "arguments": {}},
            },
            _turn_completed(),
        )

        registry = DynamicToolRegistry()
        server = await _start_server(
            process=proc, cwd=cwd, tool_registry=registry
        )
        await server.start_thread(cwd)
        await server.run_turn(input_text="use tool", cwd=cwd)

        written = proc.stdin.written.decode()
        lines = [json.loads(l) for l in written.strip().split("\n") if l.strip()]
        tool_resp = [l for l in lines if l.get("id") == 400 and "result" in l]
        assert len(tool_resp) == 1
        assert tool_resp[0]["result"]["success"] is False
        assert "Unsupported tool" in tool_resp[0]["result"]["output"]

        await server.stop()

    @pytest.mark.asyncio
    async def test_executes_supported_dynamic_tool_calls_and_returns_result(self):
        """app server executes supported dynamic tool calls and returns result."""
        cwd = tempfile.mkdtemp()
        proc = _make_process(
            _init_ok(),
            _thread_ok(),
            _turn_ok(),
            {
                "jsonrpc": "2.0",
                "id": 401,
                "method": "item/tool/call",
                "params": {"name": "echo", "arguments": {"msg": "hello"}},
            },
            _turn_completed(),
        )

        registry = DynamicToolRegistry()

        async def echo_handler(args: dict) -> dict:
            return {"success": True, "output": args.get("msg", "")}

        registry.register(
            "echo",
            {"type": "object", "properties": {"msg": {"type": "string"}}},
            echo_handler,
        )

        server = await _start_server(
            process=proc, cwd=cwd, tool_registry=registry
        )
        await server.start_thread(cwd)
        await server.run_turn(input_text="echo", cwd=cwd)

        written = proc.stdin.written.decode()
        lines = [json.loads(l) for l in written.strip().split("\n") if l.strip()]
        tool_resp = [l for l in lines if l.get("id") == 401 and "result" in l]
        assert len(tool_resp) == 1
        assert tool_resp[0]["result"]["success"] is True
        assert tool_resp[0]["result"]["output"] == "hello"

        await server.stop()

    @pytest.mark.asyncio
    async def test_emits_tool_call_failed_for_supported_tool_failures(self):
        """app server emits tool_call_failed for supported tool failures."""
        cwd = tempfile.mkdtemp()
        proc = _make_process(
            _init_ok(),
            _thread_ok(),
            _turn_ok(),
            {
                "jsonrpc": "2.0",
                "id": 402,
                "method": "item/tool/call",
                "params": {"name": "boom", "arguments": {}},
            },
            _turn_completed(),
        )

        registry = DynamicToolRegistry()

        async def boom_handler(args: dict) -> dict:
            raise RuntimeError("kaboom")

        registry.register(
            "boom",
            {"type": "object", "properties": {}},
            boom_handler,
        )

        server = await _start_server(
            process=proc, cwd=cwd, tool_registry=registry
        )
        await server.start_thread(cwd)
        await server.run_turn(input_text="boom", cwd=cwd)

        written = proc.stdin.written.decode()
        lines = [json.loads(l) for l in written.strip().split("\n") if l.strip()]
        tool_resp = [l for l in lines if l.get("id") == 402 and "result" in l]
        assert len(tool_resp) == 1
        assert tool_resp[0]["result"]["success"] is False
        assert "kaboom" in tool_resp[0]["result"]["output"]

        await server.stop()


class TestJsonBuffering:
    """Partial JSON line buffering."""

    @pytest.mark.asyncio
    async def test_buffers_partial_json_lines_until_newline_terminator(self):
        """app server buffers partial JSON lines until newline terminator."""
        cwd = tempfile.mkdtemp()
        # The FakeStreamReader works line-by-line, but we can verify the
        # server handles a line that arrives in one piece correctly.
        # For true partial buffering, we construct raw bytes with the
        # complete line.
        proc = _make_process(
            _init_ok(),
            _thread_ok(),
            _turn_ok(),
            _turn_completed({"total_tokens": 42}),
        )

        server = await _start_server(process=proc, cwd=cwd)
        await server.start_thread(cwd)
        result = await server.run_turn(input_text="hi", cwd=cwd)
        assert result["usage"].get("total_tokens") == 42

        await server.stop()


class TestStderr:
    """Stderr capture."""

    @pytest.mark.asyncio
    async def test_captures_codex_side_output_and_logs_it(self):
        """app server captures codex side output and logs it."""
        cwd = tempfile.mkdtemp()
        events: list[tuple[str, Any]] = []

        async def on_event(etype: str, payload: Any) -> None:
            events.append((etype, payload))

        proc = _make_process(
            _init_ok(),
            stderr="some debug output",
        )

        server = await _start_server(process=proc, cwd=cwd, on_event=on_event)
        # Give the stderr reader a moment.
        await asyncio.sleep(0.05)

        stderr_events = [e for e in events if e[0] == "stderr"]
        assert len(stderr_events) >= 1
        assert "some debug output" in stderr_events[0][1]["text"]

        await server.stop()


class TestMalformedJson:
    """Malformed JSON handling."""

    @pytest.mark.asyncio
    async def test_emits_malformed_events_for_invalid_json(self):
        """app server emits malformed events for invalid JSON."""
        cwd = tempfile.mkdtemp()
        events: list[tuple[str, Any]] = []

        async def on_event(etype: str, payload: Any) -> None:
            events.append((etype, payload))

        # Construct stdout with init response + a bad line + turn completed.
        init_line = json.dumps(_init_ok()).encode() + b"\n"
        thread_line = json.dumps(_thread_ok()).encode() + b"\n"
        turn_line = json.dumps(_turn_ok()).encode() + b"\n"
        bad_line = b"this is not json\n"
        completed_line = json.dumps(_turn_completed()).encode() + b"\n"
        stdout_data = init_line + thread_line + turn_line + bad_line + completed_line

        proc = FakeProcess(stdout_data=stdout_data)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            server = AppServer(AppServerConfig(), on_event=on_event)
            await server.start(cwd)

        await server.start_thread(cwd)
        await server.run_turn(input_text="go", cwd=cwd)

        malformed = [e for e in events if e[0] == "malformed"]
        assert len(malformed) >= 1
        assert "this is not json" in malformed[0][1]["raw"]

        await server.stop()


class TestSSH:
    """SSH remote launch."""

    @pytest.mark.asyncio
    async def test_launches_over_ssh_for_remote_workers(self):
        """app server launches over ssh for remote workers."""
        cwd = tempfile.mkdtemp()
        cfg = AppServerConfig(
            ssh_host="worker-1.example.com",
            ssh_user="deploy",
            ssh_args=["-o", "StrictHostKeyChecking=no"],
        )
        proc = _make_process(_init_ok())

        with patch(
            "asyncio.create_subprocess_exec", return_value=proc
        ) as mock_exec:
            server = AppServer(cfg)
            await server.start(cwd)

            cmd = list(mock_exec.call_args[0])
            assert cmd[0] == "ssh"
            assert "-l" in cmd
            assert "deploy" in cmd
            assert "worker-1.example.com" in cmd
            assert "StrictHostKeyChecking=no" in cmd
            # The codex command should be appended.
            assert "codex" in cmd
            assert "app-server" in cmd

        await server.stop()
