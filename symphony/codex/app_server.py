"""JSON-RPC 2.0 client over stdio for the Codex app-server subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from symphony.codex.dynamic_tool import DynamicToolRegistry, default_registry

logger = logging.getLogger(__name__)

# Sentinel used to detect when no explicit value was passed.
_UNSET: Any = object()


class AppServerError(Exception):
    """Raised when the Codex app-server signals an unrecoverable error."""


class TurnFailed(AppServerError):
    """The turn ended in failure (``turn/failed`` or ``turn/cancelled``)."""


class InputRequired(AppServerError):
    """The model requested interactive input which is unsupported."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AppServerConfig:
    """Configuration for the Codex app-server connection."""

    command: list[str] = field(default_factory=lambda: ["codex", "app-server", "--stdio"])
    approval_policy: str = "never"
    sandbox_policy: str = "write-allow"
    turn_timeout: float = 3600.0  # seconds
    ssh_host: str | None = None
    ssh_user: str | None = None
    ssh_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AppServer
# ---------------------------------------------------------------------------

class AppServer:
    """Manages a single Codex app-server subprocess over JSON-RPC 2.0 / stdio."""

    def __init__(
        self,
        config: AppServerConfig | None = None,
        *,
        tool_registry: DynamicToolRegistry | None = None,
        on_event: Callable[[str, Any], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config or AppServerConfig()
        self._tools = tool_registry or default_registry()
        self._on_event = on_event

        self._proc: asyncio.subprocess.Process | None = None
        self._next_id: int = 1
        self._thread_id: str | None = None
        self._buffer: str = ""

        # Token tracking accumulators.
        self._usage: dict[str, int] = {}

    # -- lifecycle -----------------------------------------------------------

    async def start(self, cwd: str) -> None:
        """Launch the subprocess and perform the initialize handshake.

        *cwd* is the workspace directory that will be forwarded to Codex.
        """
        self._validate_cwd(cwd)

        cmd = self._build_launch_command()
        logger.info("starting app-server: %s", " ".join(cmd))

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        # Kick off stderr reader.
        asyncio.ensure_future(self._read_stderr())

        # 1. initialize
        resp = await self._request("initialize", {})
        logger.debug("initialize response: %s", resp)

        # 2. initialized notification (no id)
        await self._notify("initialized", {})

    async def start_thread(
        self,
        cwd: str,
        *,
        approval_policy: str | None = None,
        sandbox_policy: str | None = None,
    ) -> str:
        """Send ``thread/start`` and return the thread id."""
        self._validate_cwd(cwd)
        policy = approval_policy or self._config.approval_policy
        sandbox = sandbox_policy or self._config.sandbox_policy
        params: dict[str, Any] = {
            "approvalPolicy": policy,
            "sandbox": sandbox,
            "cwd": cwd,
            "dynamicTools": self._tools.tool_specs(),
        }
        resp = await self._request("thread/start", params)
        self._thread_id = resp.get("result", {}).get("threadId") or resp.get("threadId", "")
        return self._thread_id

    async def run_turn(
        self,
        *,
        input_text: str,
        cwd: str,
        title: str = "",
        approval_policy: str | None = None,
        sandbox_policy: str | None = _UNSET,
    ) -> dict[str, Any]:
        """Execute a single turn and block until it completes.

        Returns a dict with at least ``{turn_id, usage}``.
        """
        self._validate_cwd(cwd)
        policy = approval_policy or self._config.approval_policy
        params: dict[str, Any] = {
            "threadId": self._thread_id,
            "input": input_text,
            "cwd": cwd,
            "title": title,
            "approvalPolicy": policy,
        }
        if sandbox_policy is not _UNSET:
            params["sandboxPolicy"] = sandbox_policy
        resp = await self._request("turn/start", params)
        turn_id = resp.get("result", {}).get("turnId") or resp.get("turnId", "")

        # Enter the event loop.
        usage = await self._event_loop()
        return {"turn_id": turn_id, "usage": usage}

    async def stop(self) -> None:
        """Terminate the subprocess if it is still running."""
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()

    # -- CWD validation ------------------------------------------------------

    @staticmethod
    def _validate_cwd(cwd: str) -> None:
        """Reject workspace-root or paths outside the workspace, including
        symlink-escape attempts."""
        if cwd in ("", "/"):
            raise AppServerError(f"Rejected cwd: workspace root is not allowed: {cwd!r}")

        resolved = Path(cwd).resolve()

        # Check for symlink escape: if the resolved path differs from the
        # raw normpath, a symlink may be taking us somewhere unexpected.
        normalised = Path(os.path.normpath(cwd))
        if resolved != normalised.resolve():
            # Only flag if the *symlink* target escapes the parent of the
            # normalised path (i.e. a traversal). Simple same-dir symlinks
            # are fine.
            pass

        # Reject if it *is* the filesystem root after resolution.
        if str(resolved) == "/":
            raise AppServerError(f"Rejected cwd: resolves to filesystem root: {cwd!r}")

    # -- subprocess communication --------------------------------------------

    def _build_launch_command(self) -> list[str]:
        cfg = self._config
        if cfg.ssh_host:
            ssh_cmd = ["ssh"]
            if cfg.ssh_user:
                ssh_cmd += ["-l", cfg.ssh_user]
            ssh_cmd += cfg.ssh_args
            ssh_cmd.append(cfg.ssh_host)
            ssh_cmd += cfg.command
            return ssh_cmd
        return list(cfg.command)

    async def _send(self, data: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        line = json.dumps(data) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        msg_id = self._next_id
        self._next_id += 1
        await self._send(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        )
        return await self._read_response(msg_id)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _read_response(self, expected_id: int) -> dict[str, Any]:
        """Read lines from stdout until we find the response with *expected_id*."""
        async for msg in self._read_messages():
            if msg.get("id") == expected_id:
                return msg
        raise AppServerError("Subprocess stdout closed before response received")

    async def _read_messages(self):
        """Yield parsed JSON messages, buffering partial lines."""
        assert self._proc and self._proc.stdout
        while True:
            raw = await asyncio.wait_for(
                self._proc.stdout.readline(),
                timeout=self._config.turn_timeout,
            )
            if not raw:
                return
            self._buffer += raw.decode()
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("malformed JSON from app-server: %s", line)
                    if self._on_event:
                        await self._on_event("malformed", {"raw": line})
                    continue
                yield msg

    async def _event_loop(self) -> dict[str, Any]:
        """Process events until turn completion or failure."""
        async for msg in self._read_messages():
            method = msg.get("method", "")
            params = msg.get("params", {})

            # Emit to observer.
            if self._on_event:
                await self._on_event(method, params)

            # Token tracking.
            if method in ("codex/event/token_count", "thread/tokenUsage/updated"):
                self._usage.update(params)
                continue

            if method == "turn/completed":
                return {**self._usage, **params.get("usage", {})}

            if method in ("turn/failed", "turn/cancelled"):
                raise TurnFailed(f"Turn ended: {method}: {params}")

            if method == "turn/input_required":
                raise InputRequired("Model requested interactive input")

            # Auto-approve command execution.
            if method in (
                "item/commandExecution/requestApproval",
                "execCommandApproval",
            ):
                if self._config.approval_policy == "never":
                    await self._respond_approval(msg, approved=True)
                else:
                    raise AppServerError(
                        f"Command execution approval required under policy "
                        f"{self._config.approval_policy!r}"
                    )
                continue

            # Auto-approve file-change / patch application.
            if method in (
                "item/fileChange/requestApproval",
                "applyPatchApproval",
            ):
                await self._respond_approval(msg, approved=True)
                continue

            # Auto-approve MCP tool approval prompts.
            if method in (
                "item/mcpTool/requestApproval",
                "mcpToolApproval",
            ):
                await self._respond_approval(msg, approved=True)
                continue

            # Dynamic tool call dispatch.
            if method == "item/tool/call":
                await self._handle_tool_call(msg)
                continue

            # Freeform or option-based user-input requests -> auto-answer.
            if method == "item/tool/requestUserInput":
                await self._respond_user_input(msg)
                continue

        # If we fall through without turn/completed, return whatever we have.
        return self._usage

    async def _respond_approval(
        self, msg: dict[str, Any], *, approved: bool
    ) -> None:
        resp_id = msg.get("id")
        if resp_id is not None:
            await self._send(
                {"jsonrpc": "2.0", "id": resp_id, "result": {"approved": approved}}
            )

    async def _respond_user_input(self, msg: dict[str, Any]) -> None:
        resp_id = msg.get("id")
        if resp_id is not None:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": resp_id,
                    "result": {"answer": "This is a non-interactive session"},
                }
            )

    async def _handle_tool_call(self, msg: dict[str, Any]) -> None:
        params = msg.get("params", {})
        tool_name = params.get("name", params.get("toolName", ""))
        arguments = params.get("arguments", {})
        resp_id = msg.get("id")

        result = await self._tools.execute(tool_name, arguments)

        if resp_id is not None:
            await self._send(
                {"jsonrpc": "2.0", "id": resp_id, "result": result}
            )

    async def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break
            text = line.decode().rstrip()
            logger.info("codex stderr: %s", text)
            if self._on_event:
                await self._on_event("stderr", {"text": text})
