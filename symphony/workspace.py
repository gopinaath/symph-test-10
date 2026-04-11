"""Workspace management for agent execution environments."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from symphony.config import ValidationConfig
from symphony.path_safety import SafeResolveError, safe_resolve

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9._-]")
_DEFAULT_HOOK_TIMEOUT = 30  # seconds


@dataclass
class WorkspaceError:
    """Describes a workspace operation failure."""

    message: str


@dataclass
class WorkspaceConfig:
    """Configuration for workspace management."""

    root: str
    after_create: str | None = None
    before_run: str | None = None
    after_run: str | None = None
    before_remove: str | None = None
    hook_timeout: float = _DEFAULT_HOOK_TIMEOUT
    ssh_host: str | None = None  # None means local


@dataclass
class ValidationResult:
    """Result of a validation run."""

    passed: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0
    assertion_results: list[dict[str, object]] = field(default_factory=list)


@dataclass
class HookError:
    """Describes a hook execution failure."""

    message: str
    exit_code: int | None = None
    timed_out: bool = False


def _sanitize(identifier: str) -> str:
    """Replace characters that are not ``[a-zA-Z0-9._-]`` with ``_``."""
    return _SANITIZE_RE.sub("_", identifier)


async def _run_hook(
    command: str | None,
    cwd: str,
    *,
    timeout: float = _DEFAULT_HOOK_TIMEOUT,
    env_extra: dict[str, str] | None = None,
) -> HookError | None:
    """Run a shell hook command. Returns ``None`` on success or a :class:`HookError`."""
    if not command:
        return None

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            return HookError(
                message=f"hook timed out after {timeout}s",
                timed_out=True,
            )

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip() if stderr else ""
            return HookError(
                message=f"hook exited with code {proc.returncode}: {stderr_text}",
                exit_code=proc.returncode,
            )
    except OSError as exc:
        return HookError(message=str(exc))

    return None


async def _run_command_capturing(
    command: str,
    cwd: str,
    timeout: float,
) -> tuple[int, str, str]:
    """Run command and return (exit_code, stdout, stderr).

    Unlike ``_run_hook``, this always captures and returns both stdout and stderr.
    On timeout the process is killed and exit_code is returned as -1.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
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
            return (-1, "", f"command timed out after {timeout}s")

        stdout_text = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        stderr_text = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
        return (proc.returncode or 0, stdout_text, stderr_text)
    except OSError as exc:
        return (-1, "", str(exc))


class Workspace:
    """Manages working directories for agent tasks."""

    def __init__(self, config: WorkspaceConfig) -> None:
        self.config = config

    @property
    def root(self) -> Path:
        """Return the resolved workspace root."""
        return Path(self.config.root).resolve()

    def path_for(self, identifier: str) -> Path:
        """Return the workspace directory path for *identifier* (deterministic)."""
        return self.root / _sanitize(identifier)

    # ------------------------------------------------------------------
    # create
    # ------------------------------------------------------------------

    async def create(self, identifier: str) -> Path | WorkspaceError:
        """Create (or reuse) a workspace directory for *identifier*.

        If an ``after_create`` hook is configured it is executed only when the
        directory is freshly created.
        """
        root = self.root
        os.makedirs(root, exist_ok=True)

        safe = safe_resolve(_sanitize(identifier), root)
        if isinstance(safe, SafeResolveError):
            return WorkspaceError(safe.message)

        ws_path = safe

        # If something exists at the path that is NOT a directory, replace it.
        if ws_path.exists() and not ws_path.is_dir():
            ws_path.unlink()

        freshly_created = not ws_path.exists()

        if freshly_created:
            ws_path.mkdir(parents=True, exist_ok=True)

        # Run after_create hook only for fresh directories.
        if freshly_created and self.config.after_create:
            err = await _run_hook(
                self.config.after_create,
                str(ws_path),
                timeout=self.config.hook_timeout,
                env_extra={"WORKSPACE": str(ws_path), "IDENTIFIER": identifier},
            )
            if err is not None:
                # Clean up the freshly created directory on hook failure.
                shutil.rmtree(ws_path, ignore_errors=True)
                return WorkspaceError(err.message)

        return ws_path

    # ------------------------------------------------------------------
    # remove
    # ------------------------------------------------------------------

    async def remove(self, identifier: str) -> WorkspaceError | None:
        """Remove the workspace for *identifier*.

        Runs ``before_remove`` hook first.  If the hook fails the removal
        still proceeds (best-effort).

        Returns ``None`` on success or a :class:`WorkspaceError` on failure.
        """
        root = self.root

        safe = safe_resolve(_sanitize(identifier), root)
        if isinstance(safe, SafeResolveError):
            return WorkspaceError(safe.message)

        ws_path = safe

        # Must not remove the root itself.
        if ws_path == root:
            return WorkspaceError("refusing to remove workspace root")

        if not ws_path.exists():
            return WorkspaceError(f"workspace does not exist: {ws_path}")

        # Run before_remove hook (continue on failure).
        await _run_hook(
            self.config.before_remove,
            str(ws_path),
            timeout=self.config.hook_timeout,
            env_extra={"WORKSPACE": str(ws_path), "IDENTIFIER": identifier},
        )

        shutil.rmtree(ws_path, ignore_errors=True)
        return None

    # ------------------------------------------------------------------
    # hook runners
    # ------------------------------------------------------------------

    async def run_before_run(self, identifier: str) -> HookError | None:
        ws = self.path_for(identifier)
        return await _run_hook(
            self.config.before_run,
            str(ws),
            timeout=self.config.hook_timeout,
            env_extra={"WORKSPACE": str(ws), "IDENTIFIER": identifier},
        )

    async def run_after_run(self, identifier: str) -> HookError | None:
        ws = self.path_for(identifier)
        return await _run_hook(
            self.config.after_run,
            str(ws),
            timeout=self.config.hook_timeout,
            env_extra={"WORKSPACE": str(ws), "IDENTIFIER": identifier},
        )

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    async def run_validation(
        self, identifier: str, config: ValidationConfig,
    ) -> ValidationResult:
        """Run validation suite: deterministic assertions first, then test command."""
        ws = self.path_for(identifier)
        ws_str = str(ws)
        start = time.monotonic()

        assertion_results: list[dict[str, object]] = []

        # 1. Run deterministic assertions first
        for assertion in config.assertions:
            kind = assertion.get("kind")
            result: dict[str, object]

            if kind == "file_exists":
                path = ws / assertion["path"]
                passed = path.exists()
                result = {
                    "kind": kind,
                    "path": assertion["path"],
                    "passed": passed,
                }
                assertion_results.append(result)
                if not passed:
                    elapsed = (time.monotonic() - start) * 1000
                    return ValidationResult(
                        passed=False,
                        assertion_results=assertion_results,
                        duration_ms=elapsed,
                    )

            elif kind == "file_contains":
                path = ws / assertion["path"]
                pattern = assertion["pattern"]
                try:
                    content = path.read_text()
                    passed = re.search(pattern, content) is not None
                except (OSError, FileNotFoundError):
                    passed = False
                result = {
                    "kind": kind,
                    "path": assertion["path"],
                    "pattern": pattern,
                    "passed": passed,
                }
                assertion_results.append(result)
                if not passed:
                    elapsed = (time.monotonic() - start) * 1000
                    return ValidationResult(
                        passed=False,
                        assertion_results=assertion_results,
                        duration_ms=elapsed,
                    )

            elif kind == "command_exit_code":
                cmd = assertion["command"]
                expected = assertion.get("expected", 0)
                timeout_s = config.timeout_ms / 1000
                exit_code, cmd_stdout, cmd_stderr = await _run_command_capturing(
                    cmd, ws_str, timeout_s,
                )
                passed = exit_code == expected
                result = {
                    "kind": kind,
                    "command": cmd,
                    "expected": expected,
                    "actual": exit_code,
                    "passed": passed,
                }
                assertion_results.append(result)
                if not passed:
                    elapsed = (time.monotonic() - start) * 1000
                    return ValidationResult(
                        passed=False,
                        exit_code=exit_code,
                        stdout=cmd_stdout,
                        stderr=cmd_stderr,
                        assertion_results=assertion_results,
                        duration_ms=elapsed,
                    )

        # 2. If all assertions pass and config.command is set, run the test command
        if config.command:
            timeout_s = config.timeout_ms / 1000
            exit_code, stdout, stderr = await _run_command_capturing(
                config.command, ws_str, timeout_s,
            )
            elapsed = (time.monotonic() - start) * 1000
            return ValidationResult(
                passed=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                assertion_results=assertion_results,
                duration_ms=elapsed,
            )

        # No command and all assertions passed
        elapsed = (time.monotonic() - start) * 1000
        return ValidationResult(
            passed=True,
            assertion_results=assertion_results,
            duration_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # bulk cleanup
    # ------------------------------------------------------------------

    async def remove_all(self) -> list[WorkspaceError]:
        """Remove every workspace directory under the root.

        Returns a list of errors (empty on full success).
        """
        root = self.root
        if not root.exists():
            return []

        errors: list[WorkspaceError] = []
        for entry in root.iterdir():
            err = await self.remove(entry.name)
            if err is not None:
                errors.append(err)
        return errors

    async def cleanup(self, identifiers: list[str | None]) -> list[WorkspaceError]:
        """Remove workspaces for the given identifiers.

        ``None`` entries in the list are silently skipped.
        """
        root = self.root
        if not root.exists():
            return []

        errors: list[WorkspaceError] = []
        for ident in identifiers:
            if ident is None:
                continue
            err = await self.remove(ident)
            if err is not None:
                errors.append(err)
        return errors
