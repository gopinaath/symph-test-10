"""Workspace management for agent execution environments."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

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
