"""PR cleanup on workspace removal.

Ports the Elixir ``workspace.before_remove`` mix task.  Uses the ``gh`` CLI
to find and close open pull requests for a given branch.
"""

from __future__ import annotations

import asyncio
import json
import shutil

_DEFAULT_TIMEOUT: float = 30.0


async def _run(
    *args: str,
    timeout: float = _DEFAULT_TIMEOUT,
) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr).

    Uses :func:`asyncio.create_subprocess_exec` so callers stay async.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", "timeout"

    return (
        proc.returncode or 0,
        stdout_bytes.decode(errors="replace") if stdout_bytes else "",
        stderr_bytes.decode(errors="replace") if stderr_bytes else "",
    )


def _is_on_path(name: str) -> bool:
    """Return True if *name* is found on PATH."""
    return shutil.which(name) is not None


async def get_current_branch(*, timeout: float = _DEFAULT_TIMEOUT) -> str | None:
    """Return the current git branch name, or ``None`` on failure."""
    if not _is_on_path("git"):
        return None

    rc, stdout, _ = await _run("git", "rev-parse", "--abbrev-ref", "HEAD", timeout=timeout)
    if rc != 0:
        return None

    branch = stdout.strip()
    return branch if branch else None


async def close_prs_for_branch(
    branch: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[dict[str, object]]:
    """Find and close all open PRs whose head matches *branch*.

    Returns a list of dicts, each with keys ``number``, ``url``,
    ``success``, and optionally ``error``.
    """
    # 1. Check gh is available
    if not _is_on_path("gh"):
        return []

    # 2. Check gh is authenticated
    rc, _, _ = await _run("gh", "auth", "status", timeout=timeout)
    if rc != 0:
        return []

    # 3. List open PRs for this branch
    rc, stdout, _ = await _run(
        "gh", "pr", "list",
        "--head", branch,
        "--state", "open",
        "--json", "number,url",
        timeout=timeout,
    )
    if rc != 0:
        return []

    try:
        prs: list[dict[str, object]] = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return []

    if not prs:
        return []

    # 4. Close each PR, tolerating individual failures
    results: list[dict[str, object]] = []
    for pr in prs:
        number = pr.get("number")
        url = pr.get("url", "")
        rc, _, stderr = await _run(
            "gh", "pr", "close", str(number), timeout=timeout
        )
        entry: dict[str, object] = {
            "number": number,
            "url": url,
            "success": rc == 0,
        }
        if rc != 0:
            entry["error"] = f"Failed to close PR #{number}" + (
                f": {stderr.strip()}" if stderr.strip() else ""
            )
        results.append(entry)

    return results


async def cleanup_workspace_prs(
    branch: str | None = None,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[dict[str, object]]:
    """Main entry point: close all open PRs for *branch*.

    If *branch* is ``None``, the current git branch is used.
    If the branch is blank or unavailable, returns an empty list (no-op).
    """
    if branch is None:
        branch = await get_current_branch(timeout=timeout)

    if not branch or not branch.strip():
        return []

    return await close_prs_for_branch(branch.strip(), timeout=timeout)
