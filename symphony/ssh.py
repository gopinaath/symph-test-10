"""SSH utilities for remote workspace and agent execution."""

import asyncio
import os
import shutil
from dataclasses import dataclass


@dataclass
class SSHTarget:
    user: str | None
    host: str
    port: int | None


def parse_ssh_target(target: str) -> SSHTarget:
    """Parse user@host:port into components.

    Handles:
    - host
    - host:port
    - user@host
    - user@host:port
    - user@[::1]:port (bracketed IPv6)
    """
    user = None
    port = None

    if "@" in target:
        user, rest = target.split("@", 1)
    else:
        rest = target

    # bracketed IPv6
    if rest.startswith("["):
        bracket_end = rest.find("]")
        if bracket_end == -1:
            return SSHTarget(user=user, host=rest, port=None)
        host = rest[: bracket_end + 1]
        after = rest[bracket_end + 1 :]
        if after.startswith(":"):
            port = int(after[1:])
        return SSHTarget(user=user, host=host, port=port)

    # Check for port suffix, but only if it looks like host:port (not bare IPv6)
    if ":" in rest and not rest.count(":") > 1:
        host_part, port_str = rest.rsplit(":", 1)
        try:
            port = int(port_str)
            rest = host_part
        except ValueError:
            pass

    return SSHTarget(user=user, host=rest, port=port)


def build_ssh_command(
    target: str,
    remote_command: str,
    ssh_config: str | None = None,
    allocate_tty: bool = False,
) -> list[str]:
    """Build an SSH command list."""
    ssh_path = shutil.which("ssh")
    if ssh_path is None:
        raise FileNotFoundError("ssh not found on PATH")

    parsed = parse_ssh_target(target)
    cmd = [ssh_path]

    if not allocate_tty:
        cmd.append("-T")

    if ssh_config is None:
        ssh_config = os.environ.get("SYMPHONY_SSH_CONFIG")
    if ssh_config:
        cmd.extend(["-F", ssh_config])

    if parsed.port is not None:
        cmd.extend(["-p", str(parsed.port)])

    host_part = parsed.host
    if parsed.user:
        host_part = f"{parsed.user}@{host_part}"

    cmd.append(host_part)
    cmd.append(remote_shell_command(remote_command))

    return cmd


def remote_shell_command(command: str) -> str:
    """Wrap a command in bash -lc with proper escaping."""
    escaped = command.replace("'", "'\\''")
    return f"bash -lc '{escaped}'"


async def ssh_run(
    target: str,
    command: str,
    timeout: float | None = None,
    ssh_config: str | None = None,
) -> tuple[int, str, str]:
    """Run a command over SSH and return (returncode, stdout, stderr)."""
    cmd = build_ssh_command(target, command, ssh_config=ssh_config)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", "timeout"

    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")
