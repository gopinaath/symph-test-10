"""Workflow loading from ``WORKFLOW.md``.

A workflow file consists of optional YAML front matter delimited by ``---``
lines followed by a Jinja2 template body.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

import yaml

from symphony.config import Config

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@dataclass
class Workflow:
    """Parsed representation of a ``WORKFLOW.md`` file."""

    config: Config
    prompt_template: str

    @classmethod
    def parse(cls, path: str) -> Workflow:
        """Parse a workflow file at *path*.

        The file may contain:
        - Only a prompt (no YAML front matter).
        - YAML front matter delimited by ``---`` lines, followed by a prompt.
        - An unterminated front-matter block (``---`` at the start but no
          closing ``---``), which is treated as YAML-only with an empty prompt.

        Raises ``FileNotFoundError`` if *path* does not exist, and
        ``ValueError`` if the front matter is not a YAML mapping.
        """
        text = _read_file(path)
        yaml_data, prompt = _split_front_matter(text)

        if yaml_data is not None:
            parsed = yaml.safe_load(yaml_data)
            if parsed is None:
                parsed = {}
            if not isinstance(parsed, dict):
                raise ValueError(f"YAML front matter in {path} must be a mapping, got {type(parsed).__name__}")
            config = Config.from_yaml(parsed)
        else:
            config = Config()

        return cls(config=config, prompt_template=prompt)


def _read_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Workflow file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _split_front_matter(text: str) -> tuple[str | None, str]:
    """Return ``(yaml_str | None, prompt_body)``."""
    stripped = text.lstrip("\n")
    if not stripped.startswith("---"):
        # No front matter — the whole file is the prompt.
        return None, text

    # Find closing ---
    after_first = stripped[3:]
    # Skip the rest of the opening --- line
    newline_pos = after_first.find("\n")
    if newline_pos == -1:
        # Only "---" in the file — empty YAML, empty prompt.
        return "", ""

    body_after_opening = after_first[newline_pos + 1 :]
    close_pos = body_after_opening.find("\n---")
    if close_pos == -1:
        # Unterminated front matter — treat everything after opening as YAML,
        # prompt is empty.
        return body_after_opening, ""

    yaml_str = body_after_opening[:close_pos]
    rest = body_after_opening[close_pos + 4 :]  # skip "\n---"
    # Skip remainder of the closing --- line
    nl = rest.find("\n")
    prompt = "" if nl == -1 else rest[nl + 1 :]

    return yaml_str, prompt


# ---------------------------------------------------------------------------
# WorkflowStore — cached, polling-based re-loader
# ---------------------------------------------------------------------------


@dataclass
class _FileStamp:
    mtime: float = 0.0
    size: int = 0


@dataclass
class WorkflowStore:
    """Caches the last good ``Workflow`` and polls for changes.

    Uses ``asyncio`` for the polling loop and ``asyncio.Lock`` for
    concurrent-safe cache access.  Call :meth:`start` / :meth:`stop`
    to manage the background polling task lifecycle.
    """

    path: str
    poll_interval: float = 1.0  # seconds

    _workflow: Workflow | None = field(default=None, init=False, repr=False)
    _stamp: _FileStamp = field(default_factory=_FileStamp, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _poll_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _last_error: Exception | None = field(default=None, init=False, repr=False)
    _stopped: bool = field(default=False, init=False, repr=False)

    # -- public API ----------------------------------------------------------

    async def init(self) -> Workflow:
        """Load the workflow for the first time.

        Raises on failure (file missing / invalid).
        """
        wf = Workflow.parse(self.path)
        async with self._lock:
            self._workflow = wf
            self._stamp = self._current_stamp()
        return wf

    @property
    def workflow(self) -> Workflow | None:
        """Return the cached workflow (non-blocking snapshot)."""
        return self._workflow

    async def get_workflow(self) -> Workflow | None:
        """Return the cached workflow under the async lock."""
        async with self._lock:
            return self._workflow

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    async def start(self) -> None:
        """Begin background polling for file changes."""
        self._stopped = False
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Cancel the polling task and wait for it to finish."""
        self._stopped = True
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    # -- internal ------------------------------------------------------------

    def _current_stamp(self) -> _FileStamp:
        try:
            st = os.stat(self.path)
            return _FileStamp(mtime=st.st_mtime, size=st.st_size)
        except OSError:
            return _FileStamp()

    async def _poll_loop(self) -> None:
        while not self._stopped:
            try:
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                return
            if self._stopped:
                break
            await self._try_reload()

    async def _try_reload(self) -> None:
        stamp = self._current_stamp()
        async with self._lock:
            if stamp.mtime == self._stamp.mtime and stamp.size == self._stamp.size:
                return

        # File changed — attempt reload.
        try:
            wf = Workflow.parse(self.path)
            async with self._lock:
                self._workflow = wf
                self._stamp = stamp
                self._last_error = None
        except Exception as exc:
            # Keep last good workflow.
            async with self._lock:
                self._stamp = stamp  # avoid retrying same bad stamp every tick
                self._last_error = exc
