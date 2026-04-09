"""Terminal status dashboard renderer.

Renders an ANSI-coloured terminal UI showing the current state of the
Symphony orchestrator: running agents, retry queue, token throughput,
rate limits, and poll status.

This is the Python equivalent of the Elixir ``StatusDashboard`` GenServer.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from symphony.observability import (
    ANSI_RE,
    EVENT_MAP,
    SPARKLINE_CHARS,
    TPSTracker,
    format_timestamp,
    humanize_event,
    sparkline,
    strip_ansi,
)
from symphony.orchestrator import OrchestratorSnapshot, RunningEntry, RetryEntry

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _green(text: str) -> str:
    return f"{GREEN}{text}{RESET}"


def _yellow(text: str) -> str:
    return f"{YELLOW}{text}{RESET}"


def _red(text: str) -> str:
    return f"{RED}{text}{RESET}"


def _cyan(text: str) -> str:
    return f"{CYAN}{text}{RESET}"


def _bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"


def _dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


def _state_colour(state: str) -> str:
    """Apply colour to a state badge string."""
    low = state.lower()
    if low in ("running", "in progress", "in_progress"):
        return _green(state)
    if low in ("retrying", "waiting", "queued"):
        return _yellow(state)
    if low in ("failed", "error", "errored"):
        return _red(state)
    return _cyan(state)


# ---------------------------------------------------------------------------
# Box-drawing helpers
# ---------------------------------------------------------------------------

BOX_H = "\u2500"   # ─
BOX_V = "\u2502"   # │
BOX_TL = "\u250c"  # ┌
BOX_TR = "\u2510"  # ┐
BOX_BL = "\u2514"  # └
BOX_BR = "\u2518"  # ┘
BOX_ML = "\u251c"  # ├
BOX_MR = "\u2524"  # ┤


def _hline(width: int, left: str = BOX_TL, right: str = BOX_TR) -> str:
    """Horizontal rule with corner/tee characters."""
    return left + BOX_H * (width - 2) + right


# ---------------------------------------------------------------------------
# Visible-length helper (ignores ANSI escapes for padding)
# ---------------------------------------------------------------------------

def _visible_len(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def _pad(s: str, width: int) -> str:
    """Pad *s* with spaces to reach *width* visible characters."""
    deficit = width - _visible_len(s)
    if deficit > 0:
        return s + " " * deficit
    return s


# ---------------------------------------------------------------------------
# StatusDashboard
# ---------------------------------------------------------------------------

@dataclass
class StatusDashboard:
    """Renders a terminal status dashboard from an ``OrchestratorSnapshot``.

    Features
    --------
    * Header line with title, project URL, dashboard URL.
    * Agent table (ISSUE, STATE, SESSION, TURNS, TOKENS, EVENT, STARTED, HOST).
    * Retry queue table (ISSUE, ATTEMPT, DUE IN, ERROR, HOST).
    * Token throughput line (rolling 5 s TPS + 10-minute sparkline).
    * Rate-limits display.
    * Poll status (countdown or "checking now...").
    * Coalesced rendering: at most one render per ``render_interval_ms``.
    """

    project_url: str = ""
    server_host: str = ""
    server_port: Optional[int] = None
    bound_port: Optional[int] = None
    render_interval_ms: int = 16
    terminal_width: Optional[int] = None

    # -- internal bookkeeping (not constructor args) -------------------------
    _last_render_at: float = field(default=0.0, init=False, repr=False)
    _pending: bool = field(default=False, init=False, repr=False)
    _render_count: int = field(default=0, init=False, repr=False)
    _tps_tracker: TPSTracker = field(default_factory=TPSTracker, init=False, repr=False)

    # -- public API ----------------------------------------------------------

    @property
    def render_count(self) -> int:
        return self._render_count

    def get_terminal_width(self) -> int:
        if self.terminal_width is not None:
            return self.terminal_width
        try:
            return os.get_terminal_size().columns
        except (OSError, ValueError):
            return 120

    def render(self, snapshot: OrchestratorSnapshot) -> str:
        """Render the full dashboard as a string."""
        width = self.get_terminal_width()
        lines: list[str] = []
        lines.append(self._render_header(width))
        lines.append(_hline(width))

        # Agent table
        has_running = bool(snapshot.running)
        if has_running:
            lines.append(self._render_agent_header(width))
            lines.append(_hline(width, left=BOX_ML, right=BOX_MR))
            for _ident, entry in sorted(snapshot.running.items()):
                lines.append(self._render_agent_row(entry, width))
            lines.append(_hline(width, left=BOX_ML, right=BOX_MR))

        # Retry queue
        has_retry = bool(snapshot.retry_queue)
        if has_retry:
            if has_running:
                lines.append("")  # spacer before backoff queue when agents shown
            else:
                lines.append("")  # spacer before backoff queue without agents
            lines.append(self._render_retry_header(width))
            lines.append(_hline(width, left=BOX_ML, right=BOX_MR))
            now = datetime.now(timezone.utc)
            for _ident, entry in sorted(snapshot.retry_queue.items()):
                lines.append(self._render_retry_row(entry, width, now))
            lines.append(_hline(width, left=BOX_ML, right=BOX_MR))

        # Throughput
        lines.append(self._render_throughput(snapshot, width))

        # Rate limits
        if snapshot.codex_rate_limits:
            lines.append(self._render_rate_limits(snapshot.codex_rate_limits, width))

        # Poll status
        lines.append(self._render_poll_status(snapshot, width))
        lines.append(_hline(width, left=BOX_BL, right=BOX_BR))

        self._render_count += 1
        return "\n".join(lines)

    def render_offline(self) -> str:
        """Render an offline placeholder dashboard."""
        width = self.get_terminal_width()
        lines: list[str] = []
        lines.append(self._render_header(width))
        lines.append(_hline(width))
        lines.append(_pad(f"{BOX_V}  {_dim('offline')}  {BOX_V}", width))
        lines.append(_hline(width, left=BOX_BL, right=BOX_BR))
        self._render_count += 1
        return "\n".join(lines)

    def maybe_render(self, snapshot: OrchestratorSnapshot) -> Optional[str]:
        """Coalesced render: returns the rendered string only if enough time
        has elapsed since the last render. Otherwise marks a pending update
        and returns ``None``.
        """
        now = time.monotonic()
        interval_s = self.render_interval_ms / 1000.0
        elapsed = now - self._last_render_at
        if elapsed >= interval_s:
            self._last_render_at = now
            self._pending = False
            return self.render(snapshot)
        else:
            self._pending = True
            return None

    def flush_pending(self, snapshot: OrchestratorSnapshot) -> Optional[str]:
        """If there is a pending update, render it now regardless of interval."""
        if self._pending:
            self._last_render_at = time.monotonic()
            self._pending = False
            return self.render(snapshot)
        return None

    # -- private rendering helpers -------------------------------------------

    def _dashboard_url(self) -> Optional[str]:
        """Build the dashboard URL from server config."""
        port = self.bound_port or self.server_port
        if port is None:
            return None
        host = self.server_host or "127.0.0.1"
        # Normalise wildcard hosts
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{port}"

    def _render_header(self, width: int) -> str:
        parts = [f"{BOX_V} {_bold('Symphony')}"]
        if self.project_url:
            parts.append(_cyan(self.project_url))
        dash_url = self._dashboard_url()
        if dash_url:
            parts.append(_dim(f"dashboard: {dash_url}"))
        header_inner = "  ".join(parts) + f"  {BOX_V}"
        return _pad(header_inner, width)

    def _render_agent_header(self, width: int) -> str:
        cols = ["ISSUE", "STATE", "SESSION", "TURNS", "TOKENS", "EVENT", "STARTED", "HOST"]
        row = f"{BOX_V} " + "  ".join(f"{_bold(c):>10s}" if i > 1 else f"{_bold(c):<12s}" for i, c in enumerate(cols)) + f"  {BOX_V}"
        return _pad(row, width)

    def _render_agent_row(self, entry: RunningEntry, width: int) -> str:
        issue_id = entry.issue.identifier[:12]
        state = entry.issue.state or "unknown"
        session = (entry.session_id or "")[:10]
        turns = str(entry.turn_count)
        tokens = str(entry.total_tokens)
        event = strip_ansi(entry.last_event or "")[:20]
        started = format_timestamp(entry.started_at) if entry.started_at else ""
        host = (entry.worker_host or "local")[:12]

        cols = [
            f"{issue_id:<12s}",
            f"{_state_colour(state):<12s}",
            f"{session:>10s}",
            f"{turns:>10s}",
            f"{tokens:>10s}",
            f"{event:>10s}",
            f"{started:>10s}",
            f"{host:>10s}",
        ]
        row = f"{BOX_V} " + "  ".join(cols) + f"  {BOX_V}"
        return _pad(row, width)

    def _render_retry_header(self, width: int) -> str:
        title = f"{BOX_V} {_bold('BACKOFF QUEUE')}"
        cols = ["ISSUE", "ATTEMPT", "DUE IN", "ERROR", "HOST"]
        row = f"{BOX_V} " + "  ".join(f"{_bold(c):<12s}" for c in cols) + f"  {BOX_V}"
        return _pad(f"{title}  {BOX_V}\n" + row, width)

    def _render_retry_row(
        self, entry: RetryEntry, width: int, now: datetime
    ) -> str:
        issue_id = entry.issue.identifier[:12]
        attempt = str(entry.attempt)
        due_delta = entry.due_at - now
        due_secs = max(0, int(due_delta.total_seconds()))
        due_in = f"{due_secs}s"
        error = strip_ansi(entry.error or "")[:30]
        host = (entry.preferred_host or "any")[:12]

        cols = [
            f"{issue_id:<12s}",
            f"{attempt:<12s}",
            f"{due_in:<12s}",
            f"{_red(error):<12s}",
            f"{host:<12s}",
        ]
        row = f"{BOX_V} " + "  ".join(cols) + f"  {BOX_V}"
        return _pad(row, width)

    def _render_throughput(self, snapshot: OrchestratorSnapshot, width: int) -> str:
        totals = snapshot.codex_totals
        total_tok = totals.get("total_tokens", 0)
        tps = self._tps_tracker.tps(time.monotonic())
        graph = sparkline(self._tps_tracker._history, width=30)
        line = (
            f"{BOX_V}  Tokens: {total_tok:,}  "
            f"TPS: {tps:.0f}  "
            f"{graph}  {BOX_V}"
        )
        return _pad(line, width)

    def _render_rate_limits(self, limits: dict, width: int) -> str:
        parts = []
        for k, v in sorted(limits.items()):
            parts.append(f"{k}={v}")
        line = f"{BOX_V}  Rate limits: {', '.join(parts)}  {BOX_V}"
        return _pad(line, width)

    def _render_poll_status(self, snapshot: OrchestratorSnapshot, width: int) -> str:
        if snapshot.poll_checking:
            status = _yellow("checking now...")
        else:
            secs = max(0, snapshot.poll_countdown_ms // 1000)
            status = f"next refresh in {secs}s"
        line = f"{BOX_V}  {status}  {BOX_V}"
        return _pad(line, width)

    # -- TPS feed (called externally when token updates arrive) ---------------

    def record_tokens(self, timestamp: float, total_tokens: int) -> None:
        """Feed token counts into the internal TPS tracker."""
        self._tps_tracker.record(timestamp, total_tokens)
