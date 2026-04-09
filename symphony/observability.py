"""Observability layer: PubSub for dashboard updates and terminal status rendering."""

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


class PubSub:
    """Simple async pub/sub for dashboard updates."""

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers = [s for s in self._subscribers if s is not q]

    async def broadcast(self, message: Any):
        for q in self._subscribers:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass

    def broadcast_sync(self, message: Any):
        """Non-async broadcast for use from sync contexts."""
        for q in self._subscribers:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass


# ── ANSI helpers ──

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\x1b\[\??\d+[hl]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"

def sparkline(values: list[float], width: int = 30) -> str:
    if not values:
        return ""
    mx = max(values) if max(values) > 0 else 1
    return "".join(SPARKLINE_CHARS[min(int(v / mx * 8), 8)] for v in values[-width:])


@dataclass
class TPSTracker:
    """Rolling token-per-second tracker with 5-second window."""
    _samples: list[tuple[float, int]] = field(default_factory=list)
    _last_tps: float = 0.0
    _last_tps_at: float = 0.0
    _history: list[float] = field(default_factory=list)  # 10-minute graph at 1s buckets

    def record(self, timestamp: float, total_tokens: int):
        self._samples.append((timestamp, total_tokens))
        # prune older than 5s
        cutoff = timestamp - 5.0
        self._samples = [(t, n) for t, n in self._samples if t >= cutoff]

    def tps(self, now: float) -> float:
        # throttle to once per second
        if now - self._last_tps_at < 1.0:
            return self._last_tps
        if len(self._samples) < 2:
            return 0.0
        dt = self._samples[-1][0] - self._samples[0][0]
        if dt <= 0:
            return 0.0
        dn = self._samples[-1][1] - self._samples[0][1]
        self._last_tps = dn / dt
        self._last_tps_at = now
        self._history.append(self._last_tps)
        # keep 10 minutes at 1s resolution
        if len(self._history) > 600:
            self._history = self._history[-600:]
        return self._last_tps


# ── Event humanization ──

EVENT_MAP = {
    "turn/completed": "turn completed",
    "turn/failed": "turn failed",
    "turn/cancelled": "turn cancelled",
    "turn/input_required": "input required",
    "item/commandExecution/requestApproval": "exec approval requested",
    "execCommandApproval": "auto-approved",
    "applyPatchApproval": "auto-approved",
    "item/fileChange/requestApproval": "file change approval requested",
    "item/tool/call": "tool call",
    "item/tool/requestUserInput": "auto-answered",
    "tool_call_completed": "tool call completed",
    "tool_call_failed": "tool call failed",
    "tool_call_unsupported": "unsupported tool call",
    "item/message/delta": "streaming response",
    "codex/event/reasoning": "reasoning",
    "codex/event/token_count": "token update",
    "thread/tokenUsage/updated": "token usage updated",
}


def humanize_event(event_type: str, payload: Optional[dict] = None) -> str:
    base = EVENT_MAP.get(event_type, event_type)
    if payload:
        if event_type == "item/message/delta" and "content" in payload:
            snippet = strip_ansi(str(payload["content"]))[:60]
            return f"{base}: {snippet}"
        if event_type == "codex/event/reasoning" and "summary" in payload:
            return f"{base}: {strip_ansi(str(payload['summary']))[:60]}"
        if "command" in payload:
            return f"{base}: {strip_ansi(str(payload['command']))[:60]}"
        if event_type.endswith("completed") and "status" in payload:
            return f"{base} ({payload['status']})"
    return base


def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")
