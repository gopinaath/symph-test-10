"""Tests for the terminal status dashboard renderer.

Ported from the Elixir ``status_dashboard_snapshot_test.exs`` test suite.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from symphony.models import Issue
from symphony.observability import (
    EVENT_MAP,
    TPSTracker,
    humanize_event,
    sparkline,
    strip_ansi,
    format_timestamp,
)
from symphony.orchestrator import OrchestratorSnapshot, RunningEntry, RetryEntry
from symphony.status_dashboard import StatusDashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_issue(
    identifier: str = "PROJ-1",
    title: str = "Fix the bug",
    state: str = "in progress",
    **kwargs,
) -> Issue:
    defaults = dict(
        id="issue-1",
        identifier=identifier,
        title=title,
        description="",
        priority=2,
        state=state,
        branch_name="fix-the-bug",
        url=f"https://github.com/org/repo/issues/1",
        assignee_id=None,
    )
    defaults.update(kwargs)
    return Issue(**defaults)


def _make_running_entry(
    identifier: str = "PROJ-1",
    state: str = "in progress",
    session_id: str = "sess-abc",
    turn_count: int = 3,
    total_tokens: int = 12500,
    last_event: str = "turn completed",
    started_at: datetime | None = None,
    worker_host: str | None = "gpu-box-1",
    **kwargs,
) -> RunningEntry:
    issue = _make_issue(identifier=identifier, state=state)
    if started_at is None:
        started_at = datetime(2026, 4, 9, 10, 30, 0, tzinfo=timezone.utc)
    task = MagicMock(spec=asyncio.Task)
    return RunningEntry(
        issue=issue,
        task=task,
        started_at=started_at,
        session_id=session_id,
        turn_count=turn_count,
        total_tokens=total_tokens,
        last_event=last_event,
        last_event_at=started_at + timedelta(minutes=2),
        worker_host=worker_host,
        **kwargs,
    )


def _make_retry_entry(
    identifier: str = "PROJ-2",
    attempt: int = 2,
    due_in_seconds: int = 30,
    error: str = "stall_timeout",
    preferred_host: str | None = "gpu-box-2",
) -> RetryEntry:
    issue = _make_issue(identifier=identifier, state="in progress")
    return RetryEntry(
        issue=issue,
        attempt=attempt,
        due_at=datetime.now(timezone.utc) + timedelta(seconds=due_in_seconds),
        error=error,
        preferred_host=preferred_host,
    )


def _empty_snapshot(**overrides) -> OrchestratorSnapshot:
    defaults = dict(
        running={},
        retry_queue={},
        completed=set(),
        codex_totals={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "seconds_running": 0.0},
        codex_rate_limits=None,
        poll_countdown_ms=25000,
        poll_checking=False,
    )
    defaults.update(overrides)
    return OrchestratorSnapshot(**defaults)


def _strip(text: str) -> str:
    """Strip ANSI escapes for assertion matching."""
    return strip_ansi(text)


# ===========================================================================
# Tests
# ===========================================================================


class TestOfflineRendering:
    """Renders offline marker."""

    def test_renders_offline_marker(self):
        dash = StatusDashboard(terminal_width=80)
        output = dash.render_offline()
        plain = _strip(output)
        assert "offline" in plain
        assert "Symphony" in plain

    def test_offline_increments_render_count(self):
        dash = StatusDashboard(terminal_width=80)
        assert dash.render_count == 0
        dash.render_offline()
        assert dash.render_count == 1


class TestHeaderRendering:
    """Renders project link in header."""

    def test_renders_project_url_in_header(self):
        dash = StatusDashboard(
            project_url="https://github.com/org/repo",
            terminal_width=120,
        )
        output = dash.render(_empty_snapshot())
        plain = _strip(output)
        assert "https://github.com/org/repo" in plain

    def test_renders_symphony_title(self):
        dash = StatusDashboard(terminal_width=80)
        output = dash.render(_empty_snapshot())
        plain = _strip(output)
        assert "Symphony" in plain

    def test_renders_dashboard_url_when_server_port_configured(self):
        dash = StatusDashboard(
            server_host="127.0.0.1",
            server_port=4000,
            terminal_width=120,
        )
        output = dash.render(_empty_snapshot())
        plain = _strip(output)
        assert "http://127.0.0.1:4000" in plain

    def test_no_dashboard_url_without_port(self):
        dash = StatusDashboard(
            server_host="127.0.0.1",
            terminal_width=120,
        )
        output = dash.render(_empty_snapshot())
        plain = _strip(output)
        assert "dashboard:" not in plain

    def test_prefers_bound_port_over_server_port(self):
        dash = StatusDashboard(
            server_host="127.0.0.1",
            server_port=4000,
            bound_port=4001,
            terminal_width=120,
        )
        output = dash.render(_empty_snapshot())
        plain = _strip(output)
        assert "4001" in plain
        assert "4000" not in plain

    def test_normalizes_wildcard_host_0000(self):
        dash = StatusDashboard(
            server_host="0.0.0.0",
            server_port=4000,
            terminal_width=120,
        )
        output = dash.render(_empty_snapshot())
        plain = _strip(output)
        assert "http://127.0.0.1:4000" in plain
        assert "0.0.0.0" not in plain

    def test_normalizes_wildcard_host_ipv6(self):
        dash = StatusDashboard(
            server_host="::",
            server_port=5000,
            terminal_width=120,
        )
        output = dash.render(_empty_snapshot())
        plain = _strip(output)
        assert "http://127.0.0.1:5000" in plain


class TestPollStatus:
    """Renders next refresh countdown and checking marker."""

    def test_renders_countdown_seconds(self):
        snap = _empty_snapshot(poll_countdown_ms=15000, poll_checking=False)
        dash = StatusDashboard(terminal_width=120)
        output = dash.render(snap)
        plain = _strip(output)
        assert "next refresh in 15s" in plain

    def test_renders_checking_now(self):
        snap = _empty_snapshot(poll_countdown_ms=0, poll_checking=True)
        dash = StatusDashboard(terminal_width=120)
        output = dash.render(snap)
        plain = _strip(output)
        assert "checking now..." in plain


class TestBackoffQueueSpacer:
    """Adds spacer before backoff queue (with and without active agents)."""

    def test_spacer_before_backoff_with_running_agents(self):
        running = {"PROJ-1": _make_running_entry("PROJ-1")}
        retry = {"PROJ-2": _make_retry_entry("PROJ-2")}
        snap = _empty_snapshot(running=running, retry_queue=retry)
        dash = StatusDashboard(terminal_width=120)
        output = dash.render(snap)
        lines = output.split("\n")
        # Find the backoff queue section
        backoff_idx = None
        for i, line in enumerate(lines):
            if "BACKOFF QUEUE" in _strip(line):
                backoff_idx = i
                break
        assert backoff_idx is not None
        # There should be an empty spacer line just before the backoff header
        assert lines[backoff_idx - 1].strip() == ""

    def test_spacer_before_backoff_without_running_agents(self):
        retry = {"PROJ-2": _make_retry_entry("PROJ-2")}
        snap = _empty_snapshot(retry_queue=retry)
        dash = StatusDashboard(terminal_width=120)
        output = dash.render(snap)
        lines = output.split("\n")
        backoff_idx = None
        for i, line in enumerate(lines):
            if "BACKOFF QUEUE" in _strip(line):
                backoff_idx = i
                break
        assert backoff_idx is not None
        assert lines[backoff_idx - 1].strip() == ""


class TestCoalescedRendering:
    """Coalesces rapid updates to one render per interval."""

    def test_first_call_always_renders(self):
        dash = StatusDashboard(render_interval_ms=100, terminal_width=80)
        snap = _empty_snapshot()
        result = dash.maybe_render(snap)
        assert result is not None
        assert dash.render_count == 1

    def test_rapid_calls_are_coalesced(self):
        dash = StatusDashboard(render_interval_ms=500, terminal_width=80)
        snap = _empty_snapshot()
        r1 = dash.maybe_render(snap)
        assert r1 is not None  # first always passes
        r2 = dash.maybe_render(snap)
        assert r2 is None  # too soon, coalesced
        r3 = dash.maybe_render(snap)
        assert r3 is None
        assert dash.render_count == 1  # only one real render happened

    def test_render_after_interval_elapsed(self):
        dash = StatusDashboard(render_interval_ms=10, terminal_width=80)
        snap = _empty_snapshot()
        r1 = dash.maybe_render(snap)
        assert r1 is not None
        # Wait enough for the interval to elapse
        time.sleep(0.02)
        r2 = dash.maybe_render(snap)
        assert r2 is not None
        assert dash.render_count == 2

    def test_flush_pending_emits_coalesced_update(self):
        dash = StatusDashboard(render_interval_ms=1000, terminal_width=80)
        snap = _empty_snapshot()
        r1 = dash.maybe_render(snap)
        assert r1 is not None
        r2 = dash.maybe_render(snap)
        assert r2 is None  # coalesced
        # flush should emit
        r3 = dash.flush_pending(snap)
        assert r3 is not None
        assert dash.render_count == 2

    def test_flush_pending_noop_when_no_pending(self):
        dash = StatusDashboard(render_interval_ms=1000, terminal_width=80)
        snap = _empty_snapshot()
        result = dash.flush_pending(snap)
        assert result is None


class TestRollingTPS:
    """Computes rolling 5-second TPS."""

    def test_tps_with_steady_input(self):
        tracker = TPSTracker()
        base = 1000.0
        for i in range(6):
            tracker.record(base + i, i * 500)
        tps = tracker.tps(base + 6.0)
        # 2500 tokens over 5 seconds = 500 TPS
        assert tps > 0

    def test_tps_prunes_old_samples(self):
        tracker = TPSTracker()
        base = 1000.0
        # Record samples spread over 10 seconds
        for i in range(11):
            tracker.record(base + i, i * 100)
        tps = tracker.tps(base + 11.0)
        # Only samples within 5s window should be used
        assert tps > 0
        # All samples older than 5s should be pruned
        oldest = min(t for t, _ in tracker._samples)
        assert oldest >= base + 6.0


class TestFormatTimestamps:
    """Formats timestamps at second precision."""

    def test_format_to_hms(self):
        dt = datetime(2026, 4, 9, 14, 30, 45, tzinfo=timezone.utc)
        assert format_timestamp(dt) == "14:30:45"

    def test_microseconds_are_dropped(self):
        dt = datetime(2026, 4, 9, 14, 30, 45, 123456, tzinfo=timezone.utc)
        result = format_timestamp(dt)
        assert result == "14:30:45"
        assert "." not in result


class TestTPSGraph:
    """Renders 10-minute TPS graph for steady and ramping throughput."""

    def test_steady_throughput_graph(self):
        tracker = TPSTracker()
        base = 1000.0
        # Simulate steady 100 TPS for 10 samples
        for i in range(11):
            tracker.record(base + i, i * 100)
            if i > 0:
                tracker.tps(base + i + 0.5)
        graph = sparkline(tracker._history, width=30)
        assert len(graph) > 0
        # Steady throughput: all bars should be the same (max)
        assert all(c == graph[0] for c in graph)

    def test_ramping_throughput_graph(self):
        tracker = TPSTracker()
        base = 1000.0
        total = 0
        for i in range(11):
            total += i * 50  # increasing token counts
            tracker.record(base + i, total)
            if i > 0:
                tracker.tps(base + i + 0.5)
        graph = sparkline(tracker._history, width=30)
        assert len(graph) > 0
        # Ramping: last bar should be max
        assert graph[-1] in ("\u2587", "\u2588")  # ▇ or █


class TestStripAnsiFromMessages:
    """Strips ANSI/control bytes from last codex message."""

    def test_strips_ansi_from_event(self):
        raw = "\x1b[32mSuccess\x1b[0m"
        assert strip_ansi(raw) == "Success"

    def test_strips_control_bytes(self):
        raw = "Line1\x00\x01\x02Line2"
        assert strip_ansi(raw) == "Line1Line2"

    def test_strips_cursor_control(self):
        raw = "\x1b[?25lHidden cursor\x1b[?25h"
        assert strip_ansi(raw) == "Hidden cursor"


class TestTerminalWidthExpansion:
    """Expands running row to requested terminal width."""

    def test_row_reaches_terminal_width(self):
        running = {"PROJ-1": _make_running_entry("PROJ-1")}
        snap = _empty_snapshot(running=running)
        for target_width in (80, 120, 200):
            dash = StatusDashboard(terminal_width=target_width)
            output = dash.render(snap)
            for line in output.split("\n"):
                # Every line with visible content should be padded to at
                # least terminal_width visible characters (unless it is a
                # blank spacer line).
                plain = _strip(line)
                if plain.strip():
                    assert len(plain) >= target_width, (
                        f"Line too short ({len(plain)} < {target_width}): {plain!r}"
                    )

    def test_fallback_width_is_120(self):
        dash = StatusDashboard()
        # When no terminal_width set and os.get_terminal_size raises
        dash.terminal_width = None
        # We test the default by setting a known value
        dash.terminal_width = None
        # The fallback happens inside get_terminal_width; we can test the
        # method directly by monkey-patching os.get_terminal_size
        import os
        original = os.get_terminal_size
        try:
            os.get_terminal_size = lambda *a, **k: (_ for _ in ()).throw(OSError)
            assert dash.get_terminal_width() == 120
        finally:
            os.get_terminal_size = original


class TestHumanizeCodexEvents:
    """Humanizes full codex event set (17 types)."""

    def test_all_17_event_types_mapped(self):
        expected = {
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
        assert len(expected) == 17, "expected exactly 17 event types"
        for event_type, expected_text in expected.items():
            result = humanize_event(event_type)
            assert result == expected_text, (
                f"humanize_event({event_type!r}) = {result!r}, "
                f"expected {expected_text!r}"
            )

    def test_unknown_event_passthrough(self):
        assert humanize_event("custom/unknown/event") == "custom/unknown/event"


class TestDynamicToolWrapperEvents:
    """Humanizes dynamic tool wrapper events."""

    def test_tool_call_completed(self):
        result = humanize_event("tool_call_completed")
        assert "completed" in result

    def test_tool_call_failed(self):
        result = humanize_event("tool_call_failed")
        assert "failed" in result

    def test_tool_call_unsupported(self):
        result = humanize_event("tool_call_unsupported")
        assert "unsupported" in result

    def test_tool_call_completed_with_status_payload(self):
        result = humanize_event("tool_call_completed", {"status": "success"})
        assert "success" in result


class TestShellCommandAsExecStatus:
    """Uses shell command line as exec status text."""

    def test_exec_approval_with_command(self):
        result = humanize_event(
            "execCommandApproval",
            {"command": "git diff --stat HEAD~1"},
        )
        assert "git diff --stat HEAD~1" in result

    def test_command_strips_ansi(self):
        result = humanize_event(
            "execCommandApproval",
            {"command": "\x1b[1mgit status\x1b[0m"},
        )
        assert "\x1b" not in result
        assert "git status" in result


class TestAutoApprovalAndAutoAnswered:
    """Formats auto-approval and auto-answered updates."""

    def test_exec_command_approval(self):
        result = humanize_event("execCommandApproval")
        assert result == "auto-approved"

    def test_apply_patch_approval(self):
        result = humanize_event("applyPatchApproval")
        assert result == "auto-approved"

    def test_auto_answered(self):
        result = humanize_event("item/tool/requestUserInput")
        assert result == "auto-answered"

    def test_auto_approval_with_command_payload(self):
        result = humanize_event("execCommandApproval", {"command": "npm test"})
        assert "npm test" in result
        assert "auto-approved" in result

    def test_auto_answered_with_content(self):
        # auto-answered should still return base text without content key
        result = humanize_event("item/tool/requestUserInput", {"tool": "readline"})
        assert result == "auto-answered"


class TestAgentTable:
    """Renders agent table rows properly."""

    def test_renders_running_agent_row(self):
        entry = _make_running_entry(
            identifier="PROJ-42",
            state="in progress",
            session_id="sess-xyz",
            turn_count=5,
            total_tokens=25000,
            last_event="turn completed",
            worker_host="gpu-box-3",
        )
        snap = _empty_snapshot(running={"PROJ-42": entry})
        dash = StatusDashboard(terminal_width=120)
        output = dash.render(snap)
        plain = _strip(output)
        assert "PROJ-42" in plain
        assert "25000" in plain
        assert "gpu-box-3" in plain

    def test_renders_local_when_no_host(self):
        entry = _make_running_entry(identifier="PROJ-1", worker_host=None)
        snap = _empty_snapshot(running={"PROJ-1": entry})
        dash = StatusDashboard(terminal_width=120)
        output = dash.render(snap)
        plain = _strip(output)
        assert "local" in plain


class TestRetryQueueTable:
    """Renders retry queue rows."""

    def test_renders_retry_entry(self):
        entry = _make_retry_entry(
            identifier="PROJ-7",
            attempt=3,
            due_in_seconds=45,
            error="rate_limited",
            preferred_host="gpu-box-1",
        )
        snap = _empty_snapshot(retry_queue={"PROJ-7": entry})
        dash = StatusDashboard(terminal_width=120)
        output = dash.render(snap)
        plain = _strip(output)
        assert "PROJ-7" in plain
        assert "rate_limited" in plain
        assert "gpu-box-1" in plain


class TestRateLimits:
    """Renders rate limits when present."""

    def test_renders_rate_limits(self):
        limits = {"requests_remaining": 42, "tokens_remaining": 10000}
        snap = _empty_snapshot(codex_rate_limits=limits)
        dash = StatusDashboard(terminal_width=120)
        output = dash.render(snap)
        plain = _strip(output)
        assert "Rate limits" in plain
        assert "requests_remaining=42" in plain

    def test_no_rate_limits_line_when_none(self):
        snap = _empty_snapshot(codex_rate_limits=None)
        dash = StatusDashboard(terminal_width=120)
        output = dash.render(snap)
        plain = _strip(output)
        assert "Rate limits" not in plain


class TestTokenRecording:
    """record_tokens feeds the internal TPS tracker."""

    def test_record_tokens_updates_tps(self):
        dash = StatusDashboard(terminal_width=80)
        base = time.monotonic()
        for i in range(6):
            dash.record_tokens(base + i, i * 1000)
        tps = dash._tps_tracker.tps(base + 6.0)
        assert tps > 0


class TestBoxDrawing:
    """Uses box drawing characters for borders."""

    def test_contains_box_characters(self):
        snap = _empty_snapshot()
        dash = StatusDashboard(terminal_width=80)
        output = dash.render(snap)
        # Should contain box-drawing characters
        assert "\u2500" in output  # ─
        assert "\u2502" in output  # │
        assert "\u250c" in output  # ┌
        assert "\u2518" in output  # ┘


class TestMessageDeltaRendering:
    """Streaming message delta includes snippet."""

    def test_message_delta_with_content(self):
        result = humanize_event(
            "item/message/delta",
            {"content": "Implementing the fix now..."},
        )
        assert "Implementing the fix now" in result

    def test_reasoning_with_summary(self):
        result = humanize_event(
            "codex/event/reasoning",
            {"summary": "Analyzing test failures"},
        )
        assert "Analyzing test failures" in result
