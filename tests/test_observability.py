"""Tests for the observability module — PubSub, ANSI stripping, TPS tracking, event humanization."""

import asyncio
import pytest
from symphony.observability import (
    PubSub, strip_ansi, sparkline, TPSTracker, humanize_event, format_timestamp,
    EVENT_MAP,
)
from datetime import datetime, timezone


class TestPubSub:
    @pytest.mark.asyncio
    async def test_subscribe_and_broadcast_deliver_updates(self):
        ps = PubSub()
        q = ps.subscribe()
        await ps.broadcast({"type": "update", "data": 42})
        msg = q.get_nowait()
        assert msg == {"type": "update", "data": 42}

    @pytest.mark.asyncio
    async def test_broadcast_is_noop_when_no_subscribers(self):
        ps = PubSub()
        await ps.broadcast("hello")  # should not raise

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self):
        ps = PubSub()
        q = ps.subscribe()
        ps.unsubscribe(q)
        await ps.broadcast("hello")
        assert q.empty()

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        ps = PubSub()
        q1 = ps.subscribe()
        q2 = ps.subscribe()
        await ps.broadcast("hello")
        assert q1.get_nowait() == "hello"
        assert q2.get_nowait() == "hello"

    def test_broadcast_sync(self):
        ps = PubSub()
        q = ps.subscribe()
        ps.broadcast_sync({"data": 1})
        assert q.get_nowait() == {"data": 1}


class TestStripAnsi:
    def test_strips_color_codes(self):
        assert strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_strips_control_bytes(self):
        assert strip_ansi("hello\x00\x01world") == "helloworld"

    def test_passthrough_clean_text(self):
        assert strip_ansi("hello world") == "hello world"


class TestSparkline:
    def test_empty_values(self):
        assert sparkline([]) == ""

    def test_steady_values(self):
        result = sparkline([10, 10, 10, 10])
        assert len(result) == 4
        assert all(c == "█" for c in result)

    def test_ramping_values(self):
        result = sparkline([0, 2, 5, 10])
        assert len(result) == 4
        # first should be lowest, last should be highest
        assert result[0] == " "
        assert result[-1] == "█"


class TestTPSTracker:
    def test_record_and_tps(self):
        t = TPSTracker()
        t.record(100.0, 0)
        t.record(101.0, 1000)
        tps = t.tps(102.0)
        assert tps > 0

    def test_throttles_tps_updates_to_once_per_second(self):
        t = TPSTracker()
        t.record(100.0, 0)
        t.record(101.0, 1000)
        tps1 = t.tps(101.5)
        # same second, should return cached
        tps2 = t.tps(101.8)
        assert tps1 == tps2

    def test_empty_samples_return_zero(self):
        t = TPSTracker()
        assert t.tps(100.0) == 0.0

    def test_keeps_history(self):
        t = TPSTracker()
        t.record(100.0, 0)
        for i in range(1, 10):
            t.record(100.0 + i, i * 100)
            t.tps(100.0 + i + 0.5)
        assert len(t._history) > 0


class TestHumanizeEvent:
    def test_known_event_types(self):
        assert humanize_event("turn/completed") == "turn completed"
        assert humanize_event("turn/failed") == "turn failed"
        assert humanize_event("execCommandApproval") == "auto-approved"
        assert humanize_event("item/tool/requestUserInput") == "auto-answered"

    def test_unknown_event_passthrough(self):
        assert humanize_event("custom/event") == "custom/event"

    def test_with_command_payload(self):
        result = humanize_event("execCommandApproval", {"command": "git status --short"})
        assert "git status --short" in result

    def test_message_delta_with_content(self):
        result = humanize_event("item/message/delta", {"content": "Hello world"})
        assert "Hello world" in result

    def test_completed_with_status(self):
        result = humanize_event("turn/completed", {"status": "completed"})
        assert "completed" in result

    def test_strips_ansi_from_payload(self):
        result = humanize_event("execCommandApproval", {"command": "\x1b[31mgit status\x1b[0m"})
        assert "\x1b" not in result

    def test_dynamic_tool_events(self):
        assert "completed" in humanize_event("tool_call_completed")
        assert "failed" in humanize_event("tool_call_failed")
        assert "unsupported" in humanize_event("tool_call_unsupported")


class TestFormatTimestamp:
    def test_formats_to_seconds(self):
        dt = datetime(2026, 3, 28, 12, 34, 56, tzinfo=timezone.utc)
        assert format_timestamp(dt) == "12:34:56"
