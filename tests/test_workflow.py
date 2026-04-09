"""Tests for symphony.workflow — WorkflowStore reload and error handling."""

from __future__ import annotations

import time

import pytest

from symphony.workflow import Workflow, WorkflowStore

# ---------------------------------------------------------------------------
# Basic Workflow.parse
# ---------------------------------------------------------------------------


class TestWorkflowParse:
    def test_prompt_only_no_front_matter(self, tmp_path) -> None:
        p = tmp_path / "WORKFLOW.md"
        p.write_text("prompt only content")
        wf = Workflow.parse(str(p))
        assert wf.prompt_template == "prompt only content"
        # Config is all defaults.
        assert wf.config.polling.interval_ms == 30000

    def test_front_matter_and_prompt(self, tmp_path) -> None:
        p = tmp_path / "WORKFLOW.md"
        p.write_text(
            "---\npolling:\n  interval_ms: 5000\n---\nHello {{ issue.title }}"
        )
        wf = Workflow.parse(str(p))
        assert wf.config.polling.interval_ms == 5000
        assert "Hello" in wf.prompt_template


# ---------------------------------------------------------------------------
# WorkflowStore
# ---------------------------------------------------------------------------


class TestWorkflowStoreReload:
    def test_init_and_reload_on_change(self, tmp_path) -> None:
        p = tmp_path / "WORKFLOW.md"
        p.write_text("---\npolling:\n  interval_ms: 1000\n---\nV1 prompt")

        store = WorkflowStore(path=str(p), poll_interval=0.1)
        wf = store.init()
        assert wf.config.polling.interval_ms == 1000
        assert "V1" in wf.prompt_template

        # Mutate the file.
        time.sleep(0.05)  # ensure mtime changes
        p.write_text("---\npolling:\n  interval_ms: 2000\n---\nV2 prompt")

        store.start_polling()
        try:
            # Wait for at most 2 seconds for the store to pick up the change.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                wf2 = store.workflow
                if wf2 and wf2.config.polling.interval_ms == 2000:
                    break
                time.sleep(0.05)
            else:
                pytest.fail("WorkflowStore did not pick up file change in time")

            assert wf2 is not None
            assert "V2" in wf2.prompt_template
        finally:
            store.stop_polling()

    def test_keeps_last_good_on_bad_reload(self, tmp_path) -> None:
        p = tmp_path / "WORKFLOW.md"
        p.write_text("---\npolling:\n  interval_ms: 1000\n---\nGood prompt")

        store = WorkflowStore(path=str(p), poll_interval=0.1)
        store.init()

        good_wf = store.workflow
        assert good_wf is not None

        # Write bad content (non-map YAML).
        time.sleep(0.05)
        p.write_text("---\n- bad\n- yaml\n---\nstill here")

        store.start_polling()
        try:
            time.sleep(0.5)
            # The store should still serve the last good workflow.
            assert store.workflow is good_wf
            assert store.last_error is not None
        finally:
            store.stop_polling()


class TestWorkflowStoreInitFailure:
    def test_init_stops_on_missing_file(self, tmp_path) -> None:
        store = WorkflowStore(path=str(tmp_path / "MISSING.md"))
        with pytest.raises(FileNotFoundError):
            store.init()

    def test_init_stops_on_invalid_front_matter(self, tmp_path) -> None:
        p = tmp_path / "WORKFLOW.md"
        p.write_text("---\n- not a map\n---\n")
        store = WorkflowStore(path=str(p))
        with pytest.raises(ValueError, match="mapping"):
            store.init()
