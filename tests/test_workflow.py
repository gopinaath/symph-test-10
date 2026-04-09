"""Tests for symphony.workflow — WorkflowStore reload and error handling."""

from __future__ import annotations

import asyncio

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
        p.write_text("---\npolling:\n  interval_ms: 5000\n---\nHello {{ issue.title }}")
        wf = Workflow.parse(str(p))
        assert wf.config.polling.interval_ms == 5000
        assert "Hello" in wf.prompt_template


# ---------------------------------------------------------------------------
# WorkflowStore
# ---------------------------------------------------------------------------


class TestWorkflowStoreReload:
    async def test_init_and_reload_on_change(self, tmp_path) -> None:
        p = tmp_path / "WORKFLOW.md"
        p.write_text("---\npolling:\n  interval_ms: 1000\n---\nV1 prompt")

        store = WorkflowStore(path=str(p), poll_interval=0.05)
        wf = await store.init()
        assert wf.config.polling.interval_ms == 1000
        assert "V1" in wf.prompt_template

        # Mutate the file.
        await asyncio.sleep(0.05)  # ensure mtime changes
        p.write_text("---\npolling:\n  interval_ms: 2000\n---\nV2 prompt")

        await store.start()
        try:
            # Wait for at most 2 seconds for the store to pick up the change.
            deadline = asyncio.get_event_loop().time() + 2.0
            while asyncio.get_event_loop().time() < deadline:
                wf2 = await store.get_workflow()
                if wf2 and wf2.config.polling.interval_ms == 2000:
                    break
                await asyncio.sleep(0.02)
            else:
                pytest.fail("WorkflowStore did not pick up file change in time")

            assert wf2 is not None
            assert "V2" in wf2.prompt_template
        finally:
            await store.stop()

    async def test_keeps_last_good_on_bad_reload(self, tmp_path) -> None:
        p = tmp_path / "WORKFLOW.md"
        p.write_text("---\npolling:\n  interval_ms: 1000\n---\nGood prompt")

        store = WorkflowStore(path=str(p), poll_interval=0.05)
        await store.init()

        good_wf = store.workflow
        assert good_wf is not None

        # Write bad content (non-map YAML).
        await asyncio.sleep(0.05)
        p.write_text("---\n- bad\n- yaml\n---\nstill here")

        await store.start()
        try:
            await asyncio.sleep(0.3)
            # The store should still serve the last good workflow.
            assert store.workflow is good_wf
            assert store.last_error is not None
        finally:
            await store.stop()


class TestWorkflowStoreInitFailure:
    async def test_init_stops_on_missing_file(self, tmp_path) -> None:
        store = WorkflowStore(path=str(tmp_path / "MISSING.md"))
        with pytest.raises(FileNotFoundError):
            await store.init()

    async def test_init_stops_on_invalid_front_matter(self, tmp_path) -> None:
        p = tmp_path / "WORKFLOW.md"
        p.write_text("---\n- not a map\n---\n")
        store = WorkflowStore(path=str(p))
        with pytest.raises(ValueError, match="mapping"):
            await store.init()


# ---------------------------------------------------------------------------
# ROB-002: Concurrent-safe caching tests
# ---------------------------------------------------------------------------


class TestWorkflowStoreConcurrency:
    async def test_concurrent_reads_during_reload(self, tmp_path) -> None:
        """Multiple concurrent reads while a reload is in progress all
        return a valid workflow and never see a partially-updated cache."""
        p = tmp_path / "WORKFLOW.md"
        p.write_text("---\npolling:\n  interval_ms: 1000\n---\nV1 prompt")

        store = WorkflowStore(path=str(p), poll_interval=0.05)
        await store.init()

        # Mutate the file so a reload is triggered.
        await asyncio.sleep(0.05)
        p.write_text("---\npolling:\n  interval_ms: 2000\n---\nV2 prompt")

        await store.start()

        results: list[Workflow | None] = []

        async def _reader() -> None:
            for _ in range(20):
                wf = await store.get_workflow()
                results.append(wf)
                await asyncio.sleep(0.01)

        # Launch several concurrent readers.
        readers = [asyncio.create_task(_reader()) for _ in range(5)]
        await asyncio.gather(*readers)

        await store.stop()

        # Every read must return a valid workflow (never None after init).
        for wf in results:
            assert wf is not None
            # interval_ms should be either 1000 (old) or 2000 (new), never
            # something else.
            assert wf.config.polling.interval_ms in (1000, 2000)

    async def test_stop_cancels_polling(self, tmp_path) -> None:
        """Calling stop() cancels the background polling task."""
        p = tmp_path / "WORKFLOW.md"
        p.write_text("---\npolling:\n  interval_ms: 1000\n---\nprompt")

        store = WorkflowStore(path=str(p), poll_interval=0.05)
        await store.init()
        await store.start()

        # Grab a reference to the internal task.
        poll_task = store._poll_task
        assert poll_task is not None
        assert not poll_task.done()

        await store.stop()

        # After stop, the task should be done (cancelled).
        assert poll_task.done()
        assert store._poll_task is None

    async def test_async_lifecycle(self, tmp_path) -> None:
        """Full lifecycle: start -> read -> modify file -> read updated -> stop."""
        p = tmp_path / "WORKFLOW.md"
        p.write_text("---\npolling:\n  interval_ms: 3000\n---\nOriginal")

        store = WorkflowStore(path=str(p), poll_interval=0.05)
        await store.init()

        # Read the initial workflow.
        wf1 = await store.get_workflow()
        assert wf1 is not None
        assert wf1.config.polling.interval_ms == 3000
        assert "Original" in wf1.prompt_template

        await store.start()

        # Modify the file.
        await asyncio.sleep(0.05)
        p.write_text("---\npolling:\n  interval_ms: 4000\n---\nUpdated")

        # Wait for the reload.
        deadline = asyncio.get_event_loop().time() + 2.0
        wf2: Workflow | None = None
        while asyncio.get_event_loop().time() < deadline:
            wf2 = await store.get_workflow()
            if wf2 and wf2.config.polling.interval_ms == 4000:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("Store did not pick up the updated file")

        assert wf2 is not None
        assert "Updated" in wf2.prompt_template

        await store.stop()

        # After stop, the cached workflow is still accessible.
        assert store.workflow is not None
        assert store.workflow.config.polling.interval_ms == 4000
