"""E2E-002: End-to-end orchestrator test with real GitHub issue and Gemma 4.

Creates a real GitHub issue in gopinaath/symph-test-10, seeds it into a
MemoryTracker, starts the Orchestrator with a mock agent runner that calls
the vLLM Gemma 4 endpoint, and verifies the full lifecycle:
  dispatch -> agent runs (calls Gemma 4) -> completion.

Requires:
  - vLLM running at localhost:8002 (or VLLM_BASE_URL)
  - gh CLI authenticated with access to gopinaath/symph-test-10

Skip with: pytest -m "not e2e"
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import json
import logging
from datetime import datetime, timezone

import httpx
import pytest

from symphony.config import AgentConfig, CodexConfig, Config, PollingConfig
from symphony.models import Issue
from symphony.orchestrator import AgentResult, Orchestrator
from symphony.tracker.memory import MemoryTracker

logger = logging.getLogger(__name__)

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8002/v1")
MODEL_ID = "google/gemma-4-31B-it"
REPO = "gopinaath/symph-test-10"

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Connectivity checks
# ---------------------------------------------------------------------------


def _have_vllm() -> bool:
    """Check if vLLM endpoint is reachable."""
    try:
        r = httpx.get(f"{VLLM_BASE_URL}/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _have_gh() -> bool:
    """Check if gh CLI is available and authenticated."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


skip_no_vllm = pytest.mark.skipif(
    not _have_vllm(),
    reason=f"vLLM not reachable at {VLLM_BASE_URL}",
)

skip_no_gh = pytest.mark.skipif(
    not _have_gh(),
    reason="gh CLI not available or not authenticated",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_github_issue(title: str, body: str) -> dict:
    """Create a real GitHub issue using gh CLI. Returns parsed JSON.

    ``gh issue create`` returns a URL, so we parse the issue number from it
    and then use ``gh issue view --json`` to get structured data.
    """
    create_result = subprocess.run(
        [
            "gh", "issue", "create",
            "--repo", REPO,
            "--title", title,
            "--body", body,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if create_result.returncode != 0:
        raise RuntimeError(f"Failed to create GitHub issue: {create_result.stderr}")

    # Output is a URL like https://github.com/owner/repo/issues/42
    url = create_result.stdout.strip()
    number_str = url.rstrip("/").rsplit("/", 1)[-1]
    if not number_str.isdigit():
        raise RuntimeError(f"Could not parse issue number from URL: {url}")

    # Fetch structured data
    view_result = subprocess.run(
        [
            "gh", "issue", "view", number_str,
            "--repo", REPO,
            "--json", "number,url,title,body,state,createdAt",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if view_result.returncode != 0:
        raise RuntimeError(f"Failed to view GitHub issue: {view_result.stderr}")
    return json.loads(view_result.stdout)


def _close_github_issue(number: int) -> None:
    """Close a GitHub issue using gh CLI."""
    subprocess.run(
        ["gh", "issue", "close", str(number), "--repo", REPO],
        capture_output=True,
        text=True,
        timeout=15,
    )


def _make_config() -> Config:
    """Build a minimal Config suitable for the e2e test."""
    return Config(
        polling=PollingConfig(interval_ms=500),
        agent=AgentConfig(max_concurrent_agents=1),
        codex=CodexConfig(stall_timeout_ms=60_000),
    )


def _issue_from_gh(gh_data: dict) -> Issue:
    """Build a symphony Issue from gh CLI JSON output."""
    number = gh_data["number"]
    return Issue(
        id=str(number),
        identifier=f"#{number}",
        title=gh_data["title"],
        description=gh_data.get("body", ""),
        priority=2,
        state="Todo",
        branch_name=f"e2e-test-{number}",
        url=gh_data["url"],
        assignee_id=None,
        created_at=datetime.now(timezone.utc),
    )


class StubWorkspace:
    """Minimal workspace stub -- no filesystem side-effects."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.removed: list[str] = []

    async def create(self, identifier: str) -> str:
        self.created.append(identifier)
        return f"/tmp/ws/{identifier}"

    async def remove(self, identifier: str) -> None:
        self.removed.append(identifier)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@skip_no_vllm
@skip_no_gh
class TestOrchestratorE2E:
    """E2E-002: Full orchestrator lifecycle with real GitHub issue and Gemma 4."""

    async def test_full_lifecycle_with_gemma4(self) -> None:
        """Create a GitHub issue, dispatch via orchestrator, call Gemma 4,
        verify completion."""
        issue_number: int | None = None

        try:
            # 1. Create a real GitHub issue
            gh_data = _create_github_issue(
                title="[E2E Test] Orchestrator lifecycle test",
                body=(
                    "Automated E2E test for Symphony orchestrator.\n\n"
                    "Task: Write a Python function `greet(name)` that returns "
                    "'Hello, {name}!'.\n\n"
                    "This issue will be closed automatically after the test."
                ),
            )
            issue_number = gh_data["number"]
            logger.info("Created GitHub issue #%d: %s", issue_number, gh_data["url"])

            # 2. Build a symphony Issue and seed it into MemoryTracker
            issue = _issue_from_gh(gh_data)
            tracker = MemoryTracker(
                issues=[issue],
                active_states={"InProgress", "in_progress"},
                terminal_states={"Done", "Cancelled", "Canceled"},
                candidate_states={"Todo", "InProgress", "in_progress"},
            )

            # 3. Build agent runner that calls Gemma 4
            agent_called = asyncio.Event()
            gemma_response: dict = {}

            async def gemma4_agent_runner(
                run_issue: Issue,
                workspace_path: str | None,
                worker_host: str | None,
            ) -> AgentResult:
                """Agent runner that calls Gemma 4 via vLLM and marks done."""
                prompt = (
                    f"You are an autonomous software engineer. "
                    f"Implement the following issue:\n\n"
                    f"## {run_issue.title}\n\n"
                    f"{run_issue.description}\n\n"
                    f"Respond with ONLY the Python code."
                )

                async with httpx.AsyncClient(timeout=45) as client:
                    resp = await client.post(
                        f"{VLLM_BASE_URL}/chat/completions",
                        json={
                            "model": MODEL_ID,
                            "messages": [
                                {"role": "user", "content": prompt},
                            ],
                            "max_tokens": 256,
                            "temperature": 0.2,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                gemma_response["content"] = content
                gemma_response["usage"] = usage

                logger.info(
                    "Gemma 4 response (%d tokens): %s",
                    usage.get("completion_tokens", 0),
                    content[:120],
                )

                # Mark the issue as done in the tracker so the orchestrator
                # sees it as terminal and won't re-dispatch.
                await tracker.update_issue_state(run_issue.identifier, "Done")

                agent_called.set()

                return AgentResult(
                    session_id=f"e2e-{issue_number}",
                    turn_count=1,
                    last_event="task_complete",
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )

            # 4. Create and start the orchestrator
            ws = StubWorkspace()
            config = _make_config()
            orch = Orchestrator(config, tracker, ws, gemma4_agent_runner)

            await orch.start()

            try:
                # 5. Wait for the agent to be called (with timeout)
                await asyncio.wait_for(agent_called.wait(), timeout=50.0)

                # Give the orchestrator a moment to process the result
                await asyncio.sleep(0.3)

                # 6. Assertions
                # a. The agent was actually called
                assert agent_called.is_set(), "Agent runner was never called"

                # b. Gemma 4 produced a response
                assert "content" in gemma_response, "No response from Gemma 4"
                content = gemma_response["content"]
                assert len(content) > 0, "Empty response from Gemma 4"
                # The response should contain a function definition
                assert "def " in content, (
                    f"Expected a function definition in response, got: {content[:200]}"
                )

                # c. Tokens were tracked
                assert gemma_response["usage"].get("total_tokens", 0) > 0, (
                    "Expected non-zero token usage"
                )

                # d. The issue moved to completed or is no longer running
                snap = orch.snapshot()
                assert issue.identifier not in snap.running, (
                    f"Issue {issue.identifier} should not still be running"
                )

                # e. Token totals accumulated in the orchestrator
                assert snap.codex_totals["total_tokens"] > 0, (
                    "Orchestrator should have accumulated token totals"
                )

                # f. Workspace was created for the issue
                assert issue.identifier in ws.created, (
                    "Workspace should have been created for the issue"
                )

                # g. The tracker shows the issue as Done
                states = await tracker.fetch_issue_states_by_ids([issue.identifier])
                assert states[issue.identifier] == "Done", (
                    f"Expected issue state 'Done', got {states[issue.identifier]}"
                )

            finally:
                await orch.stop()

        finally:
            # 7. Cleanup: close the GitHub issue
            if issue_number is not None:
                _close_github_issue(issue_number)
                logger.info("Closed GitHub issue #%d", issue_number)
