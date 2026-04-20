"""Live demo: Run Symphony orchestrator against real GitHub issues with Gemma 4.

Usage:
    python run_live_demo.py

Requires:
    - vLLM tunnel on localhost:8002
    - GITHUB_TOKEN env var set
    - Issues with "todo" label in gopinaath/symph-test-10
"""

import asyncio
import os
import sys
import time

import httpx

# Ensure symphony is importable
sys.path.insert(0, os.path.dirname(__file__))

from symphony.config import Config
from symphony.models import Issue
from symphony.orchestrator import AgentResult, Orchestrator
from symphony.tracker.memory import MemoryTracker
from symphony.workspace import Workspace, WorkspaceConfig

VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8002/v1")
MODEL_ID = "google/gemma-4-31B-it"
REPO = "gopinaath/symph-test-10"


tracker = None  # set later before orchestrator start

async def gemma4_agent(issue: object, workspace_path: str | None, worker_host: str | None) -> AgentResult:
    """Agent that calls Gemma 4 to implement the issue."""
    iss = issue  # type: Issue
    print(f"\n{'='*60}")
    print(f"  AGENT START: {iss.identifier} — {iss.title}")
    print(f"{'='*60}")

    prompt = f"""You are an autonomous software engineer. Implement the following GitHub issue.

## Issue #{iss.identifier}: {iss.title}

{iss.description}

## Instructions
- Create the files described in the issue
- Write clean, working Python code
- Include all specified test cases
- Output ONLY the file contents in markdown code blocks with filenames

Example format:
```python
# filename: lib/example.py
def example():
    pass
```
"""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{VLLM_URL}/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.2,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    print(f"\n  AGENT RESPONSE ({usage.get('completion_tokens', '?')} tokens):")
    print(f"  {content[:200]}...")

    # Write files to workspace if we have one
    if workspace_path:
        _write_code_blocks(workspace_path, content)

    # Mark issue as done in the tracker
    await tracker.update_issue_state(iss.identifier, "done")

    print(f"\n  AGENT DONE: {iss.identifier} → state=done")
    print(f"{'='*60}\n")

    return AgentResult(
        session_id=f"demo-{iss.identifier}",
        turn_count=1,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        validation_passed=True,
    )


def _write_code_blocks(workspace_path: str, content: str):
    """Extract and write code blocks from LLM response."""
    import re
    # Match ```python\n# filename: path\n...``` blocks
    pattern = r"```python\s*\n#\s*filename:\s*(.+?)\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)

    for filename, code in matches:
        filepath = os.path.join(workspace_path, filename.strip())
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(code.strip() + "\n")
        print(f"  Wrote: {filepath}")


async def main():
    print("Symphony Live Demo")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Repo: {REPO}")
    print(f"vLLM: {VLLM_URL}")
    print()

    # Fetch real issues from GitHub
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Authorization": f"token {token}"} if token else {}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"https://api.github.com/repos/{REPO}/issues",
            params={"labels": "todo", "state": "open"},
            headers=headers,
        )
        if resp.status_code != 200:
            print(f"Failed to fetch issues: {resp.status_code}")
            # Fallback: use hardcoded issues
            issues_data = []
        else:
            issues_data = resp.json()

    if not issues_data:
        print("No 'todo' issues found. Creating from memory...")
        issues_data = [
            {"number": 5, "title": "Add fibonacci function with tests", "body": "Create fibonacci.py and tests"},
            {"number": 6, "title": "Add string utilities module", "body": "Create strings.py and tests"},
            {"number": 7, "title": "Add a CLI argument parser", "body": "Create cli.py and tests"},
        ]

    print(f"Found {len(issues_data)} issues to process")
    print()

    # Create tracker with issues
    global tracker
    tracker = MemoryTracker(
        active_states=["todo", "in progress"],
        terminal_states=["done", "closed"],
    )

    for gh_issue in issues_data:
        issue = Issue(
            id=str(gh_issue["number"]),
            identifier=f"#{gh_issue['number']}",
            title=gh_issue["title"],
            description=gh_issue.get("body", "") or "",
            priority=None,
            state="todo",
            branch_name=f"symphony/{gh_issue['number']}",
            url=gh_issue.get("html_url", ""),
            assignee_id=None,
        )
        tracker.add_issue(issue)
        print(f"  Loaded: {issue.identifier} — {issue.title}")

    print()

    # Create minimal config
    config = Config.from_yaml({
        "tracker": {"kind": "memory"},
        "polling": {"interval_ms": 5000},
        "workspace": {"root": "/tmp/symphony_demo_workspaces"},
        "agent": {"max_concurrent_agents": 3, "max_turns": 1},
        "codex": {"command": "echo demo", "approval_policy": "never"},
    })

    # Create workspace
    ws_config = WorkspaceConfig(root="/tmp/symphony_demo_workspaces")
    workspace = Workspace(ws_config)

    # Create orchestrator
    orchestrator = Orchestrator(
        config=config,
        tracker=tracker,
        workspace=workspace,
        agent_runner_factory=gemma4_agent,
    )

    # Run for up to 60 seconds
    print("Starting orchestrator...")
    print("(will run for up to 60 seconds or until all issues complete)")
    print()

    start = time.time()
    task = asyncio.create_task(orchestrator.start())

    try:
        while time.time() - start < 60:
            await asyncio.sleep(2)
            snapshot = orchestrator.snapshot()
            running = len(snapshot.running)
            completed = len(snapshot.completed)
            retrying = len(snapshot.retry_queue)
            tokens = snapshot.codex_totals.get("total_tokens", 0)
            print(f"  [{time.time()-start:.0f}s] Running: {running} | Completed: {completed} | Retrying: {retrying} | Tokens: {tokens}")

            if completed >= len(issues_data) and running == 0:
                print("\n  All issues completed!")
                break
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    finally:
        await orchestrator.stop()
        task.cancel()

    # Final summary
    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    snapshot = orchestrator.snapshot()
    print(f"  Issues completed: {len(snapshot.completed)}/{len(issues_data)}")
    print(f"  Total tokens: {snapshot.codex_totals.get('total_tokens', 0)}")
    print(f"  Duration: {time.time()-start:.1f}s")

    # Show workspace contents
    ws_root = "/tmp/symphony_demo_workspaces"
    if os.path.exists(ws_root):
        print(f"\n  Workspace files:")
        for root, dirs, files in os.walk(ws_root):
            for f in files:
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                print(f"    {os.path.relpath(path, ws_root)} ({size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
