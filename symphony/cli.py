"""CLI entrypoint for Symphony."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from symphony.config import Config
from symphony.tracker.base import Tracker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symphony — autonomous coding orchestrator",
    )
    parser.add_argument(
        "--i-understand-that-this-will-be-running-without-the-usual-guardrails",
        dest="ack",
        action="store_true",
        help="Required acknowledgement flag",
    )
    parser.add_argument(
        "workflow_path",
        nargs="?",
        default=None,
        help="Path to WORKFLOW.md (default: ./WORKFLOW.md)",
    )
    parser.add_argument(
        "--logs-root",
        default=None,
        help="Root directory for log files",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override server port",
    )

    args = parser.parse_args()

    if not args.ack:
        print(
            "ERROR: Symphony runs AI agents autonomously. You must acknowledge this by passing:\n"
            "  --i-understand-that-this-will-be-running-without-the-usual-guardrails\n\n"
            "This flag confirms you understand that Symphony will:\n"
            "  - Poll your issue tracker for work\n"
            "  - Create isolated workspaces\n"
            "  - Run AI coding agents that execute shell commands\n"
            "  - Auto-approve file changes and command executions\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve workflow path
    workflow_path: str = args.workflow_path
    if workflow_path is None:
        workflow_path = os.path.join(os.getcwd(), "WORKFLOW.md")
    else:
        workflow_path = os.path.expanduser(os.path.expandvars(workflow_path))
        workflow_path = os.path.abspath(workflow_path)

    if not os.path.isfile(workflow_path):
        print(f"ERROR: Workflow file not found: {workflow_path}", file=sys.stderr)
        sys.exit(1)

    # Configure logging
    from symphony.log_file import configure_logging

    logs_root: str | None = args.logs_root
    if logs_root:
        logs_root = os.path.expanduser(os.path.expandvars(logs_root))
        logs_root = os.path.abspath(logs_root)

    configure_logging(
        log_file=os.path.join(logs_root, "log", "symphony.log") if logs_root else None,
    )

    # Set workflow path in environment for config to find
    os.environ["SYMPHONY_WORKFLOW_PATH"] = workflow_path

    # Run the orchestrator
    try:
        asyncio.run(_run(workflow_path, port_override=args.port))
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"ERROR: Failed to start: {e}", file=sys.stderr)
        sys.exit(1)


async def _run(workflow_path: str, port_override: int | None = None) -> None:
    """Main async entry point."""
    from symphony.api import create_app
    from symphony.observability import PubSub
    from symphony.orchestrator import AgentResult, Orchestrator
    from symphony.workflow import Workflow
    from symphony.workspace import Workspace, WorkspaceConfig

    # Load workflow
    workflow = Workflow.parse(workflow_path)
    config = workflow.config

    # Override port if specified
    if port_override is not None:
        config.server.port = port_override

    # Create tracker
    tracker = _create_tracker(config)

    # Create orchestrator
    ws_config = WorkspaceConfig(root=config.workspace.root)
    workspace = Workspace(ws_config)

    pubsub = PubSub()

    # Placeholder agent runner factory
    async def _agent_runner(
        issue: object,
        workspace_path: str | None,
        worker_host: str | None,
    ) -> AgentResult:
        return AgentResult()

    orchestrator = Orchestrator(
        config=config,
        tracker=tracker,
        workspace=workspace,
        agent_runner_factory=_agent_runner,
    )

    # Start API server if dashboard enabled
    if config.observability.dashboard_enabled and config.server.port:
        import uvicorn

        app = create_app(orchestrator=orchestrator, pubsub=pubsub)
        server_config = uvicorn.Config(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level="warning",
        )
        server = uvicorn.Server(server_config)

        # Run orchestrator and server concurrently
        await asyncio.gather(
            orchestrator.start(),
            server.serve(),
        )
    else:
        await orchestrator.start()


def _create_tracker(config: Config) -> Tracker:
    """Create the appropriate tracker based on config."""
    kind = config.tracker.kind
    if kind == "memory":
        from symphony.tracker.memory import MemoryTracker

        return MemoryTracker()
    elif kind == "github":
        from symphony.tracker.github import GitHubTracker

        return GitHubTracker(
            owner=config.tracker.project_slug or "",
            repo=config.tracker.project_slug or "",
            token=config.tracker.api_key or "",
        )
    else:
        raise ValueError(f"Unsupported tracker kind: {kind}")


if __name__ == "__main__":
    main()
