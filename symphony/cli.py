"""CLI entrypoint for Symphony."""

import argparse
import asyncio
import os
import sys


def main():
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
    workflow_path = args.workflow_path
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

    logs_root = args.logs_root
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


async def _run(workflow_path: str, port_override: int | None = None):
    """Main async entry point."""
    from symphony.api import create_app
    from symphony.config import Config
    from symphony.observability import PubSub
    from symphony.orchestrator import Orchestrator
    from symphony.workflow import Workflow

    # Load workflow
    workflow = Workflow.parse(workflow_path)
    config = Config.from_yaml(workflow.config)

    # Override port if specified
    if port_override is not None:
        config.server.port = port_override

    # Create tracker
    tracker = _create_tracker(config)

    # Create orchestrator
    from symphony.workspace import Workspace
    workspace = Workspace(config.workspace.root)

    pubsub = PubSub()
    orchestrator = Orchestrator(
        config=config,
        tracker=tracker,
        workspace=workspace,
        pubsub=pubsub,
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


def _create_tracker(config):
    """Create the appropriate tracker based on config."""
    kind = config.tracker.kind
    if kind == "memory":
        from symphony.tracker.memory import MemoryTracker
        return MemoryTracker()
    elif kind == "github":
        from symphony.tracker.github import GitHubTracker
        return GitHubTracker(
            endpoint=config.tracker.endpoint,
            api_key=config.tracker.api_key,
            project_slug=config.tracker.project_slug,
            assignee=config.tracker.assignee,
            active_states=config.tracker.active_states,
            terminal_states=config.tracker.terminal_states,
        )
    else:
        raise ValueError(f"Unsupported tracker kind: {kind}")


if __name__ == "__main__":
    main()
