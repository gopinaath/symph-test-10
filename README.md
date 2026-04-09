# Symphony

Python reimplementation of [OpenAI's Symphony](https://github.com/openai/symphony) — an autonomous coding orchestrator that polls an issue tracker, spawns AI coding agents in isolated workspaces, and manages the full lifecycle with retries, concurrency control, and real-time observability.

Built to run with **Gemma 4** via vLLM on your own GPU infrastructure, or any OpenAI-compatible endpoint.

## What It Does

Symphony is a **CI/CD-style daemon for AI coding agents**:

1. **Polls** GitHub Issues for work (configurable interval)
2. **Dispatches** eligible issues to AI agents (priority-sorted, blocker-aware, concurrency-limited)
3. **Creates isolated workspaces** per issue (git worktrees with lifecycle hooks)
4. **Runs AI agents** that generate code via Codex app-server or direct LLM calls
5. **Manages retries** with exponential backoff on failure, continuation on success
6. **Tracks tokens**, throughput, rate limits, and agent state in real time
7. **Serves a dashboard** — both terminal ANSI UI and Phoenix LiveView web UI

## Architecture

```
symphony/
├── models.py               # Issue, BlockerInfo dataclasses
├── config.py               # WORKFLOW.md YAML config with validation
├── workflow.py              # Workflow parsing + async hot-reload store
├── prompt_builder.py        # Jinja2 templates with strict mode
├── orchestrator.py          # Core polling loop, dispatch, retry state machine
├── agent_runner.py          # Multi-turn Codex agent execution
├── workspace.py             # Per-issue workspace with lifecycle hooks
├── path_safety.py           # Symlink-safe path resolution
├── pr_cleanup.py            # Close PRs on workspace removal
├── observability.py         # PubSub, TPS tracking, event humanization
├── status_dashboard.py      # Full ANSI terminal dashboard renderer
├── api.py                   # FastAPI REST API
├── cli.py                   # CLI entrypoint
├── ssh.py                   # SSH utilities for remote execution
├── log_file.py              # Rotating file logger
├── tracker/
│   ├── base.py              # Abstract Tracker interface
│   ├── memory.py            # In-memory tracker (testing)
│   └── github.py            # GitHub Issues adapter (retry + rate limiting)
└── codex/
    ├── app_server.py         # JSON-RPC 2.0 client over stdio
    └── dynamic_tool.py       # Extensible tool registry
```

## Quick Start

### Prerequisites

- Python 3.10+
- A running vLLM instance (or any OpenAI-compatible endpoint)
- GitHub CLI (`gh`) authenticated for issue tracking
- Optional: Elixir 1.17+ for the Phoenix LiveView dashboard

### Install

```bash
git clone https://github.com/gopinaath/symph-test-10.git
cd symph-test-10
python3 -m venv .venv && source .venv/bin/activate
pip install pydantic pyyaml jinja2 httpx fastapi uvicorn
```

### Configure

Edit `WORKFLOW.md` — it contains both the YAML configuration and the Jinja2 prompt template:

```yaml
---
tracker:
  kind: github
  api_key: $GITHUB_TOKEN
  project_slug: your-org/your-repo
  active_states: [todo, in progress]
  terminal_states: [closed, done]

agent:
  max_concurrent_agents: 5
  max_turns: 20

codex:
  command: codex app-server --stdio
  approval_policy: never
---
You are an autonomous software engineer working on issue {{ issue.identifier }}: {{ issue.title }}.
...
```

### Run

```bash
export GITHUB_TOKEN=ghp_...
export GITHUB_ASSIGNEE=your-username

# Start the orchestrator
python -m symphony.cli \
  --i-understand-that-this-will-be-running-without-the-usual-guardrails \
  WORKFLOW.md
```

The orchestrator will start polling GitHub Issues, dispatching agents, and serving the API on port 4000.

### Dashboard

**Terminal:** The status dashboard renders automatically to stdout.

**Web (Phoenix LiveView):**
```bash
cd dashboard/symphony_dashboard
mix deps.get
mix phx.server
# Visit http://localhost:4000
```

The LiveView dashboard polls the Python API at `localhost:8000/api/v1/state` every second.

## Using with Gemma 4 via vLLM

This project was built and tested against Gemma 4 31B-it served by vLLM on an 8x H200 GPU instance.

### vLLM Setup

```bash
# Serve Gemma 4 with tool calling support
vllm serve google/gemma-4-31B-it \
  --dtype float16 \
  --max-model-len 32768 \
  --port 8002 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.95
```

### Codex + vLLM

To use OpenAI Codex CLI with your vLLM endpoint, configure `~/.codex/config.toml`:

```toml
[model_providers.vllm]
name = "vLLM Gemma 4"
base_url = "http://localhost:9010/v1"
env_key = "VLLM_API_KEY"

[features]
telemetry = false
```

Then run Codex with:
```bash
codex --provider vllm --model google/gemma-4-31B-it "your prompt"
```

### Available Endpoints (tested setup)

| Port | Model | Context | GPUs |
|------|-------|---------|------|
| 8001 | google/gemma-4-31B-it | 256K | 4x H200 (TP=4) |
| 8002 | google/gemma-4-31B-it | 32K | 1x H200 |
| 8003 | zai-org/GLM-4.7-Flash | 32K | 1x H200 |

## Use Cases

- **Autonomous issue resolution** — create GitHub issues, Symphony picks them up and generates fixes
- **Self-hosted AI coding pipeline** — no API costs, no data leaving your infrastructure
- **Overnight backlog processing** — point at 50 tech debt tickets, let it churn
- **Agent evaluation platform** — compare models on real coding tasks with token tracking
- **Template for custom orchestrators** — swap tracker, LLM, workspace, or dashboard components

## Testing

```bash
# Install dev dependencies
pip install pytest pytest-asyncio pytest-timeout respx mypy ruff

# Run unit tests (342 passing)
make test

# Run with E2E tests (requires vLLM + GitHub)
PYTHONPATH=. pytest tests/ --timeout=120

# Lint + type check
make ci
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Config, workflow, prompt builder | 55 | Defaults, validation, env vars, hot-reload, templates |
| Tracker (memory + GitHub) | 26 | CRUD, pagination, retry, rate limiting, state mapping |
| Workspace + path safety | 27 | Hooks, sanitization, symlinks, SSH lifecycle |
| Orchestrator | 48 | Dispatch, retry, reconciliation, snapshots, shutdown |
| Agent runner + app server | 34 | Multi-turn, protocol, approval, tools, buffering |
| Observability + dashboard | 74 | PubSub, TPS, events, ANSI rendering, snapshots |
| API + CLI + SSH | 25 | REST endpoints, auth, tunneling |
| PR cleanup | 9 | Branch detection, PR close, error tolerance |
| E2E (Gemma 4) | 9 | Connectivity, code gen, tool calling, orchestrator lifecycle |
| Extensions | 26 | Integration across all components |
| **Total** | **351** | |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/state` | Full orchestrator snapshot |
| GET | `/api/v1/{identifier}` | Specific issue details |
| POST | `/api/v1/refresh` | Trigger immediate poll cycle |
| GET | `/health` | Health check |

## Project History

This is a Python port of [OpenAI's Symphony](https://github.com/openai/symphony) (originally written in Elixir). Key differences:

- **GitHub Issues** instead of Linear as the issue tracker
- **Gemma 4 / vLLM** instead of GPT-5.3 / OpenAI API
- **FastAPI** REST API instead of Phoenix JSON controllers
- **asyncio** instead of OTP GenServers
- **Jinja2** instead of Liquid for prompt templates
- Phoenix LiveView dashboard connects to the Python backend via REST polling

## License

Apache-2.0
