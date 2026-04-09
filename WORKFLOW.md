---
tracker:
  kind: github
  endpoint: https://api.github.com
  api_key: $GITHUB_TOKEN
  project_slug: gopinaath/symph-test-10
  assignee: $GITHUB_ASSIGNEE
  active_states:
    - todo
    - in progress
  terminal_states:
    - closed
    - cancelled
    - done

polling:
  interval_ms: 30000

workspace:
  root: /tmp/symphony_workspaces

agent:
  max_concurrent_agents: 5
  max_turns: 20
  max_retry_backoff_ms: 300000

codex:
  command: codex app-server --stdio
  approval_policy: never
  turn_timeout_ms: 3600000
  stall_timeout_ms: 300000

hooks:
  after_create: |
    cd {{ workspace_path }}
    git clone https://github.com/{{ issue.project_slug }}.git . 2>/dev/null || true
  before_run: |
    cd {{ workspace_path }}
    git checkout -b symphony/{{ issue.identifier }} 2>/dev/null || git checkout symphony/{{ issue.identifier }}
  timeout_ms: 60000

observability:
  dashboard_enabled: true
  refresh_ms: 1000

server:
  port: 4000
  host: 0.0.0.0
---
You are an autonomous software engineer working on issue {{ issue.identifier }}: {{ issue.title }}.

## Issue Details
- **ID**: {{ issue.identifier }}
- **Title**: {{ issue.title }}
- **Description**: {{ issue.description | default("No description provided.") }}
- **Priority**: {{ issue.priority | default("unset") }}
- **State**: {{ issue.state }}
- **Labels**: {{ issue.labels | join(", ") | default("none") }}

## Instructions

1. Read the issue description carefully
2. Understand the codebase context
3. Implement the requested changes
4. Write tests for your changes
5. Ensure all existing tests pass
6. Create a clear commit message referencing {{ issue.identifier }}

{% if attempt and attempt > 1 %}
## Retry Context

This is attempt {{ attempt }}. A previous attempt did not fully resolve this issue.
Review what was done before and try a different approach if needed.
Check the git log for any partial progress from the previous attempt.
{% endif %}

## Quality Gates

- All tests must pass
- Code must follow existing conventions
- Changes must be minimal and focused on the issue
