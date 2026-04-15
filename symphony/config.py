"""Configuration loaded from WORKFLOW.md YAML front matter.

Secret fields (``api_key``, ``assignee``) support ``$ENV_VAR`` references that
are resolved from ``os.environ`` at runtime.  State names are normalised to
lowercase throughout.
"""

from __future__ import annotations

import os
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")


def _resolve_env(value: Any) -> Any:
    """If *value* is a string of the form ``$ENV_VAR``, resolve it."""
    if isinstance(value, str):
        m = _ENV_RE.match(value)
        if m:
            env_name = m.group(1)
            resolved = os.environ.get(env_name)
            if resolved is None:
                raise ValueError(f"Environment variable {env_name!r} referenced by ${env_name} is not set")
            return resolved
    return value


def _normalize_states(states: list[str]) -> list[str]:
    return [s.lower() for s in states]


class TrackerConfig(BaseModel):
    kind: str = "github"
    endpoint: str | None = None
    api_key: str | None = None
    project_slug: str | None = None
    assignee: str | None = None
    active_states: list[str] = Field(default_factory=lambda: ["todo", "in progress"])
    terminal_states: list[str] = Field(
        default_factory=lambda: [
            "closed",
            "cancelled",
            "canceled",
            "duplicate",
            "done",
        ]
    )

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, v: str) -> str:
        allowed = {"github", "memory"}
        if v not in allowed:
            raise ValueError(f"tracker.kind must be one of {allowed}, got {v!r}")
        return v

    @field_validator("active_states", "terminal_states", mode="before")
    @classmethod
    def _normalize(cls, v: list[str]) -> list[str]:
        return _normalize_states(v)

    @field_validator("api_key", "assignee", mode="before")
    @classmethod
    def _resolve_secrets(cls, v: Any) -> Any:
        return _resolve_env(v)


class PollingConfig(BaseModel):
    interval_ms: int = 30000

    @field_validator("interval_ms")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("polling.interval_ms must be positive")
        return v


class WorkspaceConfig(BaseModel):
    root: str = "/tmp/symphony_workspaces"


class WorkerConfig(BaseModel):
    ssh_hosts: list[str] = Field(default_factory=list)
    max_concurrent_agents_per_host: int = 1


class AgentConfig(BaseModel):
    max_concurrent_agents: int = 10
    max_turns: int = 20
    max_retry_backoff_ms: int = 300000
    max_concurrent_agents_by_state: dict[str, int] = Field(default_factory=dict)
    agent_mode: str = "single"  # "single" or "two_phase"

    @field_validator("max_turns")
    @classmethod
    def _positive_turns(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("agent.max_turns must be positive")
        return v

    @field_validator("agent_mode")
    @classmethod
    def _validate_agent_mode(cls, v: str) -> str:
        allowed = {"single", "two_phase"}
        if v not in allowed:
            raise ValueError(f"agent.agent_mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("max_concurrent_agents_by_state", mode="before")
    @classmethod
    def _normalize_state_keys(cls, v: dict[str, int]) -> dict[str, int]:
        if not isinstance(v, dict):
            return v
        return {k.lower(): val for k, val in v.items()}


class CodexConfig(BaseModel):
    command: str = "codex"
    approval_policy: str = "auto-edit"
    thread_sandbox: str = "light"
    turn_sandbox_policy: str = "light"
    turn_timeout_ms: int = 3600000
    read_timeout_ms: int = 5000
    stall_timeout_ms: int = 300000

    @field_validator("command")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("codex.command must not be blank")
        return v

    @field_validator("approval_policy")
    @classmethod
    def _validate_policy(cls, v: str) -> str:
        allowed = {"suggest", "auto-edit", "full-auto", "never"}
        if v not in allowed:
            raise ValueError(f"codex.approval_policy must be one of {allowed}, got {v!r}")
        return v


class ValidationConfig(BaseModel):
    enabled: bool = False
    command: str | None = None
    timeout_ms: int = 120_000
    max_attempts: int = 3
    required_for_completion: bool = True
    assertions: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("timeout_ms")
    @classmethod
    def _positive_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("validation.timeout_ms must be positive")
        return v


class HooksConfig(BaseModel):
    after_create: str | None = None
    before_run: str | None = None
    after_run: str | None = None
    before_remove: str | None = None
    timeout_ms: int = 60000


class ObservabilityConfig(BaseModel):
    dashboard_enabled: bool = False
    refresh_ms: int = 1000
    render_interval_ms: int = 16


class QAConfig(BaseModel):
    enabled: bool = False
    prompt_template: str = (
        "You are a QA engineer reviewing code changes. You did NOT write this code. "
        "Test each feature through the available tools and report any failures."
    )
    approval_policy: str = "suggest"  # read-only, cannot write files
    max_turns: int = 5

    @field_validator("approval_policy")
    @classmethod
    def _validate_qa_policy(cls, v: str) -> str:
        allowed = {"suggest", "auto-edit", "full-auto", "never"}
        if v not in allowed:
            raise ValueError(f"qa.approval_policy must be one of {allowed}, got {v!r}")
        return v


class ServerConfig(BaseModel):
    port: int | None = None
    host: str = "127.0.0.1"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """Top-level Symphony configuration."""

    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    polling: PollingConfig = Field(default_factory=PollingConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    codex: CodexConfig = Field(default_factory=CodexConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    qa: QAConfig = Field(default_factory=QAConfig)

    # -- convenience ---------------------------------------------------------

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> Config:
        """Build a ``Config`` from a parsed YAML dict (front matter)."""
        if not isinstance(data, dict):
            raise TypeError(f"YAML front matter must be a mapping, got {type(data).__name__}")
        return cls.model_validate(data)

    @model_validator(mode="after")
    def _cross_validate(self) -> Config:
        # Ensure terminal and active states don't overlap.
        overlap = set(self.tracker.active_states) & set(self.tracker.terminal_states)
        if overlap:
            raise ValueError(f"active_states and terminal_states overlap: {overlap}")
        return self


# ---------------------------------------------------------------------------
# Module-level singleton accessor (optional convenience)
# ---------------------------------------------------------------------------

_current: Config | None = None


def set_config(cfg: Config) -> None:
    global _current
    _current = cfg


def settings() -> Config:
    """Return the current ``Config``, raising if none has been set."""
    if _current is None:
        raise RuntimeError("Config has not been initialised -- call set_config() first")
    return _current
