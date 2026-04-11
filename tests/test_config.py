"""Tests for symphony.config — mirrors the Elixir test suite."""

from __future__ import annotations

import os
import textwrap
from unittest import mock

import pytest
from pydantic import ValidationError

from symphony.config import (
    AgentConfig,
    CodexConfig,
    Config,
    PollingConfig,
    TrackerConfig,
    ValidationConfig,
    set_config,
    settings,
)
from symphony.workflow import Workflow

# ---------------------------------------------------------------------------
# Defaults and validation
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_default_poll_interval(self) -> None:
        cfg = Config()
        assert cfg.polling.interval_ms == 30000

    def test_default_active_states(self) -> None:
        cfg = Config()
        assert cfg.tracker.active_states == ["todo", "in progress"]

    def test_default_terminal_states(self) -> None:
        cfg = Config()
        assert set(cfg.tracker.terminal_states) == {
            "closed",
            "cancelled",
            "canceled",
            "duplicate",
            "done",
        }

    def test_default_max_turns(self) -> None:
        cfg = Config()
        assert cfg.agent.max_turns == 20

    def test_default_codex_command(self) -> None:
        cfg = Config()
        assert cfg.codex.command == "codex"

    def test_default_approval_policy(self) -> None:
        cfg = Config()
        assert cfg.codex.approval_policy == "auto-edit"


class TestConfigValidation:
    def test_poll_interval_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PollingConfig(interval_ms=0)

        with pytest.raises(ValidationError):
            PollingConfig(interval_ms=-1)

    def test_max_turns_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(max_turns=0)

    def test_codex_command_must_not_be_blank(self) -> None:
        with pytest.raises(ValidationError):
            CodexConfig(command="")

        with pytest.raises(ValidationError):
            CodexConfig(command="   ")

    def test_approval_policy_must_be_valid(self) -> None:
        with pytest.raises(ValidationError):
            CodexConfig(approval_policy="yolo")

        # Valid values should work.
        for policy in ("suggest", "auto-edit", "full-auto"):
            c = CodexConfig(approval_policy=policy)
            assert c.approval_policy == policy

    def test_tracker_kind_must_be_valid(self) -> None:
        with pytest.raises(ValidationError):
            TrackerConfig(kind="jira")

        for kind in ("github", "memory"):
            t = TrackerConfig(kind=kind)
            assert t.kind == kind


# ---------------------------------------------------------------------------
# Workflow file path defaults
# ---------------------------------------------------------------------------


class TestWorkflowFilePath:
    def test_workflow_parse_uses_given_path(self, tmp_path) -> None:
        wf_path = tmp_path / "WORKFLOW.md"
        wf_path.write_text("Hello world")
        wf = Workflow.parse(str(wf_path))
        assert wf.prompt_template == "Hello world"

    def test_missing_workflow_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            Workflow.parse(str(tmp_path / "MISSING.md"))


# ---------------------------------------------------------------------------
# Prompt-only files (no front matter)
# ---------------------------------------------------------------------------


class TestPromptOnly:
    def test_load_prompt_only(self, tmp_path) -> None:
        wf_path = tmp_path / "WORKFLOW.md"
        wf_path.write_text("Just a prompt, no YAML.")
        wf = Workflow.parse(str(wf_path))
        assert wf.prompt_template == "Just a prompt, no YAML."
        # Config should be all defaults.
        assert wf.config.polling.interval_ms == 30000


# ---------------------------------------------------------------------------
# Unterminated front matter
# ---------------------------------------------------------------------------


class TestUnterminatedFrontMatter:
    def test_unterminated_front_matter_with_empty_prompt(self, tmp_path) -> None:
        wf_path = tmp_path / "WORKFLOW.md"
        wf_path.write_text(
            textwrap.dedent("""\
                ---
                polling:
                  interval_ms: 5000
            """)
        )
        wf = Workflow.parse(str(wf_path))
        assert wf.config.polling.interval_ms == 5000
        assert wf.prompt_template == ""


# ---------------------------------------------------------------------------
# Non-map front matter
# ---------------------------------------------------------------------------


class TestNonMapFrontMatter:
    def test_rejects_list_front_matter(self, tmp_path) -> None:
        wf_path = tmp_path / "WORKFLOW.md"
        wf_path.write_text("---\n- item1\n- item2\n---\nPrompt here")
        with pytest.raises(ValueError, match="mapping"):
            Workflow.parse(str(wf_path))

    def test_rejects_scalar_front_matter(self, tmp_path) -> None:
        wf_path = tmp_path / "WORKFLOW.md"
        wf_path.write_text("---\njust a string\n---\nPrompt here")
        with pytest.raises(ValueError, match="mapping"):
            Workflow.parse(str(wf_path))


# ---------------------------------------------------------------------------
# $ENV_VAR resolution for secrets
# ---------------------------------------------------------------------------


class TestEnvVarResolution:
    def test_api_key_resolved_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"MY_API_KEY": "secret123"}):
            t = TrackerConfig(api_key="$MY_API_KEY")
            assert t.api_key == "secret123"

    def test_assignee_resolved_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"MY_ASSIGNEE": "user42"}):
            t = TrackerConfig(assignee="$MY_ASSIGNEE")
            assert t.assignee == "user42"

    def test_missing_env_var_raises(self) -> None:
        env = os.environ.copy()
        env.pop("NONEXISTENT_VAR_XYZ", None)
        with mock.patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError, match="NONEXISTENT_VAR_XYZ"):
            TrackerConfig(api_key="$NONEXISTENT_VAR_XYZ")

    def test_non_env_string_passes_through(self) -> None:
        t = TrackerConfig(api_key="literal-key")
        assert t.api_key == "literal-key"


# ---------------------------------------------------------------------------
# Per-state max concurrent agent overrides
# ---------------------------------------------------------------------------


class TestPerStateConcurrency:
    def test_state_overrides_respected(self) -> None:
        cfg = Config.from_yaml(
            {
                "agent": {
                    "max_concurrent_agents_by_state": {
                        "In Progress": 3,
                        "Todo": 5,
                    }
                }
            }
        )
        assert cfg.agent.max_concurrent_agents_by_state == {
            "in progress": 3,
            "todo": 5,
        }


# ---------------------------------------------------------------------------
# State name normalization
# ---------------------------------------------------------------------------


class TestStateNormalization:
    def test_active_states_lowercased(self) -> None:
        cfg = Config.from_yaml({"tracker": {"active_states": ["TODO", "In Progress"]}})
        assert cfg.tracker.active_states == ["todo", "in progress"]

    def test_terminal_states_lowercased(self) -> None:
        cfg = Config.from_yaml({"tracker": {"terminal_states": ["CLOSED", "Done"]}})
        assert cfg.tracker.terminal_states == ["closed", "done"]

    def test_per_state_keys_lowercased(self) -> None:
        a = AgentConfig(max_concurrent_agents_by_state={"IN PROGRESS": 2})
        assert "in progress" in a.max_concurrent_agents_by_state


# ---------------------------------------------------------------------------
# Settings singleton
# ---------------------------------------------------------------------------


class TestSettingsSingleton:
    def test_settings_raises_before_init(self) -> None:
        # Reset global.
        from symphony import config as _mod

        _mod._current = None
        with pytest.raises(RuntimeError):
            settings()

    def test_set_and_get(self) -> None:
        cfg = Config()
        set_config(cfg)
        assert settings() is cfg


# ---------------------------------------------------------------------------
# TEST-004: Config env fallback edge cases
# ---------------------------------------------------------------------------


class TestLegacyEnvReferences:
    """Legacy 'env:' references are treated as literal strings."""

    def test_env_colon_prefix_treated_as_literal(self) -> None:
        """A value like 'env:MY_SECRET' is NOT resolved -- it's a literal
        string.  Only the ``$VAR`` syntax triggers env resolution."""
        t = TrackerConfig(api_key="env:MY_SECRET")
        assert t.api_key == "env:MY_SECRET"

    def test_env_colon_prefix_with_spaces_treated_as_literal(self) -> None:
        t = TrackerConfig(api_key="env: MY_SECRET ")
        assert t.api_key == "env: MY_SECRET "

    def test_env_colon_in_assignee_treated_as_literal(self) -> None:
        t = TrackerConfig(assignee="env:SOME_USER")
        assert t.assignee == "env:SOME_USER"


class TestSchemaNormalizesPolicyKeys:
    """Schema parse normalizes policy keys (state keys lowercased)."""

    def test_from_yaml_normalizes_by_state_keys(self) -> None:
        """Config.from_yaml normalizes max_concurrent_agents_by_state keys
        to lowercase."""
        cfg = Config.from_yaml(
            {
                "agent": {
                    "max_concurrent_agents_by_state": {
                        "In Progress": 3,
                        "TODO": 5,
                        "Review": 1,
                    }
                }
            }
        )
        keys = cfg.agent.max_concurrent_agents_by_state
        assert "in progress" in keys
        assert "todo" in keys
        assert "review" in keys
        assert keys["in progress"] == 3
        assert keys["todo"] == 5
        assert keys["review"] == 1

    def test_mixed_case_state_keys_all_lowered(self) -> None:
        a = AgentConfig(
            max_concurrent_agents_by_state={
                "IN PROGRESS": 2,
                "tOdO": 4,
                "Done": 1,
            }
        )
        assert set(a.max_concurrent_agents_by_state.keys()) == {
            "in progress",
            "todo",
            "done",
        }

    def test_approval_policy_normalized_values(self) -> None:
        """Approval policy accepts the exact allowed values."""
        from symphony.config import CodexConfig

        for policy in ("suggest", "auto-edit", "full-auto"):
            c = CodexConfig(approval_policy=policy)
            assert c.approval_policy == policy


class TestSandboxPolicyResolution:
    """Sandbox policy resolution: explicit pass-through, default from config."""

    def test_explicit_sandbox_policy_passes_through(self) -> None:
        """When sandbox_policy is explicitly set, it is used as-is."""
        from symphony.config import CodexConfig

        c = CodexConfig(thread_sandbox="strict", turn_sandbox_policy="none")
        assert c.thread_sandbox == "strict"
        assert c.turn_sandbox_policy == "none"

    def test_default_sandbox_policy_from_codex_config(self) -> None:
        """Default sandbox policies come from CodexConfig defaults."""
        from symphony.config import CodexConfig

        c = CodexConfig()
        assert c.thread_sandbox == "light"
        assert c.turn_sandbox_policy == "light"

    def test_workspace_root_default(self) -> None:
        """Default workspace root is /tmp/symphony_workspaces."""
        from symphony.config import WorkspaceConfig

        w = WorkspaceConfig()
        assert w.root == "/tmp/symphony_workspaces"

    def test_explicit_workspace_root_passes_through(self) -> None:
        """Explicit workspace root is stored as-is."""
        from symphony.config import WorkspaceConfig

        w = WorkspaceConfig(root="/custom/path")
        assert w.root == "/custom/path"


class TestEmptyEnvVarFallback:
    """Empty $VAR falls back to default env var name."""

    def test_dollar_only_is_literal(self) -> None:
        """A bare '$' (no variable name) is treated as a literal string."""
        t = TrackerConfig(api_key="$")
        assert t.api_key == "$"

    def test_dollar_with_invalid_name_is_literal(self) -> None:
        """$123 does not match the $ENV_VAR pattern and passes through."""
        t = TrackerConfig(api_key="$123")
        assert t.api_key == "$123"

    def test_dollar_with_spaces_is_literal(self) -> None:
        """'$ MY_VAR' is not a valid env reference."""
        t = TrackerConfig(api_key="$ MY_VAR")
        assert t.api_key == "$ MY_VAR"

    def test_dollar_embedded_in_string_is_literal(self) -> None:
        """'prefix$VAR' is not resolved because the $ is not at the start
        of the value (after stripping)."""
        t = TrackerConfig(api_key="prefix$VAR")
        assert t.api_key == "prefix$VAR"

    def test_env_var_set_to_empty_string_resolves_to_empty(self) -> None:
        """If the env var exists but is empty, the resolved value is empty string."""
        with mock.patch.dict(os.environ, {"EMPTY_VAR": ""}):
            t = TrackerConfig(api_key="$EMPTY_VAR")
            assert t.api_key == ""


class TestWorkspaceRootTildeExpansion:
    """Config workspace root tilde expansion for local use."""

    def test_tilde_in_workspace_root_stored_as_is(self) -> None:
        """WorkspaceConfig stores the tilde path as-is (expansion happens
        at usage time, not at config parse time)."""
        from symphony.config import WorkspaceConfig

        w = WorkspaceConfig(root="~/my_workspaces")
        assert w.root == "~/my_workspaces"

    def test_tilde_expansion_can_be_done_at_runtime(self) -> None:
        """os.path.expanduser can expand the stored tilde path."""
        from symphony.config import WorkspaceConfig

        w = WorkspaceConfig(root="~/my_workspaces")
        expanded = os.path.expanduser(w.root)
        assert "~" not in expanded
        assert expanded.endswith("/my_workspaces")

    def test_absolute_workspace_root_unchanged(self) -> None:
        """An absolute path without tilde is stored unchanged."""
        from symphony.config import WorkspaceConfig

        w = WorkspaceConfig(root="/var/workspaces")
        assert w.root == "/var/workspaces"

    def test_full_config_workspace_root_with_tilde(self) -> None:
        """Tilde in workspace root survives full Config construction."""
        cfg = Config.from_yaml({"workspace": {"root": "~/symphony_ws"}})
        assert cfg.workspace.root == "~/symphony_ws"


# ---------------------------------------------------------------------------
# ValidationConfig
# ---------------------------------------------------------------------------


class TestValidationConfig:
    def test_validation_config_defaults(self) -> None:
        vc = ValidationConfig()
        assert vc.enabled is False
        assert vc.command is None
        assert vc.timeout_ms == 120_000
        assert vc.max_attempts == 3
        assert vc.required_for_completion is True
        assert vc.assertions == []

    def test_validation_config_from_yaml(self) -> None:
        cfg = Config.from_yaml(
            {
                "validation": {
                    "enabled": True,
                    "command": "pytest --tb=short",
                    "timeout_ms": 60000,
                    "max_attempts": 5,
                    "required_for_completion": False,
                    "assertions": [
                        {"kind": "file_exists", "path": "setup.py"},
                        {
                            "kind": "command_exit_code",
                            "command": "make lint",
                            "expected": 0,
                        },
                    ],
                }
            }
        )
        vc = cfg.validation
        assert vc.enabled is True
        assert vc.command == "pytest --tb=short"
        assert vc.timeout_ms == 60000
        assert vc.max_attempts == 5
        assert vc.required_for_completion is False
        assert len(vc.assertions) == 2
        assert vc.assertions[0]["kind"] == "file_exists"
        assert vc.assertions[1]["command"] == "make lint"

    def test_validation_config_timeout_positive(self) -> None:
        with pytest.raises(ValidationError):
            ValidationConfig(timeout_ms=0)
        with pytest.raises(ValidationError):
            ValidationConfig(timeout_ms=-1)

    def test_validation_assertions_schema(self) -> None:
        vc = ValidationConfig(
            assertions=[
                {"kind": "file_exists", "path": "README.md"},
                {"kind": "file_contains", "path": "main.py", "pattern": "def main"},
                {"kind": "command_exit_code", "command": "echo ok", "expected": 0},
            ]
        )
        assert len(vc.assertions) == 3
        assert vc.assertions[0]["kind"] == "file_exists"
        assert vc.assertions[1]["kind"] == "file_contains"
        assert vc.assertions[2]["kind"] == "command_exit_code"
