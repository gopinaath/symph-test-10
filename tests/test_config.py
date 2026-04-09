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
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError, match="NONEXISTENT_VAR_XYZ"):
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
        cfg = Config.from_yaml(
            {"tracker": {"active_states": ["TODO", "In Progress"]}}
        )
        assert cfg.tracker.active_states == ["todo", "in progress"]

    def test_terminal_states_lowercased(self) -> None:
        cfg = Config.from_yaml(
            {"tracker": {"terminal_states": ["CLOSED", "Done"]}}
        )
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
