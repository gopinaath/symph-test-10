"""Tests for symphony.prompt_builder."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from symphony.prompt_builder import (
    PromptBuildError,
    build_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(**overrides) -> dict:
    base = {
        "id": "issue-1",
        "identifier": "PROJ-42",
        "title": "Fix the widget",
        "description": "The widget is broken.",
        "priority": 2,
        "state": "todo",
        "branch_name": "fix-widget",
        "url": "https://example.com/PROJ-42",
        "assignee_id": "user-1",
        "blocked_by": [],
        "labels": ["bug"],
        "assigned_to_worker": True,
        "created_at": None,
        "updated_at": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Rendering basics
# ---------------------------------------------------------------------------


class TestRendering:
    def test_renders_issue_and_attempt(self) -> None:
        tpl = "Issue: {{ issue.identifier }} (attempt {{ attempt }})"
        result = build_prompt(tpl, _make_issue(), attempt=1)
        assert "PROJ-42" in result
        assert "attempt 1" in result

    def test_renders_issue_fields(self) -> None:
        tpl = "Title: {{ issue.title }}\nState: {{ issue.state }}"
        result = build_prompt(tpl, _make_issue(), attempt=1)
        assert "Fix the widget" in result
        assert "todo" in result


# ---------------------------------------------------------------------------
# DateTime handling
# ---------------------------------------------------------------------------


class TestDateTimeHandling:
    def test_datetime_fields_do_not_crash(self) -> None:
        issue = _make_issue(
            created_at=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            updated_at=datetime(2025, 1, 16, 12, 0, 0, tzinfo=timezone.utc),
        )
        tpl = "Created: {{ issue.created_at }}"
        result = build_prompt(tpl, issue, attempt=1)
        assert "2025-01-15" in result

    def test_normalizes_nested_date_values(self) -> None:
        issue = _make_issue(
            blocked_by=[
                {
                    "id": "b1",
                    "identifier": "PROJ-10",
                    "state": "done",
                    "due_date": datetime(2025, 6, 1, tzinfo=timezone.utc),
                }
            ]
        )
        tpl = "Blocker due: {{ issue.blocked_by[0].due_date }}"
        result = build_prompt(tpl, issue, attempt=1)
        assert "2025-06-01" in result


# ---------------------------------------------------------------------------
# Strict variable rendering
# ---------------------------------------------------------------------------


class TestStrictUndefined:
    def test_unknown_variable_raises(self) -> None:
        tpl = "Hello {{ unknown_var }}"
        with pytest.raises(PromptBuildError, match="Undefined variable"):
            build_prompt(tpl, _make_issue(), attempt=1)

    def test_unknown_nested_variable_raises(self) -> None:
        tpl = "Hello {{ issue.nonexistent_field }}"
        with pytest.raises(PromptBuildError, match="Undefined variable"):
            build_prompt(tpl, _make_issue(), attempt=1)


# ---------------------------------------------------------------------------
# Invalid template syntax
# ---------------------------------------------------------------------------


class TestInvalidTemplate:
    def test_surfaces_syntax_error_with_context(self) -> None:
        tpl = "{% if true %}oops"  # missing endif
        with pytest.raises(PromptBuildError, match="Template syntax error"):
            build_prompt(tpl, _make_issue(), attempt=1)

    def test_unclosed_variable_tag(self) -> None:
        tpl = "Hello {{ issue.title"
        with pytest.raises(PromptBuildError, match="Template syntax error"):
            build_prompt(tpl, _make_issue(), attempt=1)


# ---------------------------------------------------------------------------
# Default template
# ---------------------------------------------------------------------------


class TestDefaultTemplate:
    def test_blank_prompt_uses_default(self) -> None:
        result = build_prompt("", _make_issue(), attempt=1)
        assert "PROJ-42" in result
        assert "Fix the widget" in result

    def test_whitespace_only_uses_default(self) -> None:
        result = build_prompt("   \n  ", _make_issue(), attempt=1)
        assert "PROJ-42" in result

    def test_default_template_handles_missing_description(self) -> None:
        issue = _make_issue(description="")
        result = build_prompt("", issue, attempt=1)
        assert "PROJ-42" in result
        assert "Fix the widget" in result
        # Description block should not appear since description is empty.
        assert "Description:" not in result

    def test_default_template_handles_none_description(self) -> None:
        issue = _make_issue(description=None)
        result = build_prompt("", issue, attempt=1)
        assert "PROJ-42" in result


# ---------------------------------------------------------------------------
# Workflow load failures reported separately
# ---------------------------------------------------------------------------


class TestWorkflowLoadFailure:
    """Ensure that template build errors and workflow parse errors are
    distinct — a bad template does not masquerade as a workflow error."""

    def test_bad_template_raises_prompt_build_error(self) -> None:
        with pytest.raises(PromptBuildError):
            build_prompt("{% invalid %}", _make_issue(), attempt=1)

    def test_workflow_file_error_is_file_not_found(self) -> None:
        from symphony.workflow import Workflow

        with pytest.raises(FileNotFoundError):
            Workflow.parse("/nonexistent/path/WORKFLOW.md")


# ---------------------------------------------------------------------------
# Continuation guidance for retries
# ---------------------------------------------------------------------------


class TestContinuationGuidance:
    def test_no_guidance_on_first_attempt(self) -> None:
        tpl = "Do the thing."
        result = build_prompt(tpl, _make_issue(), attempt=1)
        assert "attempt" not in result.lower() or "attempt 1" not in result.lower()
        assert "previous attempt" not in result.lower()

    def test_guidance_added_on_retry(self) -> None:
        tpl = "Do the thing."
        result = build_prompt(tpl, _make_issue(), attempt=2)
        assert "attempt 2" in result.lower()
        assert "previous attempt" in result.lower()

    def test_guidance_on_higher_attempt(self) -> None:
        tpl = "Do the thing."
        result = build_prompt(tpl, _make_issue(), attempt=5)
        assert "attempt 5" in result.lower()
        assert "different approach" in result.lower()


# ---------------------------------------------------------------------------
# TEST-007: In-repo WORKFLOW.md rendering
# ---------------------------------------------------------------------------


class TestInRepoWorkflow:
    """Verify the actual WORKFLOW.md in the repository is valid and renderable.

    The WORKFLOW.md front matter references environment variables ($GITHUB_TOKEN,
    $GITHUB_ASSIGNEE) which may not be set in CI.  We use the internal
    ``_split_front_matter`` helper to extract the prompt template without
    triggering Config env-var resolution, and separately validate the YAML
    structure.
    """

    _WORKFLOW_PATH = str(
        (Path(__file__).resolve().parent.parent / "WORKFLOW.md")
    )

    @staticmethod
    def _load_prompt_template() -> str:
        """Read WORKFLOW.md and extract the prompt template body."""
        from symphony.workflow import _read_file, _split_front_matter

        text = _read_file(
            str(Path(__file__).resolve().parent.parent / "WORKFLOW.md")
        )
        yaml_str, prompt = _split_front_matter(text)
        return prompt

    def test_in_repo_workflow_is_valid(self) -> None:
        """Load the actual WORKFLOW.md from the repo root, verify it parses.

        Checks that:
        - The file exists and is readable.
        - The front matter is valid YAML (a mapping).
        - The prompt template is non-empty.
        """
        import yaml
        from symphony.workflow import _read_file, _split_front_matter

        text = _read_file(self._WORKFLOW_PATH)
        yaml_str, prompt = _split_front_matter(text)

        # Front matter should exist and be a valid YAML mapping.
        assert yaml_str is not None, "WORKFLOW.md should have YAML front matter"
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict), "Front matter must be a YAML mapping"

        # Prompt template should be non-empty.
        assert prompt.strip() != "", "Prompt template body should not be empty"

    def test_in_repo_workflow_renders_with_issue_context(self) -> None:
        """Render the WORKFLOW.md template with a sample issue, verify output."""
        prompt_template = self._load_prompt_template()

        issue = _make_issue(
            identifier="TEST-99",
            title="Add pagination support",
            description="We need pagination for the listing endpoint.",
            priority=2,
            state="todo",
            labels=["enhancement", "backend"],
        )

        result = build_prompt(prompt_template, issue, attempt=1)

        # The rendered output should contain our issue fields.
        assert "TEST-99" in result
        assert "Add pagination support" in result
        assert "We need pagination" in result
        assert "todo" in result
        # Labels should appear.
        assert "enhancement" in result
        assert "backend" in result

    def test_in_repo_workflow_renders_retry_context(self) -> None:
        """Render WORKFLOW.md with attempt > 1, verify retry block appears."""
        prompt_template = self._load_prompt_template()

        issue = _make_issue(identifier="TEST-100", title="Retry test")

        result = build_prompt(prompt_template, issue, attempt=3)

        # The WORKFLOW.md has a retry context block for attempt > 1.
        assert "attempt 3" in result.lower()


# ---------------------------------------------------------------------------
# Validation failure context
# ---------------------------------------------------------------------------


class TestValidationContext:
    def test_validation_output_rendered_in_prompt(self) -> None:
        tpl = "Do the thing."
        result = build_prompt(
            tpl, _make_issue(), attempt=1, validation_output="FAILED: test_foo"
        )
        assert "Validation Failed" in result
        assert "FAILED: test_foo" in result

    def test_empty_validation_output_produces_no_block(self) -> None:
        tpl = "Do the thing."
        result = build_prompt(tpl, _make_issue(), attempt=1, validation_output="")
        assert "Validation Failed" not in result

    def test_validation_attempt_shown_when_positive(self) -> None:
        tpl = "Do the thing."
        result = build_prompt(
            tpl,
            _make_issue(),
            attempt=1,
            validation_output="FAILED: test_bar",
            validation_attempt=2,
        )
        assert "attempt 2" in result.lower()

    def test_validation_context_combined_with_retry_context(self) -> None:
        tpl = "Do the thing."
        result = build_prompt(
            tpl,
            _make_issue(),
            attempt=2,
            validation_output="FAILED: test_baz",
            validation_attempt=2,
        )
        # Retry guidance from attempt > 1
        assert "previous attempt" in result.lower()
        # Validation block
        assert "Validation Failed" in result
        assert "FAILED: test_baz" in result
        assert "attempt 2" in result.lower()

    def test_validation_output_truncated_for_very_long_output(self) -> None:
        from symphony.prompt_builder import MAX_VALIDATION_OUTPUT_LENGTH

        long_output = "x" * 100_000  # 100 KB
        result = build_prompt(
            "Do the thing.",
            _make_issue(),
            attempt=1,
            validation_output=long_output,
        )
        assert "Validation Failed" in result
        # The raw output should have been truncated.
        assert "[output truncated]" in result
        # The total embedded validation output should not exceed the limit
        # plus the truncation notice.
        assert len(result) < MAX_VALIDATION_OUTPUT_LENGTH + 1000
