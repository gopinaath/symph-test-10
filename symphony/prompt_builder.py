"""Prompt builder — renders a Jinja2 template with issue + attempt context.

Equivalent to the Liquid-based prompt builder in the Elixir implementation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, UndefinedError


# ---------------------------------------------------------------------------
# Default template used when the workflow prompt body is blank
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE = """\
You are an autonomous coding agent. Solve the following issue.

Issue: {{ issue.identifier }} — {{ issue.title }}
{% if issue.description is defined and issue.description %}
Description:
{{ issue.description }}
{% endif %}
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_value(value: Any) -> Any:
    """Convert datetime-like values to ISO 8601 strings recursively."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _normalize_issue(issue: dict) -> dict:
    """Return a copy of *issue* with all nested date-like values serialised."""
    return {k: _normalize_value(v) for k, v in issue.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PromptBuildError(Exception):
    """Raised when the Jinja2 template cannot be rendered."""


def build_prompt(
    template_source: str,
    issue: dict,
    attempt: int = 1,
) -> str:
    """Render the prompt for a given *issue* and *attempt*.

    Parameters
    ----------
    template_source:
        Raw Jinja2 template text from the workflow file.  If blank, the
        :data:`DEFAULT_TEMPLATE` is used instead.
    issue:
        Issue fields as a plain dict (e.g. from ``Issue.to_dict()``).
    attempt:
        1-based attempt counter.  When ``attempt > 1``, continuation guidance
        is appended after the rendered template.

    Raises
    ------
    PromptBuildError
        On template syntax errors or undefined variable references.
    """
    if not template_source or not template_source.strip():
        template_source = DEFAULT_TEMPLATE

    env = Environment(undefined=StrictUndefined)

    try:
        template = env.from_string(template_source)
    except TemplateSyntaxError as exc:
        raise PromptBuildError(
            f"Template syntax error at line {exc.lineno}: {exc.message}"
        ) from exc

    normalised_issue = _normalize_issue(issue)

    try:
        rendered = template.render(issue=normalised_issue, attempt=attempt)
    except UndefinedError as exc:
        raise PromptBuildError(f"Undefined variable in template: {exc}") from exc
    except Exception as exc:
        raise PromptBuildError(f"Template rendering failed: {exc}") from exc

    if attempt > 1:
        rendered = _add_continuation_guidance(rendered, attempt)

    return rendered


def _add_continuation_guidance(rendered: str, attempt: int) -> str:
    """Append retry guidance to an already-rendered prompt."""
    guidance = (
        f"\n\n---\n"
        f"NOTE: This is attempt {attempt}. A previous attempt to solve this "
        f"issue did not succeed. Please review any earlier work carefully, "
        f"identify what went wrong, and try a different approach if necessary."
    )
    return rendered + guidance
