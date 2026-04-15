"""Executes a single issue autonomously through the Codex app-server."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from symphony.codex.app_server import AppServer, AppServerConfig
from symphony.config import QAConfig, ValidationConfig
from symphony.feature_tracker import FeatureTracker
from symphony.models import FeatureList, FeatureTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward-reference type stubs (real implementations live elsewhere)
# ---------------------------------------------------------------------------


class Issue(Protocol):  # symphony.models.Issue
    id: str
    title: str
    state: str

    def is_terminal(self) -> bool: ...


class Workspace(Protocol):  # symphony.workspace.Workspace
    path: str

    async def prepare(self) -> None: ...

    async def cleanup(self) -> None: ...


class PromptBuilder(Protocol):  # symphony.prompt_builder.PromptBuilder
    def initial_prompt(self, issue: Any) -> str: ...

    def continuation_prompt(self, issue: Any, turn: int) -> str: ...


# ---------------------------------------------------------------------------
# Update callback protocol
# ---------------------------------------------------------------------------


class RunUpdateCallback(Protocol):
    """Callback the orchestrator passes in to receive progress updates."""

    async def session_started(self, session_id: str) -> None: ...

    async def turn_completed(self, turn_number: int, usage: dict[str, Any]) -> None: ...

    async def codex_update(self, event_type: str, payload: Any) -> None: ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkspacePrepareFailedError(Exception):
    """SSH or local workspace preparation failed."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentRunnerConfig:
    max_turns: int = 5
    app_server_config: AppServerConfig = field(default_factory=AppServerConfig)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    issue_id: str
    turns_executed: int
    total_usage: dict[str, Any] = field(default_factory=dict)
    stopped_reason: str = ""
    validation_passed: bool = False
    validation_output: str = ""
    features_completed: int = 0
    features_total: int = 0
    qa_passed: bool = False
    qa_findings: str = ""


# ---------------------------------------------------------------------------
# Hook protocol
# ---------------------------------------------------------------------------


class RunHook(Protocol):
    async def before_run(self, issue: Any, workspace: Any) -> None: ...

    async def after_run(self, issue: Any, workspace: Any, result: RunResult) -> None: ...


class _NullHook:
    async def before_run(self, issue: Any, workspace: Any) -> None:
        pass

    async def after_run(self, issue: Any, workspace: Any, result: RunResult) -> None:
        pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AgentRunner:
    """Runs a single issue through one or more Codex turns."""

    def __init__(
        self,
        config: AgentRunnerConfig | None = None,
        *,
        prompt_builder: PromptBuilder | None = None,
        hook: RunHook | None = None,
        app_server_factory: Callable[..., AppServer] | None = None,
        agent_mode: str = "single",
        qa_config: QAConfig | None = None,
        feature_tracker: FeatureTracker | None = None,
    ) -> None:
        self._config = config or AgentRunnerConfig()
        self._prompt_builder = prompt_builder
        self._hook: RunHook = hook or _NullHook()
        self._app_server_factory = app_server_factory or self._default_app_server
        self._agent_mode = agent_mode
        self._qa_config = qa_config or QAConfig()
        self._feature_tracker = feature_tracker

    # -- public API ----------------------------------------------------------

    async def run(
        self,
        issue: Any,
        workspace: Any,
        *,
        callback: RunUpdateCallback | None = None,
        reuse_workspace: bool = False,
        validation_config: ValidationConfig | None = None,
    ) -> RunResult:
        """Execute the issue end-to-end and return the result."""

        # 1. Prepare workspace.
        if not reuse_workspace:
            try:
                await workspace.prepare()
            except Exception as exc:
                raise WorkspacePrepareFailedError(str(exc)) from exc

        # 2. Before-run hook.
        await self._hook.before_run(issue, workspace)

        # 3. Dispatch based on agent_mode.
        if self._agent_mode == "two_phase":
            result_obj = await self._run_two_phase(
                issue, workspace, callback=callback,
                validation_config=validation_config,
            )
        else:
            result_obj = await self._run_single(
                issue, workspace, callback=callback,
                validation_config=validation_config,
            )

        # 4. QA phase (runs after either single or two_phase).
        if self._qa_config.enabled:
            await self._run_qa_phase(issue, workspace, result_obj, callback=callback)

        # 5. After-run hook.
        await self._hook.after_run(issue, workspace, result_obj)

        return result_obj

    # -- single-agent loop ---------------------------------------------------

    async def _run_single(
        self,
        issue: Any,
        workspace: Any,
        *,
        callback: RunUpdateCallback | None = None,
        validation_config: ValidationConfig | None = None,
    ) -> RunResult:
        """Original single-agent turn loop."""
        events: list[tuple[str, Any]] = []

        async def _on_event(etype: str, payload: Any) -> None:
            ts = time.time()
            events.append((etype, payload))
            if callback:
                await callback.codex_update(etype, {"ts": ts, **({} if not isinstance(payload, dict) else payload)})

        server = self._app_server_factory(
            config=self._config.app_server_config,
            on_event=_on_event,
        )
        try:
            await server.start(workspace.path)
            session_id = await server.start_thread(workspace.path)
            if callback:
                await callback.session_started(session_id)

            total_usage: dict[str, Any] = {}
            turns_executed = 0
            stopped_reason = "max_turns"
            validation_passed = False
            validation_output = ""
            validation_feedback: str | None = None

            for turn_number in range(1, self._config.max_turns + 1):
                if turn_number > 1 and issue.is_terminal():
                    stopped_reason = "terminal_state"
                    break

                if validation_feedback is not None:
                    prompt = (
                        f"The previous attempt failed validation. "
                        f"Please fix the issues and try again.\n\n"
                        f"Validation output:\n{validation_feedback}"
                    )
                elif self._prompt_builder is not None:
                    if turn_number == 1:
                        prompt = self._prompt_builder.initial_prompt(issue)
                    else:
                        prompt = self._prompt_builder.continuation_prompt(issue, turn_number)
                else:
                    prompt = issue.title if turn_number == 1 else f"Continue working on: {issue.title}"

                result = await server.run_turn(
                    input_text=prompt,
                    cwd=workspace.path,
                    title=issue.title,
                )
                turn_usage = result.get("usage", {})
                turns_executed = turn_number
                _merge_usage(total_usage, turn_usage)

                if callback:
                    await callback.turn_completed(turn_number, turn_usage)

                validation_feedback = None

                if (
                    validation_config is not None
                    and validation_config.enabled
                    and hasattr(workspace, "run_validation")
                ):
                    v_result = await workspace.run_validation(issue.id, validation_config)
                    if v_result.passed:
                        validation_passed = True
                        validation_output = v_result.stdout
                        stopped_reason = "validation_passed"
                        break
                    else:
                        validation_output = v_result.stderr or v_result.stdout
                        validation_feedback = validation_output

            return RunResult(
                issue_id=issue.id,
                turns_executed=turns_executed,
                total_usage=total_usage,
                stopped_reason=stopped_reason,
                validation_passed=validation_passed,
                validation_output=validation_output,
            )
        finally:
            await server.stop()

    # -- two-phase loop ------------------------------------------------------

    async def _run_two_phase(
        self,
        issue: Any,
        workspace: Any,
        *,
        callback: RunUpdateCallback | None = None,
        validation_config: ValidationConfig | None = None,
    ) -> RunResult:
        """Two-phase flow: initializer decomposes, coder implements per-feature."""
        tracker = self._feature_tracker
        events: list[tuple[str, Any]] = []

        async def _on_event(etype: str, payload: Any) -> None:
            ts = time.time()
            events.append((etype, payload))
            if callback:
                await callback.codex_update(etype, {"ts": ts, **({} if not isinstance(payload, dict) else payload)})

        server = self._app_server_factory(
            config=self._config.app_server_config,
            on_event=_on_event,
        )
        try:
            await server.start(workspace.path)
            session_id = await server.start_thread(workspace.path)
            if callback:
                await callback.session_started(session_id)

            total_usage: dict[str, Any] = {}
            turns_executed = 0

            # -- Phase 1: Load or generate feature list --------------------------
            feature_list: FeatureList | None = None
            if tracker is not None:
                feature_list = tracker.load(issue.id)

            if feature_list is None:
                # Run initializer turn to decompose the issue.
                decomposition_prompt = (
                    f"Break this issue into a list of verifiable features. "
                    f"Output a JSON array of objects with fields: "
                    f"id, description, category, steps, test_command.\n\n"
                    f"Issue: {issue.title}"
                )
                result = await server.run_turn(
                    input_text=decomposition_prompt,
                    cwd=workspace.path,
                    title=issue.title,
                )
                turn_usage = result.get("usage", {})
                turns_executed += 1
                _merge_usage(total_usage, turn_usage)

                if callback:
                    await callback.turn_completed(turns_executed, turn_usage)

                # Parse agent response to extract JSON feature array.
                response_text = result.get("response", "") or ""
                feature_list = _parse_feature_list(issue.id, response_text)

                if tracker is not None and feature_list is not None:
                    tracker.save(issue.id, feature_list)

            if feature_list is None:
                # Could not decompose; return early.
                return RunResult(
                    issue_id=issue.id,
                    turns_executed=turns_executed,
                    total_usage=total_usage,
                    stopped_reason="decomposition_failed",
                )

            # -- Phase 2: Coder turns per feature --------------------------------
            while turns_executed < self._config.max_turns:
                feature = feature_list.next_pending()
                if feature is None:
                    break  # All features handled.

                feature.status = "in_progress"

                # Build a focused prompt for this feature.
                steps_text = "\n".join(f"  - {s}" for s in feature.steps)
                prompt = (
                    f"Implement the following feature:\n"
                    f"ID: {feature.id}\n"
                    f"Description: {feature.description}\n"
                    f"Category: {feature.category}\n"
                    f"Steps:\n{steps_text}"
                )
                if feature.test_command:
                    prompt += f"\n\nValidation command: {feature.test_command}"

                result = await server.run_turn(
                    input_text=prompt,
                    cwd=workspace.path,
                    title=issue.title,
                )
                turn_usage = result.get("usage", {})
                turns_executed += 1
                _merge_usage(total_usage, turn_usage)

                if callback:
                    await callback.turn_completed(turns_executed, turn_usage)

                # Run validation if configured.
                v_passed = False
                if (
                    validation_config is not None
                    and validation_config.enabled
                    and hasattr(workspace, "run_validation")
                ):
                    v_result = await workspace.run_validation(issue.id, validation_config)
                    v_passed = v_result.passed
                    if not v_passed:
                        v_output = v_result.stderr or v_result.stdout
                        feature_list.mark_failed(feature.id, error=v_output)
                    else:
                        feature_list.mark_passed(feature.id)
                else:
                    # No validation: assume passed.
                    feature_list.mark_passed(feature.id)

                if tracker is not None:
                    tracker.save(issue.id, feature_list)

            completed, total = feature_list.progress()
            stopped_reason = "all_features_passed" if feature_list.all_passed() else "max_turns"

            return RunResult(
                issue_id=issue.id,
                turns_executed=turns_executed,
                total_usage=total_usage,
                stopped_reason=stopped_reason,
                validation_passed=feature_list.all_passed(),
                features_completed=completed,
                features_total=total,
            )
        finally:
            await server.stop()

    # -- QA phase ------------------------------------------------------------

    async def _run_qa_phase(
        self,
        issue: Any,
        workspace: Any,
        result_obj: RunResult,
        *,
        callback: RunUpdateCallback | None = None,
    ) -> None:
        """Run QA agent to verify passed features. Modifies *result_obj* in place."""
        qa_config = self._qa_config
        tracker = self._feature_tracker

        # Load feature list for QA review.
        feature_list: FeatureList | None = None
        if tracker is not None:
            feature_list = tracker.load(issue.id)

        # Build list of passed features for the QA prompt.
        passed_features: list[FeatureTask] = []
        if feature_list is not None:
            passed_features = [f for f in feature_list.features if f.status == "passed"]

        if not passed_features and feature_list is not None and feature_list.features:
            # No passed features to verify.
            result_obj.qa_passed = False
            result_obj.qa_findings = "No passed features to verify."
            return

        # Build QA prompt.
        features_text = "\n".join(
            f"- {f.id}: {f.description}" for f in passed_features
        )
        qa_prompt = (
            f"{qa_config.prompt_template}\n\n"
            f"Features marked as passed:\n{features_text}\n\n"
            f"Workspace: {workspace.path}\n\n"
            f"For each feature, output a verdict line in the format:\n"
            f"VERDICT:<feature_id>:PASS or VERDICT:<feature_id>:FAIL:<reason>"
        )

        events: list[tuple[str, Any]] = []

        async def _on_event(etype: str, payload: Any) -> None:
            ts = time.time()
            events.append((etype, payload))
            if callback:
                await callback.codex_update(etype, {"ts": ts, **({} if not isinstance(payload, dict) else payload)})

        server = self._app_server_factory(
            config=self._config.app_server_config,
            on_event=_on_event,
        )
        try:
            await server.start(workspace.path)
            session_id = await server.start_thread(
                workspace.path,
                approval_policy=qa_config.approval_policy,
            )

            for qa_turn in range(1, qa_config.max_turns + 1):
                result = await server.run_turn(
                    input_text=qa_prompt if qa_turn == 1 else "Continue reviewing.",
                    cwd=workspace.path,
                    title=f"QA: {issue.title}",
                    approval_policy=qa_config.approval_policy,
                )
                result_obj.turns_executed += 1
                turn_usage = result.get("usage", {})
                _merge_usage(result_obj.total_usage, turn_usage)
        finally:
            await server.stop()

        # Parse QA response for verdicts.
        qa_response = result.get("response", "") or ""
        verdicts = _parse_qa_verdicts(qa_response)

        # Apply verdicts: QA can only fail features, not pass them.
        passed_ids = {f.id for f in passed_features}
        verified_ids: set[str] = set()

        for feat_id, verdict, reason in verdicts:
            if feat_id not in passed_ids:
                continue
            if verdict == "FAIL" and feature_list is not None:
                feature_list.mark_failed(feat_id, error=reason)
            elif verdict == "PASS":
                verified_ids.add(feat_id)

        # Any passed feature without a PASS verdict gets reset to failed.
        if feature_list is not None:
            for f in passed_features:
                if f.id not in verified_ids:
                    # Check if QA already failed it.
                    current = next((ft for ft in feature_list.features if ft.id == f.id), None)
                    if current is not None and current.status == "passed":
                        feature_list.mark_failed(f.id, error="No QA verdict received")

            if tracker is not None:
                tracker.save(issue.id, feature_list)

            completed, total = feature_list.progress()
            result_obj.features_completed = completed
            result_obj.features_total = total

        # Save QA findings.
        findings_lines = []
        for feat_id, verdict, reason in verdicts:
            findings_lines.append(f"{feat_id}: {verdict}" + (f" - {reason}" if reason else ""))

        qa_findings_text = "\n".join(findings_lines) if findings_lines else "No verdicts parsed."
        result_obj.qa_findings = qa_findings_text
        result_obj.qa_passed = (
            feature_list is not None
            and feature_list.all_passed()
        )

        # Write qa-findings.txt to workspace.
        findings_path = os.path.join(workspace.path, "qa-findings.txt")
        try:
            os.makedirs(os.path.dirname(findings_path), exist_ok=True)
            with open(findings_path, "w") as fh:
                fh.write(qa_findings_text)
        except OSError:
            logger.warning("Failed to write QA findings to %s", findings_path)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _default_app_server(
        config: AppServerConfig | None = None,
        on_event: Any = None,
    ) -> AppServer:
        return AppServer(config, on_event=on_event)


def _merge_usage(total: dict[str, Any], turn: dict[str, Any]) -> None:
    for k, v in turn.items():
        if isinstance(v, (int, float)):
            total[k] = total.get(k, 0) + v
        else:
            total[k] = v


def _parse_feature_list(issue_id: str, response_text: str) -> FeatureList | None:
    """Extract a JSON array of feature objects from agent response text."""
    text = response_text.strip()

    # Try to parse the entire text as JSON first.
    items = _try_parse_json_array(text)

    # If that fails, try to find a JSON array in the text using bracket matching.
    if items is None:
        start = text.find("[")
        while start != -1:
            # Find the matching closing bracket.
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        items = _try_parse_json_array(candidate)
                        if items is not None:
                            break
                        break
            if items is not None:
                break
            start = text.find("[", start + 1)

    if items is None or not items:
        return None

    features: list[FeatureTask] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            features.append(
                FeatureTask(
                    id=item["id"],
                    description=item["description"],
                    category=item.get("category", "general"),
                    steps=item.get("steps", []),
                    test_command=item.get("test_command"),
                )
            )
        except (KeyError, TypeError):
            continue

    if not features:
        return None

    return FeatureList(issue_id=issue_id, features=features)


def _try_parse_json_array(text: str) -> list[Any] | None:
    """Try to parse text as a JSON array. Returns None on failure."""
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_qa_verdicts(response_text: str) -> list[tuple[str, str, str]]:
    """Parse VERDICT lines from QA agent response.

    Returns a list of (feature_id, verdict, reason) tuples.
    verdict is either "PASS" or "FAIL", reason is empty for PASS.
    """
    results: list[tuple[str, str, str]] = []
    for line in response_text.splitlines():
        line = line.strip()
        if not line.startswith("VERDICT:"):
            continue
        parts = line.split(":", maxsplit=3)
        # VERDICT:<id>:PASS or VERDICT:<id>:FAIL:<reason>
        if len(parts) >= 3:
            feat_id = parts[1]
            verdict = parts[2].upper()
            reason = parts[3] if len(parts) > 3 else ""
            if verdict in ("PASS", "FAIL"):
                results.append((feat_id, verdict, reason))
    return results
