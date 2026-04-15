"""Tests for two-phase agent pattern (Phase 2d) and QA agent (Phase 2e)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from symphony.agent_runner import (
    AgentRunner,
    AgentRunnerConfig,
    RunResult,
    _parse_feature_list,
    _parse_qa_verdicts,
)
from symphony.codex.app_server import AppServer
from symphony.config import AgentConfig, QAConfig, ValidationConfig
from symphony.feature_tracker import FeatureTracker
from symphony.models import FeatureList, FeatureTask
from symphony.workspace import ValidationResult

# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------


@dataclass
class FakeIssue:
    id: str = "ISSUE-1"
    title: str = "Fix the bug"
    state: str = "active"

    def is_terminal(self) -> bool:
        return self.state == "terminal"


@dataclass
class FakeWorkspace:
    path: str = "/tmp/workspace"
    prepared: bool = False
    cleaned: bool = False
    _prepare_error: Exception | None = None
    _validation_results: list[ValidationResult] | None = None
    _validation_call_count: int = 0

    async def prepare(self) -> None:
        if self._prepare_error:
            raise self._prepare_error
        self.prepared = True

    async def cleanup(self) -> None:
        self.cleaned = True

    async def run_validation(self, identifier: str, config: ValidationConfig) -> ValidationResult:
        if self._validation_results is None:
            return ValidationResult(passed=True, exit_code=0)
        idx = min(self._validation_call_count, len(self._validation_results) - 1)
        result = self._validation_results[idx]
        self._validation_call_count += 1
        return result


# ---------------------------------------------------------------------------
# App server helpers
# ---------------------------------------------------------------------------


def _fake_app_server(
    *,
    session_id: str = "session-42",
    turn_usage: dict[str, Any] | None = None,
    turn_responses: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a mock AppServer that successfully completes turns."""
    server = AsyncMock(spec=AppServer)
    server.start = AsyncMock()
    server.start_thread = AsyncMock(return_value=session_id)

    if turn_responses is not None:
        server.run_turn = AsyncMock(side_effect=turn_responses)
    else:
        server.run_turn = AsyncMock(
            return_value={"turn_id": "t-1", "usage": turn_usage or {"input_tokens": 5}}
        )

    server.stop = AsyncMock()
    return server


def _factory_returning(server: Any):
    """Return a factory callable that always yields *server*."""

    def factory(config=None, on_event=None):
        server._on_event = on_event
        return server

    return factory


def _make_feature_json(features: list[dict[str, Any]]) -> str:
    """Build a JSON response string containing a feature array."""
    return json.dumps(features)


# ---------------------------------------------------------------------------
# Two-agent pattern tests (Phase 2d)
# ---------------------------------------------------------------------------


class TestSingleModeIsDefault:
    """test_single_mode_is_default: verify default agent_mode='single'."""

    def test_single_mode_is_default(self):
        config = AgentConfig()
        assert config.agent_mode == "single"

    def test_agent_runner_defaults_to_single(self):
        runner = AgentRunner()
        assert runner._agent_mode == "single"


class TestTwoPhaseInitializerGeneratesFeatureList:
    """test_two_phase_initializer_generates_feature_list: mock agent returns
    JSON feature list, verify FeatureList created."""

    @pytest.mark.asyncio
    async def test_two_phase_initializer_generates_feature_list(self, tmp_path):
        features_json = _make_feature_json([
            {
                "id": "feat-1",
                "description": "Add login form",
                "category": "functional",
                "steps": ["Create form component", "Add validation"],
                "test_command": "pytest test_login.py",
            },
            {
                "id": "feat-2",
                "description": "Add logout button",
                "category": "ui",
                "steps": ["Add button to navbar"],
                "test_command": None,
            },
        ])

        # First call is decomposition turn (returns features JSON).
        # Subsequent calls are coder turns.
        initializer_response = {
            "turn_id": "t-init",
            "usage": {"input_tokens": 10},
            "response": features_json,
        }
        coder_response = {
            "turn_id": "t-code",
            "usage": {"input_tokens": 5},
        }

        server = _fake_app_server(turn_responses=[
            initializer_response, coder_response, coder_response,
        ])

        tracker = FeatureTracker(str(tmp_path))
        issue = FakeIssue()
        ws = FakeWorkspace(path=str(tmp_path / issue.id))
        os.makedirs(ws.path, exist_ok=True)

        cfg = AgentRunnerConfig(max_turns=10)
        runner = AgentRunner(
            cfg,
            app_server_factory=_factory_returning(server),
            agent_mode="two_phase",
            feature_tracker=tracker,
        )
        result = await runner.run(issue, ws)

        # Verify feature list was created and saved.
        loaded = tracker.load(issue.id)
        assert loaded is not None
        assert len(loaded.features) == 2
        assert loaded.features[0].id == "feat-1"
        assert loaded.features[1].id == "feat-2"

        # Both features should be passed (no validation configured).
        assert result.features_total == 2
        assert result.features_completed == 2


class TestTwoPhaseCoderIteratesPerFeature:
    """test_two_phase_coder_iterates_per_feature: mock agent + validation,
    verify features marked pass/fail."""

    @pytest.mark.asyncio
    async def test_two_phase_coder_iterates_per_feature(self, tmp_path):
        features_json = _make_feature_json([
            {"id": "f1", "description": "Feature 1", "category": "general",
             "steps": ["step1"], "test_command": "pytest"},
            {"id": "f2", "description": "Feature 2", "category": "general",
             "steps": ["step2"], "test_command": "pytest"},
        ])

        responses = [
            # Initializer
            {"turn_id": "t-init", "usage": {"input_tokens": 10}, "response": features_json},
            # Coder turn for f1 (validation passes)
            {"turn_id": "t-1", "usage": {"input_tokens": 5}},
            # Coder turn for f2 (validation fails)
            {"turn_id": "t-2", "usage": {"input_tokens": 5}},
        ]
        server = _fake_app_server(turn_responses=responses)

        tracker = FeatureTracker(str(tmp_path))
        issue = FakeIssue()
        ws = FakeWorkspace(
            path=str(tmp_path / issue.id),
            _validation_results=[
                ValidationResult(passed=True, exit_code=0, stdout="pass"),
                ValidationResult(passed=False, exit_code=1, stderr="FAIL"),
            ],
        )
        os.makedirs(ws.path, exist_ok=True)

        v_config = ValidationConfig(enabled=True, command="pytest")
        # max_turns=3: 1 init + 2 coder, so f2 fails and no more turns to retry.
        cfg = AgentRunnerConfig(max_turns=3)
        runner = AgentRunner(
            cfg,
            app_server_factory=_factory_returning(server),
            agent_mode="two_phase",
            feature_tracker=tracker,
        )
        result = await runner.run(issue, ws, validation_config=v_config)

        loaded = tracker.load(issue.id)
        assert loaded is not None
        # f1 should pass, f2 should fail.
        f1 = next(f for f in loaded.features if f.id == "f1")
        f2 = next(f for f in loaded.features if f.id == "f2")
        assert f1.status == "passed"
        assert f2.status == "failed"


class TestTwoPhasePartialCompletionTracked:
    """test_two_phase_partial_completion_tracked: 3 features, 2 pass, 1 fails,
    verify RunResult.features_completed=2, features_total=3."""

    @pytest.mark.asyncio
    async def test_two_phase_partial_completion_tracked(self, tmp_path):
        features_json = _make_feature_json([
            {"id": "f1", "description": "F1", "category": "g", "steps": ["s1"]},
            {"id": "f2", "description": "F2", "category": "g", "steps": ["s2"]},
            {"id": "f3", "description": "F3", "category": "g", "steps": ["s3"]},
        ])

        responses = [
            {"turn_id": "t-init", "usage": {"input_tokens": 10}, "response": features_json},
            {"turn_id": "t-1", "usage": {"input_tokens": 5}},  # f1
            {"turn_id": "t-2", "usage": {"input_tokens": 5}},  # f2
            {"turn_id": "t-3", "usage": {"input_tokens": 5}},  # f3
        ]
        server = _fake_app_server(turn_responses=responses)

        tracker = FeatureTracker(str(tmp_path))
        issue = FakeIssue()
        ws = FakeWorkspace(
            path=str(tmp_path / issue.id),
            _validation_results=[
                ValidationResult(passed=True, exit_code=0),   # f1 passes
                ValidationResult(passed=True, exit_code=0),   # f2 passes
                ValidationResult(passed=False, exit_code=1, stderr="f3 fail"),  # f3 fails
            ],
        )
        os.makedirs(ws.path, exist_ok=True)

        v_config = ValidationConfig(enabled=True, command="pytest")
        # max_turns=4: 1 init + 3 coder. f3 fails but no more turns to retry.
        cfg = AgentRunnerConfig(max_turns=4)
        runner = AgentRunner(
            cfg,
            app_server_factory=_factory_returning(server),
            agent_mode="two_phase",
            feature_tracker=tracker,
        )
        result = await runner.run(issue, ws, validation_config=v_config)

        assert result.features_completed == 2
        assert result.features_total == 3


class TestTwoPhaseResumesFromExistingFeatures:
    """test_two_phase_resumes_from_existing_features: pre-populate features.json
    with 1 passed, verify coder skips it."""

    @pytest.mark.asyncio
    async def test_two_phase_resumes_from_existing_features(self, tmp_path):
        # Pre-populate with f1 already passed.
        tracker = FeatureTracker(str(tmp_path))
        issue = FakeIssue()
        pre_existing = FeatureList(
            issue_id=issue.id,
            features=[
                FeatureTask(id="f1", description="F1", status="passed", steps=["s1"]),
                FeatureTask(id="f2", description="F2", status="pending", steps=["s2"]),
            ],
        )
        os.makedirs(tmp_path / issue.id, exist_ok=True)
        tracker.save(issue.id, pre_existing)

        # Only one coder turn needed (for f2), no initializer turn.
        coder_response = {"turn_id": "t-code", "usage": {"input_tokens": 5}}
        server = _fake_app_server(turn_responses=[coder_response])

        ws = FakeWorkspace(path=str(tmp_path / issue.id))
        cfg = AgentRunnerConfig(max_turns=10)
        runner = AgentRunner(
            cfg,
            app_server_factory=_factory_returning(server),
            agent_mode="two_phase",
            feature_tracker=tracker,
        )
        result = await runner.run(issue, ws)

        # Only 1 turn should execute (for f2), not 2.
        assert result.turns_executed == 1
        assert server.run_turn.call_count == 1

        # f1 should still be passed.
        loaded = tracker.load(issue.id)
        assert loaded is not None
        f1 = next(f for f in loaded.features if f.id == "f1")
        assert f1.status == "passed"

        assert result.features_completed == 2
        assert result.features_total == 2


class TestTwoPhaseMaxTurnsRespected:
    """test_two_phase_max_turns_respected: verify turn limit stops iteration
    even with pending features."""

    @pytest.mark.asyncio
    async def test_two_phase_max_turns_respected(self, tmp_path):
        features_json = _make_feature_json([
            {"id": "f1", "description": "F1", "category": "g", "steps": ["s1"]},
            {"id": "f2", "description": "F2", "category": "g", "steps": ["s2"]},
            {"id": "f3", "description": "F3", "category": "g", "steps": ["s3"]},
        ])

        responses = [
            # Initializer (turn 1).
            {"turn_id": "t-init", "usage": {"input_tokens": 10}, "response": features_json},
            # Coder turn for f1 (turn 2 = max_turns).
            {"turn_id": "t-1", "usage": {"input_tokens": 5}},
        ]
        server = _fake_app_server(turn_responses=responses)

        tracker = FeatureTracker(str(tmp_path))
        issue = FakeIssue()
        ws = FakeWorkspace(path=str(tmp_path / issue.id))
        os.makedirs(ws.path, exist_ok=True)

        # max_turns=2: 1 for decomposition + 1 for coding.
        cfg = AgentRunnerConfig(max_turns=2)
        runner = AgentRunner(
            cfg,
            app_server_factory=_factory_returning(server),
            agent_mode="two_phase",
            feature_tracker=tracker,
        )
        result = await runner.run(issue, ws)

        # Only 2 turns: init + 1 coder. f2 and f3 still pending.
        assert result.turns_executed == 2
        assert result.features_completed == 1  # only f1 passed
        assert result.features_total == 3
        assert result.stopped_reason == "max_turns"


# ---------------------------------------------------------------------------
# QA agent tests (Phase 2e)
# ---------------------------------------------------------------------------


class TestQADisabledByDefault:
    """test_qa_disabled_by_default: verify qa.enabled=False skips QA."""

    def test_qa_disabled_by_default(self):
        qa = QAConfig()
        assert qa.enabled is False

    @pytest.mark.asyncio
    async def test_qa_not_run_when_disabled(self):
        issue = FakeIssue()
        ws = FakeWorkspace()
        server = _fake_app_server()
        cfg = AgentRunnerConfig(max_turns=1)

        runner = AgentRunner(
            cfg,
            app_server_factory=_factory_returning(server),
            qa_config=QAConfig(enabled=False),
        )
        result = await runner.run(issue, ws)

        assert result.qa_passed is False
        assert result.qa_findings == ""
        # Only the single-mode turn, no QA turns.
        assert result.turns_executed == 1


class TestQAResetsFeauresWithoutVerdict:
    """test_qa_resets_features_without_verdict: mock QA that only verdicts 2 of 3
    features, verify third reset to failed."""

    @pytest.mark.asyncio
    async def test_qa_resets_features_without_verdict(self, tmp_path):
        # Set up pre-existing feature list with all 3 passed.
        tracker = FeatureTracker(str(tmp_path))
        issue = FakeIssue()
        feature_list = FeatureList(
            issue_id=issue.id,
            features=[
                FeatureTask(id="f1", description="F1", status="passed", steps=["s1"]),
                FeatureTask(id="f2", description="F2", status="passed", steps=["s2"]),
                FeatureTask(id="f3", description="F3", status="passed", steps=["s3"]),
            ],
        )
        os.makedirs(tmp_path / issue.id, exist_ok=True)
        tracker.save(issue.id, feature_list)

        # QA response only gives verdicts for f1 and f2, not f3.
        qa_response = (
            "VERDICT:f1:PASS\n"
            "VERDICT:f2:PASS\n"
        )

        # Two servers: one for the main run (single mode), one for QA.
        main_server = _fake_app_server()
        qa_server = _fake_app_server(turn_responses=[
            {"turn_id": "t-qa", "usage": {"input_tokens": 3}, "response": qa_response},
        ])

        # Factory returns main_server first, qa_server second.
        server_sequence = iter([main_server, qa_server])

        def factory(config=None, on_event=None):
            srv = next(server_sequence)
            srv._on_event = on_event
            return srv

        ws = FakeWorkspace(path=str(tmp_path / issue.id))
        cfg = AgentRunnerConfig(max_turns=1)
        runner = AgentRunner(
            cfg,
            app_server_factory=factory,
            qa_config=QAConfig(enabled=True, max_turns=1),
            feature_tracker=tracker,
        )
        result = await runner.run(issue, ws)

        # f3 should have been reset to failed.
        loaded = tracker.load(issue.id)
        assert loaded is not None
        f3 = next(f for f in loaded.features if f.id == "f3")
        assert f3.status == "failed"
        assert "No QA verdict" in f3.last_error

        # f1, f2 should still be passed.
        f1 = next(f for f in loaded.features if f.id == "f1")
        f2 = next(f for f in loaded.features if f.id == "f2")
        assert f1.status == "passed"
        assert f2.status == "passed"


class TestQAFindingsSavedToWorkspace:
    """test_qa_findings_saved_to_workspace: verify qa-findings.txt written."""

    @pytest.mark.asyncio
    async def test_qa_findings_saved_to_workspace(self, tmp_path):
        tracker = FeatureTracker(str(tmp_path))
        issue = FakeIssue()
        feature_list = FeatureList(
            issue_id=issue.id,
            features=[
                FeatureTask(id="f1", description="F1", status="passed", steps=["s1"]),
            ],
        )
        os.makedirs(tmp_path / issue.id, exist_ok=True)
        tracker.save(issue.id, feature_list)

        qa_response = "VERDICT:f1:PASS\n"

        main_server = _fake_app_server()
        qa_server = _fake_app_server(turn_responses=[
            {"turn_id": "t-qa", "usage": {"input_tokens": 3}, "response": qa_response},
        ])

        server_sequence = iter([main_server, qa_server])

        def factory(config=None, on_event=None):
            srv = next(server_sequence)
            srv._on_event = on_event
            return srv

        ws_path = str(tmp_path / issue.id)
        ws = FakeWorkspace(path=ws_path)
        cfg = AgentRunnerConfig(max_turns=1)
        runner = AgentRunner(
            cfg,
            app_server_factory=factory,
            qa_config=QAConfig(enabled=True, max_turns=1),
            feature_tracker=tracker,
        )
        result = await runner.run(issue, ws)

        findings_path = os.path.join(ws_path, "qa-findings.txt")
        assert os.path.exists(findings_path)
        content = open(findings_path).read()
        assert "f1" in content


class TestQAUsesRestrictedApprovalPolicy:
    """test_qa_uses_restricted_approval_policy: verify QA session uses 'suggest'
    not 'never'."""

    @pytest.mark.asyncio
    async def test_qa_uses_restricted_approval_policy(self, tmp_path):
        tracker = FeatureTracker(str(tmp_path))
        issue = FakeIssue()
        feature_list = FeatureList(
            issue_id=issue.id,
            features=[
                FeatureTask(id="f1", description="F1", status="passed", steps=["s1"]),
            ],
        )
        os.makedirs(tmp_path / issue.id, exist_ok=True)
        tracker.save(issue.id, feature_list)

        qa_response = "VERDICT:f1:PASS\n"

        main_server = _fake_app_server()
        qa_server = _fake_app_server(turn_responses=[
            {"turn_id": "t-qa", "usage": {"input_tokens": 3}, "response": qa_response},
        ])

        server_sequence = iter([main_server, qa_server])

        def factory(config=None, on_event=None):
            srv = next(server_sequence)
            srv._on_event = on_event
            return srv

        ws = FakeWorkspace(path=str(tmp_path / issue.id))
        cfg = AgentRunnerConfig(max_turns=1)
        runner = AgentRunner(
            cfg,
            app_server_factory=factory,
            qa_config=QAConfig(enabled=True, max_turns=1, approval_policy="suggest"),
            feature_tracker=tracker,
        )
        result = await runner.run(issue, ws)

        # Verify QA server was started with "suggest" policy.
        qa_server.start_thread.assert_called_once()
        call_kwargs = qa_server.start_thread.call_args
        # start_thread is called with (workspace.path, approval_policy="suggest")
        assert call_kwargs.kwargs.get("approval_policy") == "suggest" or \
            (len(call_kwargs.args) >= 1 and call_kwargs.kwargs.get("approval_policy") == "suggest")

        # Verify run_turn also used "suggest" policy.
        for call in qa_server.run_turn.call_args_list:
            assert call.kwargs.get("approval_policy") == "suggest"


class TestQACannotMarkFeaturesAsPassed:
    """test_qa_cannot_mark_features_as_passed: verify QA can only fail features,
    not pass them (failed->passed is not allowed during QA)."""

    @pytest.mark.asyncio
    async def test_qa_cannot_mark_features_as_passed(self, tmp_path):
        tracker = FeatureTracker(str(tmp_path))
        issue = FakeIssue()
        # f1 is passed (QA should verify), f2 is failed (QA should NOT be able to pass it).
        feature_list = FeatureList(
            issue_id=issue.id,
            features=[
                FeatureTask(id="f1", description="F1", status="passed", steps=["s1"]),
                FeatureTask(id="f2", description="F2", status="failed", steps=["s2"]),
            ],
        )
        os.makedirs(tmp_path / issue.id, exist_ok=True)
        tracker.save(issue.id, feature_list)

        # QA tries to pass both f1 and f2.
        qa_response = (
            "VERDICT:f1:PASS\n"
            "VERDICT:f2:PASS\n"
        )

        main_server = _fake_app_server()
        qa_server = _fake_app_server(turn_responses=[
            {"turn_id": "t-qa", "usage": {"input_tokens": 3}, "response": qa_response},
        ])

        server_sequence = iter([main_server, qa_server])

        def factory(config=None, on_event=None):
            srv = next(server_sequence)
            srv._on_event = on_event
            return srv

        ws = FakeWorkspace(path=str(tmp_path / issue.id))
        cfg = AgentRunnerConfig(max_turns=1)
        runner = AgentRunner(
            cfg,
            app_server_factory=factory,
            qa_config=QAConfig(enabled=True, max_turns=1),
            feature_tracker=tracker,
        )
        result = await runner.run(issue, ws)

        # f2 was already failed before QA, QA cannot pass it back.
        # QA only looks at passed features, so f2 stays failed.
        loaded = tracker.load(issue.id)
        assert loaded is not None
        f2 = next(f for f in loaded.features if f.id == "f2")
        assert f2.status == "failed"

        # f1 was passed and QA gave PASS, so it stays passed.
        f1 = next(f for f in loaded.features if f.id == "f1")
        assert f1.status == "passed"


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestParseFeatureList:
    """Unit tests for _parse_feature_list."""

    def test_parses_valid_json_array(self):
        response = json.dumps([
            {"id": "f1", "description": "desc1", "category": "ui",
             "steps": ["s1"], "test_command": "pytest"},
        ])
        result = _parse_feature_list("issue-1", response)
        assert result is not None
        assert len(result.features) == 1
        assert result.features[0].id == "f1"

    def test_returns_none_for_invalid_json(self):
        result = _parse_feature_list("issue-1", "not json at all")
        assert result is None

    def test_extracts_json_from_markdown(self):
        response = 'Here is the list:\n```json\n[{"id": "f1", "description": "d1", "steps": []}]\n```'
        result = _parse_feature_list("issue-1", response)
        assert result is not None
        assert len(result.features) == 1


class TestParseQAVerdicts:
    """Unit tests for _parse_qa_verdicts."""

    def test_parses_pass_verdict(self):
        verdicts = _parse_qa_verdicts("VERDICT:f1:PASS\n")
        assert len(verdicts) == 1
        assert verdicts[0] == ("f1", "PASS", "")

    def test_parses_fail_verdict_with_reason(self):
        verdicts = _parse_qa_verdicts("VERDICT:f1:FAIL:test failed\n")
        assert len(verdicts) == 1
        assert verdicts[0] == ("f1", "FAIL", "test failed")

    def test_parses_multiple_verdicts(self):
        text = "VERDICT:f1:PASS\nSome other text\nVERDICT:f2:FAIL:broken\n"
        verdicts = _parse_qa_verdicts(text)
        assert len(verdicts) == 2

    def test_ignores_invalid_lines(self):
        text = "This is not a verdict\nVERDICT:f1:INVALID\nVERDICT:f2:PASS\n"
        verdicts = _parse_qa_verdicts(text)
        assert len(verdicts) == 1
        assert verdicts[0][0] == "f2"


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestAgentModeValidation:
    """Verify agent_mode validator rejects invalid values."""

    def test_valid_single(self):
        config = AgentConfig(agent_mode="single")
        assert config.agent_mode == "single"

    def test_valid_two_phase(self):
        config = AgentConfig(agent_mode="two_phase")
        assert config.agent_mode == "two_phase"

    def test_invalid_mode_rejected(self):
        with pytest.raises(Exception):
            AgentConfig(agent_mode="invalid")
