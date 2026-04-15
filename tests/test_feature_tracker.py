"""Tests for FeatureTask, FeatureList, and FeatureTracker."""

from __future__ import annotations

import json
import os

import pytest

from symphony.feature_tracker import FeatureTracker
from symphony.models import FeatureList, FeatureTask


# =========================================================================
# FeatureTask status transitions
# =========================================================================


class TestFeatureTaskStatusTransitions:
    def test_pending_to_in_progress_to_passed(self) -> None:
        task = FeatureTask(id="t1", description="do something")
        assert task.status == "pending"
        task.status = "in_progress"
        assert task.status == "in_progress"
        task.status = "passed"
        assert task.status == "passed"

    def test_pending_to_failed(self) -> None:
        task = FeatureTask(id="t2", description="might fail")
        assert task.status == "pending"
        task.status = "failed"
        assert task.status == "failed"


# =========================================================================
# FeatureList.all_passed
# =========================================================================


class TestFeatureListAllPassed:
    def test_all_passed_returns_true(self) -> None:
        fl = FeatureList(
            issue_id="issue-1",
            features=[
                FeatureTask(id="f1", description="a", status="passed"),
                FeatureTask(id="f2", description="b", status="passed"),
            ],
        )
        assert fl.all_passed() is True

    def test_some_pending_returns_false(self) -> None:
        fl = FeatureList(
            issue_id="issue-1",
            features=[
                FeatureTask(id="f1", description="a", status="passed"),
                FeatureTask(id="f2", description="b", status="pending"),
            ],
        )
        assert fl.all_passed() is False

    def test_empty_returns_false(self) -> None:
        fl = FeatureList(issue_id="issue-1", features=[])
        assert fl.all_passed() is False


# =========================================================================
# FeatureList.next_pending
# =========================================================================


class TestFeatureListNextPending:
    def test_returns_first_pending(self) -> None:
        fl = FeatureList(
            issue_id="issue-1",
            features=[
                FeatureTask(id="f1", description="a", status="passed"),
                FeatureTask(id="f2", description="b", status="pending"),
                FeatureTask(id="f3", description="c", status="pending"),
            ],
        )
        nxt = fl.next_pending()
        assert nxt is not None
        assert nxt.id == "f2"

    def test_returns_first_failed(self) -> None:
        fl = FeatureList(
            issue_id="issue-1",
            features=[
                FeatureTask(id="f1", description="a", status="passed"),
                FeatureTask(id="f2", description="b", status="failed"),
                FeatureTask(id="f3", description="c", status="pending"),
            ],
        )
        nxt = fl.next_pending()
        assert nxt is not None
        assert nxt.id == "f2"

    def test_returns_none_when_all_passed(self) -> None:
        fl = FeatureList(
            issue_id="issue-1",
            features=[
                FeatureTask(id="f1", description="a", status="passed"),
                FeatureTask(id="f2", description="b", status="passed"),
            ],
        )
        assert fl.next_pending() is None


# =========================================================================
# FeatureList.progress
# =========================================================================


class TestFeatureListProgress:
    def test_returns_correct_tuple(self) -> None:
        fl = FeatureList(
            issue_id="issue-1",
            features=[
                FeatureTask(id="f1", description="a", status="passed"),
                FeatureTask(id="f2", description="b", status="pending"),
                FeatureTask(id="f3", description="c", status="passed"),
                FeatureTask(id="f4", description="d", status="failed"),
            ],
        )
        assert fl.progress() == (2, 4)

    def test_empty_list(self) -> None:
        fl = FeatureList(issue_id="issue-1", features=[])
        assert fl.progress() == (0, 0)


# =========================================================================
# FeatureList.mark_passed / mark_failed
# =========================================================================


class TestFeatureListMarkPassedAndFailed:
    def test_mark_passed_updates_correct_feature(self) -> None:
        fl = FeatureList(
            issue_id="issue-1",
            features=[
                FeatureTask(id="f1", description="a", status="pending"),
                FeatureTask(id="f2", description="b", status="pending"),
            ],
        )
        fl.mark_passed("f1")
        assert fl.features[0].status == "passed"
        assert fl.features[1].status == "pending"

    def test_mark_failed_updates_correct_feature(self) -> None:
        fl = FeatureList(
            issue_id="issue-1",
            features=[
                FeatureTask(id="f1", description="a", status="pending"),
                FeatureTask(id="f2", description="b", status="pending"),
            ],
        )
        fl.mark_failed("f2", error="something broke")
        assert fl.features[1].status == "failed"
        assert fl.features[1].attempts == 1
        assert fl.features[1].last_error == "something broke"
        # First feature unchanged.
        assert fl.features[0].status == "pending"

    def test_mark_failed_increments_attempts(self) -> None:
        fl = FeatureList(
            issue_id="issue-1",
            features=[FeatureTask(id="f1", description="a", attempts=2)],
        )
        fl.mark_failed("f1", error="again")
        assert fl.features[0].attempts == 3


# =========================================================================
# FeatureTracker: save / load roundtrip
# =========================================================================


class TestTrackerSaveAndLoadRoundtrip:
    def test_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        tracker = FeatureTracker(workspace_root=str(tmp_path))
        fl = FeatureList(
            issue_id="issue-42",
            features=[
                FeatureTask(
                    id="feat-1",
                    description="add login form",
                    status="passed",
                    category="functional",
                    steps=["create form", "add validation"],
                    attempts=1,
                    last_error="",
                ),
                FeatureTask(
                    id="feat-2",
                    description="style header",
                    status="pending",
                    category="style",
                    test_command="npm test",
                ),
            ],
        )

        tracker.save("issue-42", fl)
        loaded = tracker.load("issue-42")

        assert loaded is not None
        assert loaded.issue_id == "issue-42"
        assert len(loaded.features) == 2

        f1 = loaded.features[0]
        assert f1.id == "feat-1"
        assert f1.description == "add login form"
        assert f1.status == "passed"
        assert f1.category == "functional"
        assert f1.steps == ["create form", "add validation"]
        assert f1.attempts == 1

        f2 = loaded.features[1]
        assert f2.id == "feat-2"
        assert f2.description == "style header"
        assert f2.status == "pending"
        assert f2.category == "style"
        assert f2.test_command == "npm test"


# =========================================================================
# FeatureTracker: load nonexistent
# =========================================================================


class TestTrackerLoadNonexistent:
    def test_returns_none(self, tmp_path: pytest.TempPathFactory) -> None:
        tracker = FeatureTracker(workspace_root=str(tmp_path))
        assert tracker.load("no-such-issue") is None


# =========================================================================
# FeatureTracker: exists
# =========================================================================


class TestTrackerExists:
    def test_true_when_file_exists(self, tmp_path: pytest.TempPathFactory) -> None:
        tracker = FeatureTracker(workspace_root=str(tmp_path))
        fl = FeatureList(
            issue_id="issue-1",
            features=[FeatureTask(id="f1", description="x")],
        )
        tracker.save("issue-1", fl)
        assert tracker.exists("issue-1") is True

    def test_false_when_not_exists(self, tmp_path: pytest.TempPathFactory) -> None:
        tracker = FeatureTracker(workspace_root=str(tmp_path))
        assert tracker.exists("issue-999") is False


# =========================================================================
# FeatureTracker: save merges with existing
# =========================================================================


class TestTrackerSaveMergesWithExisting:
    def test_merges_mutable_fields_only(self, tmp_path: pytest.TempPathFactory) -> None:
        tracker = FeatureTracker(workspace_root=str(tmp_path))

        # Initial save with original descriptions and steps.
        original = FeatureList(
            issue_id="issue-5",
            features=[
                FeatureTask(
                    id="feat-1",
                    description="original description",
                    status="pending",
                    category="functional",
                    steps=["step A", "step B"],
                ),
                FeatureTask(
                    id="feat-2",
                    description="second task",
                    status="pending",
                    category="api",
                ),
            ],
        )
        tracker.save("issue-5", original)

        # Second save: try to change description and steps (immutable),
        # and also change status (mutable).
        modified = FeatureList(
            issue_id="issue-5",
            features=[
                FeatureTask(
                    id="feat-1",
                    description="CHANGED description",
                    status="passed",
                    category="style",
                    steps=["CHANGED step"],
                    attempts=2,
                    last_error="some error",
                ),
                FeatureTask(
                    id="feat-2",
                    description="CHANGED second",
                    status="failed",
                    attempts=1,
                    last_error="oops",
                ),
            ],
        )
        tracker.save("issue-5", modified)

        loaded = tracker.load("issue-5")
        assert loaded is not None

        f1 = loaded.features[0]
        # Immutable fields should be preserved from the original save.
        assert f1.description == "original description"
        assert f1.steps == ["step A", "step B"]
        assert f1.category == "functional"
        # Mutable fields should be updated.
        assert f1.status == "passed"
        assert f1.attempts == 2
        assert f1.last_error == "some error"

        f2 = loaded.features[1]
        assert f2.description == "second task"
        assert f2.category == "api"
        assert f2.status == "failed"
        assert f2.attempts == 1
        assert f2.last_error == "oops"


# =========================================================================
# FeatureTracker: handles corrupt JSON
# =========================================================================


class TestTrackerHandlesCorruptJson:
    def test_malformed_json_returns_none(self, tmp_path: pytest.TempPathFactory) -> None:
        tracker = FeatureTracker(workspace_root=str(tmp_path))
        # Create the directory and write garbage to features.json.
        issue_dir = os.path.join(str(tmp_path), "bad-issue")
        os.makedirs(issue_dir, exist_ok=True)
        with open(os.path.join(issue_dir, "features.json"), "w") as fh:
            fh.write("{{{not valid json!!!")

        result = tracker.load("bad-issue")
        assert result is None
