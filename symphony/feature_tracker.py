"""JSON persistence for feature decomposition lists.

Reads and writes ``features.json`` files inside workspace directories,
following the cc-session-sync pattern of flat JSON arrays.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from symphony.models import FeatureList, FeatureTask

_FILENAME = "features.json"

# Fields that may be updated after initial creation.
_MUTABLE_FIELDS = ("status", "attempts", "last_error")


class FeatureTracker:
    """Persists feature lists as JSON in workspace directories."""

    def __init__(self, workspace_root: str) -> None:
        self._root = workspace_root

    def _features_path(self, identifier: str) -> str:
        """Path to features.json for an issue."""
        return os.path.join(self._root, identifier, _FILENAME)

    def save(self, identifier: str, feature_list: FeatureList) -> None:
        """Write feature list to disk.

        If ``features.json`` already exists, only the mutable fields
        (``status``, ``attempts``, ``last_error``) of existing features
        are updated.  The immutable fields (``id``, ``description``,
        ``steps``, ``category``, ``test_command``, ``max_attempts``) are
        preserved from the file on disk.
        """
        path = self._features_path(identifier)

        # Build a lookup of the incoming features keyed by id.
        incoming = {f.id: asdict(f) for f in feature_list.features}

        # If the file already exists, merge mutable fields only.
        if os.path.exists(path):
            try:
                with open(path, "r") as fh:
                    existing_records: list[dict] = json.load(fh)
            except (json.JSONDecodeError, OSError):
                existing_records = []

            existing_by_id = {r["id"]: r for r in existing_records}

            merged: list[dict] = []
            for feat_id, new_data in incoming.items():
                if feat_id in existing_by_id:
                    # Start from the existing record (preserves immutable fields).
                    record = existing_by_id[feat_id].copy()
                    for field in _MUTABLE_FIELDS:
                        record[field] = new_data[field]
                    merged.append(record)
                else:
                    # Brand-new feature — store everything.
                    merged.append(new_data)

            records_to_write = merged
        else:
            records_to_write = list(incoming.values())

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(records_to_write, fh, indent=2)

    def load(self, identifier: str) -> FeatureList | None:
        """Load feature list from disk. Returns ``None`` if not found."""
        path = self._features_path(identifier)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r") as fh:
                records = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

        if not isinstance(records, list):
            return None

        features: list[FeatureTask] = []
        for rec in records:
            try:
                features.append(
                    FeatureTask(
                        id=rec["id"],
                        description=rec["description"],
                        status=rec.get("status", "pending"),
                        category=rec.get("category", "general"),
                        test_command=rec.get("test_command"),
                        steps=rec.get("steps", []),
                        attempts=rec.get("attempts", 0),
                        max_attempts=rec.get("max_attempts", 3),
                        last_error=rec.get("last_error", ""),
                    )
                )
            except (KeyError, TypeError):
                # Skip malformed entries rather than failing entirely.
                continue

        return FeatureList(issue_id=identifier, features=features)

    def exists(self, identifier: str) -> bool:
        """Check if features.json exists for this issue."""
        return os.path.exists(self._features_path(identifier))
