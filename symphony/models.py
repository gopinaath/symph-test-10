"""Data models for Symphony issues and related entities."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BlockerInfo:
    """Reference to an issue that blocks another issue."""

    id: str
    identifier: str
    state: str


@dataclass
class Issue:
    """Represents a tracked work item (issue/ticket).

    Priority follows Linear conventions: 1 = urgent, 4 = low.
    ``None`` means no priority (treated as 5 in sort order).
    """

    id: str
    identifier: str
    title: str
    description: str
    priority: Optional[int]  # 1-4, None = no priority (treated as 5)
    state: str
    branch_name: str
    url: str
    assignee_id: Optional[str]
    blocked_by: list[BlockerInfo] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    assigned_to_worker: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for template rendering.

        DateTime values are converted to ISO 8601 strings.
        """
        result: dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, datetime):
                result[k] = v.isoformat()
            elif isinstance(v, list) and v and isinstance(v[0], BlockerInfo):
                result[k] = [b.__dict__.copy() for b in v]
            else:
                result[k] = v
        return result
