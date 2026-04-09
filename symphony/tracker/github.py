"""GitHub Issues tracker adapter using httpx."""

from __future__ import annotations

import re
from typing import Any

import httpx

from symphony.models import Issue
from symphony.tracker.base import Tracker

_BLOCKED_RE = re.compile(r"blocked\s+by\s+#(\d+)", re.IGNORECASE)
_PRIORITY_RE = re.compile(r"^priority:(\d+)$")

_PAGE_SIZE = 100


class GitHubTracker(Tracker):
    """Adapter that maps GitHub Issues to the Symphony Tracker interface."""

    def __init__(
        self,
        *,
        owner: str,
        repo: str,
        token: str = "",
        candidate_states: list[str] | None = None,
        state_labels: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self._candidate_states = candidate_states or ["open"]
        # Mapping: Symphony state name -> GitHub label name.
        self._state_labels = state_labels or {}
        self._client = client or httpx.AsyncClient(
            base_url="https://api.github.com",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    # -- internal helpers --------------------------------------------------

    def _repo_url(self, path: str = "") -> str:
        return f"/repos/{self.owner}/{self.repo}{path}"

    def _label_for_state(self, state: str) -> str:
        return self._state_labels.get(state, state)

    def _state_for_label(self, label: str) -> str | None:
        for state, lbl in self._state_labels.items():
            if lbl == label:
                return state
        return label

    @staticmethod
    def _extract_priority(labels: list[str]) -> int | None:
        for lbl in labels:
            m = _PRIORITY_RE.match(lbl)
            if m:
                return int(m.group(1))
        return None

    @staticmethod
    def _extract_blocked_by(body: str) -> list[str]:
        return _BLOCKED_RE.findall(body or "")

    def _issue_from_gh(self, data: dict[str, Any]) -> Issue:
        label_names = [lbl["name"] for lbl in data.get("labels", [])]
        body = data.get("body") or ""
        assignee_data = data.get("assignee")
        assignee = assignee_data["login"] if assignee_data else None

        # Determine symphony state from labels.
        state = data.get("state", "open")
        for lbl in label_names:
            s = self._state_for_label(lbl)
            if s and s in self._candidate_states:
                state = s
                break

        from symphony.models import BlockerInfo

        blocked_ids = self._extract_blocked_by(body)
        blockers = [
            BlockerInfo(id=bid, identifier=bid, state="unknown")
            for bid in blocked_ids
        ]

        return Issue(
            id=str(data["number"]),
            identifier=str(data["number"]),
            title=data.get("title", ""),
            description=body,
            state=state,
            branch_name=f"symphony/{data['number']}",
            url=data.get("html_url", ""),
            assignee_id=assignee,
            priority=self._extract_priority(label_names),
            blocked_by=blockers,
            labels=label_names,
        )

    async def _paginated_get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = dict(params or {})
        params.setdefault("per_page", _PAGE_SIZE)
        page = 1
        all_items: list[dict[str, Any]] = []
        while True:
            params["page"] = page
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            items = resp.json()
            if not items:
                break
            all_items.extend(items)
            if len(items) < _PAGE_SIZE:
                break
            page += 1
        return all_items

    # -- Tracker interface -------------------------------------------------

    async def fetch_candidate_issues(self) -> list[Issue]:
        all_issues: list[Issue] = []
        for state in self._candidate_states:
            label = self._label_for_state(state)
            items = await self._paginated_get(
                self._repo_url("/issues"),
                params={"labels": label, "state": "all"},
            )
            all_issues.extend(self._issue_from_gh(item) for item in items)
        # Deduplicate by identifier.
        seen: set[str] = set()
        deduped: list[Issue] = []
        for iss in all_issues:
            if iss.identifier not in seen:
                seen.add(iss.identifier)
                deduped.append(iss)
        return deduped

    async def fetch_issues_by_states(self, states: list[str]) -> list[Issue]:
        if not states:
            return []
        all_issues: list[Issue] = []
        for state in states:
            label = self._label_for_state(state)
            items = await self._paginated_get(
                self._repo_url("/issues"),
                params={"labels": label, "state": "all"},
            )
            all_issues.extend(self._issue_from_gh(item) for item in items)
        seen: set[str] = set()
        deduped: list[Issue] = []
        for iss in all_issues:
            if iss.identifier not in seen:
                seen.add(iss.identifier)
                deduped.append(iss)
        return deduped

    async def fetch_issue_states_by_ids(
        self, identifiers: list[str]
    ) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        for ident in identifiers:
            resp = await self._client.get(self._repo_url(f"/issues/{ident}"))
            if resp.status_code == 200:
                data = resp.json()
                issue = self._issue_from_gh(data)
                result[ident] = issue.state
            else:
                result[ident] = None
        return result

    async def create_comment(self, identifier: str, body: str) -> None:
        resp = await self._client.post(
            self._repo_url(f"/issues/{identifier}/comments"),
            json={"body": body},
        )
        resp.raise_for_status()

    async def update_issue_state(self, identifier: str, state: str) -> None:
        label = self._label_for_state(state)
        # Fetch current labels to swap state labels.
        resp = await self._client.get(self._repo_url(f"/issues/{identifier}"))
        resp.raise_for_status()
        data = resp.json()
        current_labels = [lbl["name"] for lbl in data.get("labels", [])]

        # Remove existing state labels.
        all_state_labels = set(self._state_labels.values()) | set(
            self._candidate_states
        )
        new_labels = [l for l in current_labels if l not in all_state_labels]
        new_labels.append(label)

        gh_state = "closed" if state == "closed" else "open"

        resp = await self._client.patch(
            self._repo_url(f"/issues/{identifier}"),
            json={"labels": new_labels, "state": gh_state},
        )
        resp.raise_for_status()
