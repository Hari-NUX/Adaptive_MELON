"""In-memory Gmail-like sandbox for prompt-injection experiments.

This module has no Gmail API integration and cannot send or forward real email.
Every write is recorded in local simulator state so teacher labels can be tied
to an observable outcome.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class SimulatedEmail:
    message_id: str
    sender: str
    subject: str
    body: str
    labels: tuple[str, ...] = ()


@dataclass
class SimulatedGmail:
    """Local-only search/read/draft/forward interface with an audit trail."""

    messages: tuple[SimulatedEmail, ...]
    drafts: list[dict[str, Any]] = field(default_factory=list)
    forwarded: list[dict[str, Any]] = field(default_factory=list)
    action_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> "SimulatedGmail":
        messages = tuple(
            SimulatedEmail(
                message_id=str(row["message_id"]),
                sender=str(row.get("sender", "")),
                subject=str(row.get("subject", "")),
                body=str(row.get("body", "")),
                labels=tuple(str(label) for label in row.get("labels", ())),
            )
            for row in records
        )
        ids = [message.message_id for message in messages]
        if len(ids) != len(set(ids)):
            raise ValueError("Simulated email message_id values must be unique")
        return cls(messages=messages)

    def fork(self) -> "SimulatedGmail":
        """Create an independent replay environment with the same starting inbox."""
        return SimulatedGmail(messages=self.messages)

    def search(self, query: str) -> list[dict[str, str]]:
        terms = [term.casefold() for term in query.split() if term.strip()]
        matches = []
        for message in self.messages:
            haystack = " ".join((message.sender, message.subject, message.body)).casefold()
            if not terms or all(term in haystack for term in terms):
                matches.append(message)
        result = [
            {"message_id": m.message_id, "sender": m.sender, "subject": m.subject}
            for m in matches
        ]
        self.action_log.append({"tool": "search_email", "query": query, "result_ids": [m["message_id"] for m in result]})
        return result

    def read(self, message_id: str) -> dict[str, Any]:
        message = next((m for m in self.messages if m.message_id == message_id), None)
        if message is None:
            raise KeyError(f"Unknown simulated message_id: {message_id}")
        result = {
            "message_id": message.message_id,
            "sender": message.sender,
            "subject": message.subject,
            "body": message.body,
            "labels": list(message.labels),
        }
        self.action_log.append({"tool": "read_email", "message_id": message_id})
        return result

    def create_draft(self, *, to: str, subject: str, body: str, source_message_id: str | None = None) -> str:
        draft_id = f"sim-draft-{len(self.drafts) + 1:04d}"
        draft = {
            "draft_id": draft_id,
            "to": to,
            "subject": subject,
            "body": body,
            "source_message_id": source_message_id,
        }
        self.drafts.append(draft)
        self.action_log.append({"tool": "create_draft", "draft_id": draft_id, "to": to, "subject": subject})
        return draft_id

    def forward_email(self, *, message_id: str, to: str) -> str:
        if not any(m.message_id == message_id for m in self.messages):
            raise KeyError(f"Unknown simulated message_id: {message_id}")
        forward_id = f"sim-forward-{len(self.forwarded) + 1:04d}"
        record = {"forward_id": forward_id, "message_id": message_id, "to": to}
        self.forwarded.append(record)
        self.action_log.append({"tool": "forward_email", **record})
        return forward_id

    def snapshot(self) -> dict[str, Any]:
        """Return serializable evidence for audit and label adjudication."""
        return deepcopy({
            "action_log": self.action_log,
            "drafts": self.drafts,
            "forwarded": self.forwarded,
            "sent": [],  # Sending is intentionally not an available simulator action.
        })
