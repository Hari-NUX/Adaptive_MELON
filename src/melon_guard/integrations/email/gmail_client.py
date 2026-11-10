"""Read-only Gmail API client for the local MELON assistant."""
from __future__ import annotations

import base64
import re
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Any


class GmailClient:
    """Search/read Gmail; optionally create unsent drafts after user approval.

    This client deliberately exposes no send, delete, archive, or modify method.
    """

    READ_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
    DRAFT_SCOPES = READ_SCOPES + ["https://www.googleapis.com/auth/gmail.compose"]

    def __init__(self, credentials_file: Path, token_file: Path, *, allow_drafts: bool = False):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise ImportError(
                "Install Gmail support with: py -m pip install -r requirements-gmail.txt"
            ) from exc

        credentials_file = Path(credentials_file).expanduser()
        token_file = Path(token_file).expanduser()
        scopes = self.DRAFT_SCOPES if allow_drafts else self.READ_SCOPES
        if not credentials_file.is_file():
            raise FileNotFoundError(
                f"OAuth client file not found: {credentials_file}. See docs/gmail_agent.md."
            )

        token_file.parent.mkdir(parents=True, exist_ok=True)
        creds = None
        if token_file.is_file():
            creds = Credentials.from_authorized_user_file(str(token_file), scopes)
            if not creds.has_scopes(scopes):
                creds = None
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_file), scopes
                )
                creds = flow.run_local_server(port=0)
            token_file.write_text(creds.to_json(), encoding="utf-8")

        self.service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        self.allow_drafts = allow_drafts

    def create_draft(self, to: str, subject: str, body: str) -> str:
        """Create an unsent Gmail draft. There is intentionally no send method."""
        if not self.allow_drafts:
            raise PermissionError("Draft creation requires explicit --create-draft mode.")
        to = to.strip()
        address = parseaddr(to)[1]
        if not re.fullmatch(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+", address):
            raise ValueError("One valid recipient email address is required.")
        if not subject.strip() or "\r" in subject or "\n" in subject or not body.strip():
            raise ValueError("Draft subject and body must not be empty.")
        message = EmailMessage()
        message["To"] = to
        message["Subject"] = subject.strip()
        message.set_content(body)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
        created = self.service.users().drafts().create(
            userId="me", body={"message": {"raw": raw}}
        ).execute()
        return str(created["id"])

    def search_messages(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return metadata and plain-text bodies for the most recent matches."""
        if not query.strip():
            raise ValueError("Gmail search query must not be empty.")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        ids: list[str] = []
        page_token = None
        while len(ids) < limit:
            parameters = {
                "userId": "me",
                "q": query,
                "maxResults": min(100, limit - len(ids)),
            }
            if page_token:
                parameters["pageToken"] = page_token
            response = self.service.users().messages().list(**parameters).execute()
            ids.extend(item["id"] for item in response.get("messages", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        messages = []
        for message_id in ids[:limit]:
            message = self.service.users().messages().get(
                userId="me", id=message_id, format="full"
            ).execute()
            payload = message.get("payload", {})
            headers = {
                str(header.get("name", "")).lower(): str(header.get("value", ""))
                for header in payload.get("headers", [])
            }
            messages.append({
                "id": message.get("id", ""),
                "thread_id": message.get("threadId", ""),
                "from": headers.get("from", ""),
                "to": headers.get("to", ""),
                "subject": headers.get("subject", "(no subject)"),
                "date": headers.get("date", ""),
                "body": self._plain_text(payload),
            })
        return messages

    @classmethod
    def _plain_text(cls, payload: dict[str, Any]) -> str:
        plain_parts: list[str] = []
        html_parts: list[str] = []

        def visit(part: dict[str, Any]) -> None:
            if part.get("filename"):
                return
            mime = str(part.get("mimeType", "")).lower()
            data = part.get("body", {}).get("data")
            if data and mime in {"text/plain", "text/html"}:
                raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
                decoded = raw.decode("utf-8", errors="replace")
                (plain_parts if mime == "text/plain" else html_parts).append(decoded)
            for child in part.get("parts", []) or []:
                visit(child)

        visit(payload)
        if plain_parts:
            return "\n".join(plain_parts)
        if html_parts:
            html = "\n".join(html_parts)
            return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", html)).strip()
        return ""
