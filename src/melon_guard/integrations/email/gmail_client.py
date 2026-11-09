"""Read-only Gmail API client for the local MELON assistant."""
from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any


class GmailClient:
    """Search and read Gmail messages using a local desktop OAuth token.

    This client deliberately exposes no compose, modify, delete, or send method.
    """

    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    def __init__(self, credentials_file: Path, token_file: Path):
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
        if not credentials_file.is_file():
            raise FileNotFoundError(
                f"OAuth client file not found: {credentials_file}. See docs/gmail_agent.md."
            )

        token_file.parent.mkdir(parents=True, exist_ok=True)
        creds = None
        if token_file.is_file():
            creds = Credentials.from_authorized_user_file(str(token_file), self.SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_file), self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            token_file.write_text(creds.to_json(), encoding="utf-8")

        self.service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    def search_messages(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return metadata and plain-text bodies for the most recent matches."""
        if not query.strip():
            raise ValueError("Gmail search query must not be empty.")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20.")
        response = self.service.users().messages().list(
            userId="me", q=query, maxResults=limit
        ).execute()
        messages = []
        for item in response.get("messages", []):
            message = self.service.users().messages().get(
                userId="me", id=item["id"], format="full"
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
