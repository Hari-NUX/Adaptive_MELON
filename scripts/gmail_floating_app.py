"""Local always-on-top Gmail reader and draft assistant backed by Ollama."""
from __future__ import annotations

import base64
import json
import queue
import re
import threading
import tkinter as tk
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import getaddresses
from html.parser import HTMLParser
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SECRETS_DIR = ROOT / "secrets"
CLIENT_FILE = SECRETS_DIR / "credentials.json"
TOKEN_FILE = SECRETS_DIR / "gmail_token.json"
MODEL_TAG = "gmail-qwen3-0.6b-lora"
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MAX_EMAIL_CHARS = 1500


def decoded_header(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except (LookupError, UnicodeError, ValueError):
        return value


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _decode_part(data: str) -> str:
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    return raw.decode("utf-8", errors="replace")


def _message_body(payload: dict[str, Any]) -> str:
    plain: list[str] = []
    html_parts: list[str] = []

    def visit(part: dict[str, Any]) -> None:
        mime = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data")
        if data and mime in {"text/plain", "text/html"}:
            decoded = _decode_part(data)
            (plain if mime == "text/plain" else html_parts).append(decoded)
        for child in part.get("parts", []) or []:
            visit(child)

    visit(payload)
    if plain:
        return "\n\n".join(plain).strip()
    if html_parts:
        parser = _HTMLText()
        parser.feed("\n".join(html_parts))
        return "\n".join(parser.parts)
    return "[This email has no readable plain-text body; it may contain only attachments.]"


def gmail_service() -> Any:
    if not CLIENT_FILE.is_file():
        raise FileNotFoundError(
            f"OAuth client JSON not found. Save your Google Desktop app file at: {CLIENT_FILE}"
        )
    from google.auth.transport.requests import Request as GoogleRequest
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    credentials = None
    if TOKEN_FILE.is_file():
        credentials = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        existing_scopes = set(credentials.scopes or [])
        if existing_scopes and not set(SCOPES).issubset(existing_scopes):
            raise RuntimeError(
                f"Saved OAuth token lacks the current Gmail permissions. Close the app, remove {TOKEN_FILE}, "
                "then launch again to authorize read and draft access."
            )
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(GoogleRequest())
    elif not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_FILE), SCOPES)
        credentials = flow.run_local_server(port=0, open_browser=True)
    TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def list_inbox(service: Any, limit: int = 20) -> list[dict[str, str]]:
    result = service.users().messages().list(userId="me", labelIds=["INBOX"], maxResults=limit).execute()
    rows: list[dict[str, str]] = []
    for item in result.get("messages", []):
        metadata = service.users().messages().get(
            userId="me", id=item["id"], format="metadata", metadataHeaders=["From", "Subject", "Date"]
        ).execute()
        headers = {h["name"].lower(): h["value"] for h in metadata.get("payload", {}).get("headers", [])}
        rows.append({
            "id": item["id"],
            "thread_id": item.get("threadId", ""),
            "from": decoded_header(headers.get("from")),
            "subject": decoded_header(headers.get("subject")) or "(no subject)",
            "date": decoded_header(headers.get("date")),
            "snippet": metadata.get("snippet", ""),
        })
    return rows


def get_email(service: Any, message_id: str, metadata: dict[str, str]) -> dict[str, str]:
    message = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    body = _message_body(message.get("payload", {}))
    return {**metadata, "body": body[:MAX_EMAIL_CHARS], "body_truncated": str(len(body) > MAX_EMAIL_CHARS)}


def parse_model_json(text: str) -> dict[str, Any] | None:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        begin, end = value.find("{"), value.rfind("}")
        if begin < 0 or end <= begin:
            return None
        try:
            data = json.loads(value[begin:end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    action = data.get("proposed_action")
    if action is None and isinstance(data.get("tool"), str):
        action = {"tool": data["tool"], "arguments": data.get("arguments", {})}
    if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
        return None
    args = action.get("arguments", {})
    if not isinstance(args, dict):
        return None
    data["proposed_action"] = {"tool": action["tool"], "arguments": args}
    decision = data.get("decision")
    data["decision"] = decision.upper() if isinstance(decision, str) else "UNKNOWN"
    return data


def generate_with_ollama(task: str, body: str) -> tuple[str, dict[str, Any] | None]:
    # The Modelfile supplies the matching trained prefix and suffix. Email text
    # remains data in the prompt; the application never runs model-proposed tools.
    prompt = f"{task}\n\nRetrieved email/context:\n{body}"
    request_body = json.dumps({
        "model": MODEL_TAG,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 160, "num_ctx": 512},
    }).encode("utf-8")
    request = Request(OLLAMA_URL, data=request_body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise ConnectionError(
            f"Cannot reach local Ollama at {OLLAMA_URL}. Start Ollama and confirm the model exists with "
            f"`ollama show {MODEL_TAG}`. Details: {exc}"
        ) from exc
    output = result.get("response", "")
    return output, parse_model_json(output)


def create_gmail_draft(service: Any, recipient: str, subject: str, body: str) -> str:
    if any(char in recipient + subject for char in "\r\n"):
        raise ValueError("Recipient and subject cannot contain line breaks.")
    addresses = [address for _, address in getaddresses([recipient]) if address]
    if not addresses or any("@" not in address or " " in address for address in addresses):
        raise ValueError("Enter a valid recipient email address in the To field.")
    message = EmailMessage()
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
    draft = service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
    return str(draft.get("id", "created"))


class GmailFloatingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Local Gmail Draft Assistant · Qwen3-0.6B")
        self.root.geometry("980x780+40+60")
        self.root.minsize(780, 620)
        self.root.attributes("-topmost", True)
        self.jobs: queue.Queue[tuple[bool, Callable[[], None] | str]] = queue.Queue()
        self.busy = False
        self.service: Any = None
        self.inbox: list[dict[str, str]] = []
        self.selected_email: dict[str, str] | None = None
        self._build_ui()
        self.root.after(100, self._poll_jobs)

    def _build_ui(self) -> None:
        root = self.root
        top = ttk.Frame(root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Local Gmail Draft Assistant", font=("Segoe UI", 15, "bold")).pack(side="left")
        self.float_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Keep floating", variable=self.float_var,
                        command=lambda: root.attributes("-topmost", self.float_var.get())).pack(side="right")
        ttk.Label(root, text="Local Ollama model · reads inbox · saves drafts only · no send feature",
                  foreground="#46566b").pack(anchor="w", padx=12)

        controls = ttk.Frame(root, padding=(10, 6))
        controls.pack(fill="x")
        self.refresh_button = ttk.Button(controls, text="Refresh inbox", command=self.refresh_inbox)
        self.refresh_button.pack(side="left")
        ttk.Button(controls, text="New message", command=self.new_message).pack(side="left", padx=(6, 0))
        self.status = tk.StringVar(value="Add OAuth credentials, then refresh inbox.")
        ttk.Label(controls, textvariable=self.status).pack(side="left", padx=10)

        panes = ttk.Panedwindow(root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        left = ttk.Frame(panes, width=310)
        right = ttk.Frame(panes)
        panes.add(left, weight=1)
        panes.add(right, weight=3)
        ttk.Label(left, text="Inbox (latest 20)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.email_list = tk.Listbox(left, activestyle="dotbox", exportselection=False)
        self.email_list.pack(fill="both", expand=True, pady=(4, 0))
        self.email_list.bind("<<ListboxSelect>>", self.select_email)

        ttk.Label(right, text="Selected email · untrusted message content", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.email_view = scrolledtext.ScrolledText(right, height=16, wrap="word", state="disabled")
        self.email_view.pack(fill="both", expand=True, pady=(4, 8))

        ttk.Label(root, text="Task for Qwen", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12)
        self.task = tk.Text(root, height=3, wrap="word")
        self.task.pack(fill="x", padx=12, pady=(3, 6))
        self.task.insert("1.0", "Summarize this email in 2–3 bullet points. Prepare an unsent draft. Do not send or forward it.")
        self.generate_button = ttk.Button(root, text="Generate draft text with local Qwen", command=self.generate_draft)
        self.generate_button.pack(anchor="w", padx=12, pady=(0, 6))
        self.decision = tk.StringVar(value="Model proposal: not generated")
        ttk.Label(root, textvariable=self.decision, foreground="#8b4a00").pack(anchor="w", padx=12)

        form = ttk.Frame(root, padding=(10, 4))
        form.pack(fill="x")
        ttk.Label(form, text="To (enter/review manually)").grid(row=0, column=0, sticky="w")
        self.to_entry = ttk.Entry(form)
        self.to_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(form, text="Subject (review/edit)").grid(row=0, column=1, sticky="w")
        self.subject_entry = ttk.Entry(form)
        self.subject_entry.grid(row=1, column=1, sticky="ew")
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=2)
        ttk.Label(root, text="Draft body · edit before saving", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(4, 0))
        self.draft_body = scrolledtext.ScrolledText(root, height=7, wrap="word")
        self.draft_body.pack(fill="both", expand=False, padx=12, pady=(3, 6))
        self.save_button = ttk.Button(root, text="Save as Gmail draft (never sends)", command=self.save_draft)
        self.save_button.pack(anchor="e", padx=12, pady=(0, 8))
        ttk.Label(root, text="Review the recipient and all text. The model can copy malicious instructions from emails. Saving creates only a Gmail draft; this app has no send operation.",
                  foreground="#8b2f2f", wraplength=920).pack(anchor="w", padx=12, pady=(0, 8))

    def _async(self, work: Callable[[], Any], done: Callable[[Any], None], busy_text: str) -> None:
        if self.busy:
            return
        self.busy = True
        self.status.set(busy_text)
        for button in (self.refresh_button, self.generate_button, self.save_button):
            button.configure(state="disabled")

        def worker() -> None:
            try:
                result = work()
                self.jobs.put((True, lambda: done(result)))
            except Exception as exc:
                self.jobs.put((False, f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_jobs(self) -> None:
        try:
            while True:
                ok, payload = self.jobs.get_nowait()
                self.busy = False
                for button in (self.refresh_button, self.generate_button, self.save_button):
                    button.configure(state="normal")
                if ok:
                    assert callable(payload)
                    payload()
                else:
                    self.status.set("Action failed; see message.")
                    messagebox.showerror("Gmail Draft Assistant", str(payload), parent=self.root)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_jobs)

    def _get_service(self) -> Any:
        if self.service is None:
            self.service = gmail_service()
        return self.service

    def refresh_inbox(self) -> None:
        def work() -> list[dict[str, str]]:
            return list_inbox(self._get_service())

        def done(rows: list[dict[str, str]]) -> None:
            self.inbox = rows
            self.email_list.delete(0, tk.END)
            for row in rows:
                self.email_list.insert(tk.END, f"{row['subject']}\n{row['from']} · {row['date']}")
            self.status.set(f"Loaded {len(rows)} inbox messages. Select one to inspect it.")

        self._async(work, done, "Connecting to Gmail and loading inbox…")

    def select_email(self, _event: Any = None) -> None:
        selection = self.email_list.curselection()
        if not selection:
            return
        metadata = self.inbox[selection[0]]

        def work() -> dict[str, str]:
            return get_email(self._get_service(), metadata["id"], metadata)

        def done(row: dict[str, str]) -> None:
            self.selected_email = row
            self.email_view.configure(state="normal")
            self.email_view.delete("1.0", tk.END)
            truncation = "\n\n[Body shortened for display/model context.]" if row["body_truncated"] == "True" else ""
            self.email_view.insert("1.0", f"From: {row['from']}\nDate: {row['date']}\nSubject: {row['subject']}\n\n{row['body']}{truncation}")
            self.email_view.configure(state="disabled")
            self.subject_entry.delete(0, tk.END)
            subj = row["subject"]
            self.subject_entry.insert(0, subj if subj.lower().startswith("re:") else f"Re: {subj}")
            self.status.set("Email loaded. Enter a task and generate text for review.")

        self._async(work, done, "Reading selected email…")

    def new_message(self) -> None:
        self.email_list.selection_clear(0, tk.END)
        self.selected_email = None
        self.email_view.configure(state="normal")
        self.email_view.delete("1.0", tk.END)
        self.email_view.insert("1.0", "New message: no email context selected. The model will use your task only.")
        self.email_view.configure(state="disabled")
        self.to_entry.delete(0, tk.END)
        self.subject_entry.delete(0, tk.END)
        self.draft_body.delete("1.0", tk.END)
        self.task.delete("1.0", tk.END)
        self.task.insert("1.0", "Compose a concise email using these instructions. Prepare an unsent draft; do not send or forward it.")
        self.decision.set("Model proposal: not generated")
        self.status.set("New message. Enter a task, then generate and review the draft.")

    def generate_draft(self) -> None:
        task = self.task.get("1.0", "end-1c").strip()
        if not task:
            messagebox.showinfo("Add a task", "Enter what you want the draft to say.", parent=self.root)
            return
        body = self.selected_email["body"] if self.selected_email else ""

        def work() -> tuple[str, dict[str, Any] | None]:
            return generate_with_ollama(task, body)

        def done(result: tuple[str, dict[str, Any] | None]) -> None:
            raw, parsed = result
            self.draft_body.delete("1.0", tk.END)
            if parsed is None:
                self.draft_body.insert("1.0", raw)
                self.decision.set("Model proposal could not be parsed; this text is not actioned.")
                self.status.set("Review or edit manually; model output was not structured JSON.")
                return
            action = parsed["proposed_action"]
            args = action["arguments"]
            if action["tool"] == "create_draft":
                self.draft_body.insert("1.0", str(args.get("body", "")))
                self.decision.set(f"Model proposal: {parsed['decision']} · proposed tool: create_draft (advisory only)")
                self.status.set("Draft text generated locally. Verify To, subject, and body before saving.")
            else:
                self.draft_body.insert("1.0", raw)
                self.decision.set(f"Model proposal: {parsed['decision']} · proposed tool: {action['tool']} (not executed)")
                self.status.set("No proposed action was run. Edit the draft body yourself if needed.")

        self._async(work, done, "Generating with local Ollama… (no email action is being taken)")

    def save_draft(self) -> None:
        recipient = self.to_entry.get().strip()
        subject = self.subject_entry.get().strip()
        body = self.draft_body.get("1.0", "end-1c").strip()
        if not recipient or not subject or not body:
            messagebox.showinfo("Complete the draft", "Enter a recipient, subject, and body before saving.", parent=self.root)
            return
        if not messagebox.askyesno(
            "Save Gmail draft?",
            f"Save an UNSENT draft to Gmail?\n\nTo: {recipient}\nSubject: {subject}\n\nReview its body in the main window first.",
            parent=self.root,
        ):
            return

        def work() -> str:
            return create_gmail_draft(self._get_service(), recipient, subject, body)

        def done(draft_id: str) -> None:
            self.status.set(f"Saved unsent Gmail draft ({draft_id}). Nothing was sent.")
            messagebox.showinfo("Draft saved", "Saved as an unsent Gmail draft. This app cannot send it.", parent=self.root)

        self._async(work, done, "Saving unsent Gmail draft…")


def main() -> int:
    root = tk.Tk()
    GmailFloatingApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
