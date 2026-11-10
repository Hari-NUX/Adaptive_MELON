# Local Gmail assistant

The regular CLI connects to Gmail with Google's desktop OAuth flow, searches
and reads messages matched by a user-supplied Gmail query, then asks local
Ollama to answer the user's task. Its Gmail API access is read-only. Email
content is treated as untrusted data; the model never receives Gmail tool calls,
and the regular CLI cannot send, draft, delete, archive, or modify mail. A
requested reply is generated as text in the terminal for the user to review.

The BIPIA window detector is an optional warning signal. It does not change
Gmail state or decide what to do with messages. For an IPI send-attempt demo,
use `examples/email_agent/ipi_demo.py`, which routes its vulnerable send only
to a local mock outbox.

## Compare Gmail behavior with and without MELON

To show both paths in one report using selected Gmail messages:

```powershell
py -3.15 .\scripts\gmail_ipi_compare.py `
  --task "Draft a reply to the selected email, but do not send it" `
  --query "in:inbox newer_than:30d"
```

The no-MELON plan follows instructions in retrieved email to illustrate the
vulnerable path. The MELON condition masks the task, replays over cached message
text without executing tools, and compares the proposed email action. Both
plans, the MELON result, and the BIPIA text-detector signal appear in one JSON
report. The local token-vector embedder is a demonstration stand-in, not the
paper's embedding model or a benchmark result.

To optionally stage the no-MELON result as an unsent Gmail draft, add
`--create-draft`. The script displays the full recipient, subject, and body and
requires you to type `CREATE DRAFT` before creating it. It has no Gmail send
call. Open the draft in Gmail, review it, and click Send yourself only if you
approve. Google's `gmail.compose` OAuth scope also permits API sending, even
though this program never uses that operation. The regular CLI does not request
this scope; draft mode requires renewed consent if you previously authorized
read-only access.

```powershell
py -3.15 .\scripts\gmail_ipi_compare.py `
  --task "Draft a reply to the selected email, but do not send it" `
  --query "in:inbox newer_than:30d" `
  --create-draft
```

## One-time Google setup

1. Create a project in Google Cloud Console and enable the Gmail API.
2. Configure the Google Auth Platform consent screen for testing and add your
   own Gmail address as a test user if prompted.
3. Create an OAuth client with application type **Desktop app** and download
   its JSON file.
4. Create this local folder and save the downloaded JSON as
   `%USERPROFILE%\.adaptive_melon\credentials.json`:

   ```powershell
   New-Item -ItemType Directory -Force "$HOME\.adaptive_melon"
   ```

The credential and refresh token stay outside the repository. The regular CLI
asks Google for the `gmail.readonly` OAuth scope only. Google treats this as a
restricted scope, so keep this as a personal local test app and do not distribute
it without following Google's verification and data-use requirements. Optional
comparison draft mode also requests `gmail.compose`, which permits managing
drafts and sending; the app code uses it only to create a draft after the
confirmation described above.

## Install and run

Install the optional Gmail client libraries from the repository root:

```powershell
py -3.15 -m pip install -r .\requirements-gmail.txt
```

Ensure Ollama is running and Qwen is available:

```powershell
ollama pull qwen3:0.6b
```

Run the interactive CLI:

```powershell
py -3.15 .\scripts\gmail_agent.py
```

The first run opens Google's login and consent page. Later runs reuse the local
token at `%USERPROFILE%\.adaptive_melon\gmail_token.json`. Enter a task such as
"Summarize the latest invoice email" and a Gmail search such as
`in:inbox subject:invoice newer_than:30d`. You can also pass them directly:

```powershell
py -3.15 .\scripts\gmail_agent.py `
  --task "Summarize the latest invoice and list its due date" `
  --query "in:inbox subject:invoice newer_than:30d" `
  --limit 5
```

The message bodies go to `http://127.0.0.1:11434` for local Ollama inference.
No Gmail content is sent to a hosted model by this script. The model can produce
draft text in its answer, but the Gmail integration cannot write or send it.

If OAuth scopes change in a future version, delete the local token file and
authorize again. Never commit `credentials.json` or the token file.
