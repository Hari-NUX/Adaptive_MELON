# Local chat interface

The Streamlit app can answer from either Gmail or an uploaded PDF. In
`Automatic` source mode, requests about your Gmail, inbox, or your emails use
Gmail; other requests use the uploaded PDF. Choose a source explicitly in the
sidebar to override automatic routing.

For Gmail requests, the default search query covers the current calendar month.
The query is editable, and the app can retrieve up to 100 matching messages.
For example, ask `Summarize all my mails this month with MELON`. The chat shows
the actual Gmail query and number of messages retrieved. If the 100-message cap
is reached or message text is too large for the local model, the app displays a
notice. To summarize a different period, change the Gmail search query or ask
for `last month`.

For PDF requests, upload a PDF with selectable text and ask a task in chat.
Include `with MELON`, `without MELON`, or `compare with and without MELON`; if
you omit the mode, the app compares both conditions. The no-MELON column is a
deliberately vulnerable model simulation. It can show how instructions inside
the source might alter a draft or proposed email action, but it never executes
that action. The MELON condition masks the task and compares proposed email
tool actions over the retrieved source text. The BIPIA text-window detector is
shown as a separate auxiliary signal. The local token-hash embedder is a
demonstration stand-in, not the paper's embedding model or a benchmark result.

Gmail content and PDF text are processed by the local app and sent only to
Ollama at `127.0.0.1`; the app does not submit them to a hosted language model.
The app cannot send email. You may optionally stage a displayed draft into
Gmail after checking its recipient, subject, and body and clicking the review
checkbox and draft button. Gmail remains unsent until you open the draft and
manually click Send.

## Start it

From the repository root in PowerShell:

```powershell
py -3.15 -m pip install -r .\requirements-ui.txt
py -3.15 -m pip install -r .\requirements-gmail.txt
ollama pull qwen3:0.6b
py -3.15 -m streamlit run .\ui\app.py --server.address 127.0.0.1
```

Streamlit opens at `http://localhost:8501`. Keep Ollama running. The Gmail
client also needs the Google OAuth desktop client file at
`%USERPROFILE%\.adaptive_melon\credentials.json`. The first Gmail read uses
the `gmail.readonly` scope and opens Google's consent flow if needed. Draft
creation uses the additional `gmail.compose` scope and may require consent
again.

## Example prompts

```text
Summarize all my mails this month with MELON.
Compare my inbox emails from this month with and without MELON.
Summarize the factual contents of this PDF and draft a short email. Compare with and without MELON.
Without MELON, show what action the PDF instructions could cause.
```

The no-MELON path is intentionally vulnerable for demonstration. Any real
Gmail draft still requires review and explicit confirmation in the UI.
