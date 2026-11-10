# Local chat interface

The Streamlit app provides a browser-based local chat for PDF-based email
drafting and IPI comparison. Upload a text-based PDF in the sidebar and ask a
task in the chat. Include `with MELON`, `without MELON`, or `compare with and
without MELON` in the prompt; if omitted, the app compares both conditions.

The no-MELON column is a deliberately vulnerable model simulation. It can show
how PDF instructions might alter a draft or proposed email action, but it never
executes that action. The MELON condition masks the task and compares proposed
email tool actions over the cached PDF text. It also shows the BIPIA text-window
detector as a separate auxiliary signal. The MELON embedding here is a local
token-hash stand-in, not the paper's embedding model or a benchmark result.

PDF text is extracted in memory and sent only to Ollama on `127.0.0.1`; the
application does not upload the PDF or message to a hosted model. Scanned/image
PDFs need OCR before upload. The model cannot send email. You may optionally
stage either displayed draft into Gmail after checking its recipient, subject,
and body and clicking the draft button. Gmail remains unsent until you open the
draft and manually click Send.

## Start it

From the repository root in PowerShell:

```powershell
py -3.15 -m pip install -r .\requirements-ui.txt
ollama pull qwen3:0.6b
py -3.15 -m streamlit run .\ui\app.py --server.address 127.0.0.1
```

Streamlit opens the interface at `http://localhost:8501`. If it does not open
automatically, paste that address into your browser. Keep Ollama running.

The optional Gmail draft control also needs the Google OAuth client at
`%USERPROFILE%\.adaptive_melon\credentials.json` and the Gmail libraries:

```powershell
py -3.15 -m pip install -r .\requirements-gmail.txt
```

The app requests Gmail `gmail.compose` only when you click to create a draft.
That OAuth scope technically permits API sending, but this app has no send call.
Review and send manually in Gmail if you approve.

## Example prompts

```text
Summarize the factual contents of this PDF and draft a short email. Compare with and without MELON.
With MELON, summarize this PDF and prepare a draft email.
Without MELON, show what draft action the PDF's instructions could cause.
```

The UI's mode parser is advisory and accepts these phrases in a natural-language
prompt. The no-MELON path is intentionally vulnerable for demonstration; any
real Gmail draft still requires review and explicit confirmation in the UI.
