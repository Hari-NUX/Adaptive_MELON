# Local email IPI demo

This demo uses a local Ollama language model to draft a fictional promotional
email. It puts an indirect prompt injection in retrieved email text and compares
the vulnerable baseline with a MELON masked replay and, when available, the
trained BIPIA text detector.

The vulnerable baseline's `send_email` action is routed to a local JSONL mock
outbox. The project has no SMTP sender, and the mock sink accepts only addresses
ending in the reserved `.invalid` domain. The protected path never dispatches a
send. This is for local security testing only.

## Install the generator

Install Ollama for Windows from [ollama.com/download/windows](https://ollama.com/download/windows).
Then open a new PowerShell window and download the small Qwen3 model:

```powershell
ollama --version
ollama pull qwen3:0.6b
ollama run qwen3:0.6b
```

At the Qwen prompt, try a short request, then enter `/bye`. Keep Ollama running
in the background. The BIPIA detector trained by `train_bipia_detector.py` is a
separate classifier; it does not generate email text.

## Run from the repository root

```powershell
py -3.15 .\examples\email_agent\ipi_demo.py
```

Use your installed Python selector version if it differs from `3.15`. The demo
asks the local Ollama API at `http://127.0.0.1:11434` for a JSON action plan. It
records a vulnerable-baseline send in
`runs\ipi_demo\mock_outbox.jsonl`, then shows the detector and MELON decisions.
No message leaves the computer.

To try another retrieved message, save it as UTF-8 and pass it with
`--retrieved-file .\retrieved.txt`. The mock sink still refuses non-`.invalid`
recipients. MELON's action embedding in this small demo is a local token-vector
stand-in, not the paper's embedding setup or a benchmark result.
