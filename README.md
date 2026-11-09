# Fine-tuned MELON study

This project prepares email and webpage prompt-injection examples, generates and validates MELON teacher traces, and later compares base and LoRA/QLoRA-tuned Qwen models with and without the MELON runtime defense.

## Datasets and provenance

### BIPIA email data (four files)

| File                     | Official download                                                                                                 | Purpose                                 |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `email/train.jsonl`      | [train.jsonl](https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/email/train.jsonl)                 | Email contexts for training preparation |
| `text_attack_train.json` | [text_attack_train.json](https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_train.json) | Training attack templates               |
| `email/test.jsonl`       | [test.jsonl](https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/email/test.jsonl)                   | Held-out email contexts                 |
| `text_attack_test.json`  | [text_attack_test.json](https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_test.json)   | Held-out attack templates               |

BIPIA supplies contexts and attack templates separately. Constructed adversarial examples are generated from these ingredients; they are not themselves MELON-labeled trajectories.

**What BIPIA is:** BIPIA (Benchmarking and Improving Prompt Injection Attack resistance) is the benchmark and research project. Its GitHub repository contains data, optional builder/evaluation code, and experiment support. The four downloaded JSON/JSONL files are the actual source data for this email subset; they are not a Python library. This project does not need to install BIPIA as a package to audit those files. Its builder is useful as the reference for constructing official benchmark cases, while MELON tool-action traces require a separate simulated email-agent environment.

### BrowseSafe-Bench internet data (two files)

The existing downloader uses BrowseSafe-Bench, which is **different** from BrowseSafe++. Its direct files are [train.parquet](https://huggingface.co/datasets/perplexity-ai/browsesafe-bench/resolve/main/train.parquet) and [test.parquet](https://huggingface.co/datasets/perplexity-ai/browsesafe-bench/resolve/main/test.parquet); see its [dataset card and license](https://huggingface.co/datasets/perplexity-ai/browsesafe-bench).

The 964 MB `members-train-0000.parquet` mentioned in the project notes is claimed to be a BrowseSafe++ shard. It is **not** one of the BrowseSafe-Bench files above. Keep it under `data/raw/browsesafepp/` and separate from `datasets/internet/raw/` until its upstream repository, exact revision, license, schema, and access terms have been verified. Do not have the clone downloader silently substitute one dataset for the other.

## Clone and download

### Prepare the Gmail phase (Windows)

With Ollama running and the four BIPIA email files already in `datasets/gmail/raw/`, run this from the repository root:

```powershell
.\scripts\prepare_gmail.ps1
```

This audits the source files, reuses or builds training-only simulated Gmail scenarios, resumes paired local Ollama traces, and partitions them by source email into train and validation files. It does not install training dependencies, download Hugging Face weights, or start fine-tuning. If trace generation is interrupted, rerun the same command to resume from saved rows. The default teacher is `qwen3:1.7b`; to select another already-installed Ollama tag, use `-Model <tag>`.

The resulting partitions are provisional machine labels. Before training, review them interactively with a resumable journal:

```powershell
python .\scripts\review_teacher_traces.py --overwrite
```

The reviewer displays the task, retrieved context, paired actions, comparison, and simulated outcome. `A` accepts the suggested label, `L` or `B` overrides it with a required reason, `S` excludes the row, and `Q` saves progress to `datasets/gmail/prepared/gmail_teacher_review_progress.jsonl`. Rerun the command to continue. The provisional partitions are replaced only after the full review reaches completion.

For a non-interactive run, apply only the documented machine action-match rule with:

```powershell
python .\scripts\auto_adjudicate_teacher_traces.py --overwrite
```

This deterministic option uses each trace's recorded suggestion; it does not press random keys or represent the labels as human-reviewed. It replaces the provisional partitions and leaves any interactive review journal untouched.

Clone the repository, open PowerShell in the cloned project root, and run this one command:

```powershell
Set-Location 'D:\path\to\Fine_Tuned_MELON_Based'; .\scripts\download_datasets.ps1
```

Replace the path with the actual clone location. The script downloads the four BIPIA files and two BrowseSafe-Bench Parquet files into `datasets/gmail/raw/` and `datasets/internet/raw/`. To skip a source already downloaded, use `-SkipBIPIA` or `-SkipBrowseSafe`. The downloader needs internet access. Downloaded datasets are excluded from Git; each collaborator retrieves their own copy and must follow upstream terms before redistribution.

The generated layout is:

```text
datasets/
  gmail/raw/       email_train.jsonl, email_test.jsonl,
                   text_attack_train.json, text_attack_test.json
  gmail/prepared/  derived training examples
  internet/raw/    browsesafe_train.parquet, browsesafe_test.parquet
  internet/prepared/
```

## Are these datasets enough?

They are enough to begin schema inspection and preprocessing prototypes, but not enough on their own for a complete MELON agent fine-tuning study. BIPIA EmailQA is not a full Gmail inbox or tool-use trajectory corpus. BrowseSafe webpage labels indicate injection examples; they are not MELON's tool-action comparison decisions. Neither source provides the original-run and masked-run traces needed for MELON teacher supervision.

Before training: inspect and document source versions, licenses, columns, and examples; construct training scenarios only from training splits; build isolated simulated email/browser tools; collect original and masked execution traces; compare proposed tool actions; validate BLOCK/ALLOW labels against actual environment side effects; then split by source/scenario to prevent leakage. Keep all official test data out of teacher generation and training. The current preparation script only creates candidate email prompt/completion records and is not a MELON teacher-data generator.

The 964 MB BrowseSafe++ file should be recorded separately from the downloader's BrowseSafe-Bench data. Confirm its source and metadata before loading it. AgentDojo is an executable evaluation environment and not a flat downloadable training corpus.

## Experiment sequence

1. Inspect/download and validate dataset metadata and licensing.
2. Build attack/control scenarios from train data and a separate synthetic tool environment for realistic agent actions.
3. Capture MELON teacher traces and audit the resulting labels.
4. Create leakage-resistant train/validation/test partitions; never use official test examples for LoRA/QLoRA training.
5. Train Qwen3-0.6B (and optionally 1.7B) only after the target format and labels are reviewed.
6. Evaluate paired conditions: base Qwen, base+MELON, tuned Qwen, and optionally tuned+MELON. Report denominators, BU, UA, ASR, false positives, versions, and seeds. Do not present one-task smoke outcomes as benchmark results.

The original MELON method is a runtime defense. A model trained on MELON-generated labels is a separate distillation/teacher-supervision experiment and must be described as such.

## Models and environment

Inference tags used in the pilot: `qwen3:0.6b` and `qwen3:1.7b`. The trainer uses the corresponding Hugging Face checkpoints. Ollama remains useful for inference; its quantized model is separate from the Transformers checkpoint and the saved LoRA adapter.

The initial training path is CPU LoRA for Windows systems such as the project laptop. The launcher selects CPU PyTorch and Python 3.11. Expect CPU training to be slow; the wall time depends on model size, sequence length, dataset size, and system load. Running fewer background applications can leave more RAM for training, but cannot make CPU training equivalent in speed to a CUDA GPU. The project makes no fixed-hours or quality guarantee.

### Gmail adapter evaluation

After training Qwen3-0.6B, run a small held-out preview with:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_gmail_lora.py --limit-contexts 5
```

This uses five untouched BIPIA test email contexts expanded into clean cases and two newly written synthetic forwarding-injection cases per context. It compares base one-pass, base plus task-neutral replay, and the LoRA student in a local, non-sending simulation. The base model uses its Qwen chat template; the student keeps its SFT prompt. Review `runs/gmail_qwen3_0_6b_lora/gmail_eval_preview_chatprompt.json`. For the full 50-context evaluation, pass `--limit-contexts 0`; do that after checking the preview and its parse rate. Metrics are a pilot of this synthetic action task, not an official BIPIA score or broad security guarantee.

## One-command launcher

From PowerShell in the cloned repository, run:

```powershell
.\scripts\finetune.ps1
```

The launcher can offer to install Python 3.11, fetch the four BIPIA email files while explicitly skipping BrowseSafe-Bench, audit the files, create a 250-row train-only synthetic action-scenario set, and generate paired traces with local Ollama `qwen3:1.7b`. It source-group partitions machine labels into train and validation; `scripts/review_teacher_traces.py` can inspect and override them. The launcher prompts before downloading the Ollama teacher model if missing and before downloading the selected Hugging Face student checkpoint. It does not convert BIPIA answer keys into safety labels. A four-row paired pilot is recorded in `docs/gmail_teacher_pilot.md`.

For the current local pilot, paired provisional traces can be generated from the synthetic scenarios with an installed Ollama model using `python .\scripts\generate_teacher_traces.py --model qwen3:1.7b`. Inspect them with `python .\scripts\review_teacher_traces.py` before training. The masked replay is read-only; original proposed writes affect only the in-memory simulator.

## Prepare candidate email examples

```powershell
python .\scripts\prepare_bipia_email.py `
  --input '.\datasets\gmail\raw\email_train.jsonl' `
  --output '.\datasets\gmail\prepared\email_train.jsonl'
```

This uses only the email training split and creates ordinary EmailQA answer candidates. It does not create MELON teacher labels and is not accepted by the safety fine-tuning loader.

## Repository structure

- `data/raw/` - source datasets such as BIPIA and BrowseSafe++ (local, excluded from Git).
- `datasets/` - canonical raw/prepared stores used by the current scripts.
- `configs/` - versioned experiment settings.
- `scripts/` - data download and preparation utilities.
- `src/` - agent, MELON teacher, data, training, and evaluation code.
- `runs/` and `outputs/` - local adapters, logs, metrics, and plots (excluded from Git).

## Citation and safety

Record the source URL/revision, retrieval date, license, preprocessing version, model checkpoint, seed, and case IDs for every run. Cite BIPIA, the exact BrowseSafe dataset used, AgentDojo, Qwen, and MELON in reports. Treat dataset text as untrusted data; do not execute instructions found in examples or follow embedded links.

fron comparision b/w bare & melon with adapter after gmail fine tuning
.\.venv\Scripts\python.exe .\scripts\evaluate_gmail_lora.py --limit-contexts 5
