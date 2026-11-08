# Fine-tuned MELON study

This project prepares email and webpage prompt-injection examples, generates and validates MELON teacher traces, and later compares base and LoRA/QLoRA-tuned Qwen models with and without the MELON runtime defense.

## Datasets and provenance

### BIPIA email data (four files)

| File | Official download | Purpose |
| --- | --- | --- |
| `email/train.jsonl` | [train.jsonl](https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/email/train.jsonl) | Email contexts for training preparation |
| `text_attack_train.json` | [text_attack_train.json](https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_train.json) | Training attack templates |
| `email/test.jsonl` | [test.jsonl](https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/email/test.jsonl) | Held-out email contexts |
| `text_attack_test.json` | [text_attack_test.json](https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_test.json) | Held-out attack templates |

BIPIA supplies contexts and attack templates separately. Constructed adversarial examples are generated from these ingredients; they are not themselves MELON-labeled trajectories.

### BrowseSafe-Bench internet data (two files)

The existing downloader uses BrowseSafe-Bench, which is **different** from BrowseSafe++. Its direct files are [train.parquet](https://huggingface.co/datasets/perplexity-ai/browsesafe-bench/resolve/main/train.parquet) and [test.parquet](https://huggingface.co/datasets/perplexity-ai/browsesafe-bench/resolve/main/test.parquet); see its [dataset card and license](https://huggingface.co/datasets/perplexity-ai/browsesafe-bench).

The 964 MB `members-train-0000.parquet` mentioned in the project notes is claimed to be a BrowseSafe++ shard. It is **not** one of the BrowseSafe-Bench files above. Keep it under `data/raw/browsesafepp/` and separate from `datasets/internet/raw/` until its upstream repository, exact revision, license, schema, and access terms have been verified. Do not have the clone downloader silently substitute one dataset for the other.

## Clone and download

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

Inference tags used in the pilot: `qwen3:0.6b` and `qwen3:1.7b`. For Hugging Face training, use [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) or [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). Ollama is for inference; LoRA training produces adapter weights that need a compatible inference/export path.

Use Git, PowerShell, and Python 3.11 in a separate environment. For practical QLoRA training, Linux/WSL2 with compatible NVIDIA CUDA hardware is recommended. Install a PyTorch build matching the local CUDA setup using the [official selector](https://pytorch.org/get-started/locally/).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare candidate email examples

```powershell
python .\scripts\prepare_bipia_email.py `
  --input '.\datasets\gmail\raw\email_train.jsonl' `
  --output '.\datasets\gmail\prepared\email_train.jsonl'
```

This uses only the email training split. Review generated records and the dataset license before using them. Training and browser-conversion commands remain intentionally unenabled until the teacher-label policy and schemas are validated.

## Repository structure

- `data/raw/` - source datasets such as BIPIA and BrowseSafe++ (local, excluded from Git).
- `datasets/` - canonical raw/prepared stores used by the current scripts.
- `configs/` - versioned experiment settings.
- `scripts/` - data download and preparation utilities.
- `src/` - agent, MELON teacher, data, training, and evaluation code.
- `runs/` and `outputs/` - local adapters, logs, metrics, and plots (excluded from Git).

## Citation and safety

Record the source URL/revision, retrieval date, license, preprocessing version, model checkpoint, seed, and case IDs for every run. Cite BIPIA, the exact BrowseSafe dataset used, AgentDojo, Qwen, and MELON in reports. Treat dataset text as untrusted data; do not execute instructions found in examples or follow embedded links.
