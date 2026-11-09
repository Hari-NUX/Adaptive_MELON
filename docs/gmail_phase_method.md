# Gmail phase: data, teacher labels, and efficiency claim

## Current status

- The four BIPIA email source files are present and pass the schema audit in `scripts/audit_bipia_email.py`.
- BIPIA has 50 train and 50 test EmailQA contexts. The train and test attack-template files each contain 15 categories with 5 templates per category. The context records have no explicit IDs; the audit and scenario sample derive stable SHA-256 identifiers.
- `datasets/gmail/prepared/email_scenario_sample.jsonl` is a 60-row inspection sample from train only. It contains two contexts, one template from each train category, and start/end insertion. It is not a full official benchmark build and has no MELON labels.
- `datasets/gmail/prepared/gmail_action_scenarios.jsonl` contains 250 synthetic action scenarios built from the 50 BIPIA train contexts: one clean and four clearly marked synthetic forwarding-injection variants per context. Train/validation assignment is by source-context hash, so variants stay together. These are pilot scenarios, not official BIPIA cases or teacher labels.
- BIPIA's `ideal` field is the expected EmailQA answer. It is not an ALLOW/BLOCK label, an action trace, or evidence that an email side effect occurred.
- Keep every source string intact. Malformed records should be reported for review, never silently dropped. Do not read BIPIA test data when generating teacher labels.

## What MELON does

For a user task, an agent reads tool output such as an email and proposes a tool action. MELON takes that execution state and replays it with a task-neutral masked user request while reusing the retrieved evidence and cached tool results. It compares the original and replayed tool actions. A suspiciously similar action after the user task has been masked may be driven by an instruction inside the email. The replay is a detector, not a second task-completion attempt; the masked run must not be allowed to perform real side effects.

For our email study, all tools must use a local simulated mailbox. A side effect such as creating a draft is recorded in the simulator; sending mail is disabled. `scripts/generate_teacher_traces.py` runs the original task and masked replay with a local Ollama model and records machine-generated action-comparison labels. A teacher record preserves the user task, source email/context and synthetic template IDs, original proposed action, masked replay action, action-comparison result, simulated state change, label policy, and review status. Labels are generated from the action comparison, not from whether a message merely looks malicious. The optional review script can inspect and override records.

The SFT loader accepts only paired-trace JSONL rows with `split` (`train` or `validation`), `split_group_id`, `scenario_id`, an approved or recognized machine-generated status, `user_task`, `retrieved_context`, `original_action`, `masked_action`, `action_comparison`, `simulated_outcome`, `teacher_decision` (`ALLOW` or `BLOCK`), and `teacher_reason`. The split group identifies the source scenario, so all variants of one email/task stay in the same partition. Replay evidence is retained for audit but hidden from the student's input.

## Important dataset limitation

BIPIA EmailQA contexts and attack templates test whether injected email text changes the answer to a question. They do not define the user's requested Gmail operations or supply tool-use trajectories. Therefore BIPIA can provide email text and an attack benchmark, but by itself cannot create MELON action labels. The action-oriented Gmail tasks and simulated tool environment must be authored separately and clearly marked as synthetic. BIPIA-only answer-integrity results and simulated side-effect results must be reported as separate evaluations.

## Fine-tuning plan

1. Finish the read-only audit; preserve the four raw files and their hashes.
2. Build train-only BIPIA answer-integrity scenarios and keep official test data held out for final evaluation.
3. Create a local-only simulated Gmail environment with explicit user goals, benign emails, injected emails, and auditable read/search/draft actions. Never expose a real send action.
4. Generate paired original and task-neutral replay traces from a fixed teacher model and fixed settings. Compare structured action names and security-relevant arguments. Audit a representative sample and manually correct ambiguous labels; preserve the machine label policy and model version on every row.
5. Split by underlying email/task scenario before generating examples: train and validation may use only the BIPIA training contexts; official BIPIA test contexts/templates remain evaluation-only. Do not split paraphrases or multiple injection positions from one source context across train and validation.
6. After the trace policy and a representative sample are accepted, run LoRA SFT on the Hugging Face causal-LM checkpoint. For the single-pass student experiment, the target is a structured proposed tool action plus `ALLOW`/`BLOCK` and a short reason. The student must not see the masked replay action in its input; that evidence is retained only for audit. The text target is trained with next-token cross-entropy; it is not a decision-tree model.
7. Compare base model, base+MELON replay, tuned model without replay, and tuned model+replay. Report utility, false blocks, attack success, and actual simulated side effects, plus input/output tokens and wall-clock latency.

## Teacher-label interpretation

The teacher is a labeling procedure: it observes the original and masked runs, compares their proposed actions, and applies a documented adjudication rule. The teacher is not BIPIA, and the teacher output is not automatically correct. Store the evidence so a person can inspect and correct labels. Training on these labels is MELON-inspired supervised distillation, not the original training-free MELON defense.

## Hardware, checkpoints, and claims

The 2 GB AMD integrated GPU on this Windows laptop is not a practical CUDA training device. This does not rule out LoRA: CPU LoRA may work for Qwen3-0.6B with constrained sequence lengths and batch sizes, but can be slow and memory-limited. A compatible remote GPU is another option. LoRA versus QLoRA is a memory/setup tradeoff, not a guarantee of quality. NVIDIA accelerates supported training; it does not guarantee equivalent or better model quality by itself. Compare the same base checkpoint, data, seed, and training settings where possible.

Ollama remains useful for inference baselines. Its local quantized model is not automatically the Hugging Face training checkpoint. A typical workflow downloads a compatible Hugging Face checkpoint separately, trains a PEFT adapter, then exports/loads it through a supported inference path. Keep enough disk space for the checkpoint, environment, logs, and adapter.

Distillation may let the tuned agent propose an action and its gate in one generation without MELON's replay, reducing model calls and tokens if evaluation supports that result. The application must enforce `BLOCK` before any write. It is only a hypothesis: a one-pass student can miss attacks the replay catches. If replay is still enabled, the tuned model has not removed MELON's extra pass, and there is no basis to claim lower token cost. Treat no-replay deployment as a separate evaluated ablation, not an assumption or a guaranteed replacement.

## Clone-and-run workflow status

`scripts/finetune.ps1` is the single Windows entry point. It can fetch BIPIA email files only, audit them, and create the inspection sample and synthetic action scenarios. The paired trace generator, source-group partitioner, and optional manual review tool are present. The Hugging Face checkpoint download requires an explicit confirmation, and the trainer uses CPU LoRA. It stops until the full trace set has been generated and partitioned into separate train/validation files; no training can be claimed before then.
