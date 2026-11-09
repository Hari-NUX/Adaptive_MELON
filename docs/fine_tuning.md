# Fine-tuning

## Current BIPIA experiment

scripts/prepare_bipia_sft.py creates clean and attack-injected EmailQA examples
from the official BIPIA train split. It also creates a validation partition
with held-out contexts and attack prompt variants. The original ideal answer is
the target in both cases, and the prompt tells Qwen to treat email text as
untrusted.

scripts/train_qwen_sft.py runs completion-only LoRA or QLoRA training. The
trainer checks split metadata and accepts only source_split=train records in
the train and validation files. The test split is never read by the trainer.
Prompt tokens receive label -100, so the loss applies only to completion
tokens.

scripts/evaluate_qwen_bipia.py reports exact match and token F1 for clean and
injected test contexts. These are task-answer metrics, not MELON ASR.

## MELON teacher distillation

This is not implemented as an automatic training pipeline yet. Before doing
that experiment, generate and audit the masked-run trajectory labels described
in teacher_generation.md, partition by scenario, then train and compare the
adapter and base model on the same held-out AgentDojo cases.
