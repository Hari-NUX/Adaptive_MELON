"""Interactive, CPU-first entry point for Gmail paired-trace LoRA distillation."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from melon_guard.training.data import load_teacher_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA = PROJECT_ROOT / "datasets/gmail/prepared/melon_teacher_train.jsonl"
VALIDATION_DATA = PROJECT_ROOT / "datasets/gmail/prepared/melon_teacher_validation.jsonl"
MODEL_CHOICES = {
    "1": ("Qwen3-0.6B (recommended for 16 GB RAM)", "Qwen/Qwen3-0.6B", "qwen3_0_6b"),
    "2": ("Qwen3-1.7B (higher CPU memory use)", "Qwen/Qwen3-1.7B", "qwen3_1_7b"),
}


def _select_models() -> list[tuple[str, str, str]]:
    print("\nModels available for LoRA fine-tuning:")
    for key, (label, model_id, _) in MODEL_CHOICES.items():
        print(f"  {key}) {label} [{model_id}]")
    print("  both) Train both sequentially (allow extra CPU time and disk space)")
    answer = input("Choose 1, 2, or both: ").strip().lower()
    if answer == "both":
        return list(MODEL_CHOICES.values())
    if answer in MODEL_CHOICES:
        return [MODEL_CHOICES[answer]]
    raise ValueError("Choose 1, 2, or both.")


def main() -> int:
    print("Gmail MELON-inspired LoRA fine-tuning")
    print(f"Project: {PROJECT_ROOT}")
    print("Training runs on CPU by default. Runtime may be several hours or longer; it depends on data and system load.")
    print("This workflow trains a single-pass action-and-safety student from paired Gmail teacher traces.")

    try:
        selected = _select_models()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if not TRAIN_DATA.is_file() or not VALIDATION_DATA.is_file():
        print("\nMELON teacher train/validation data is not ready, so no model will be downloaded or trained.")
        print(f"Expected training records:   {TRAIN_DATA}")
        print(f"Expected validation records: {VALIDATION_DATA}")
        print("Raw BIPIA files and unlabeled scenario samples cannot be used as safety labels.")
        print("Next, generate paired original/masked traces in the simulated Gmail environment, review them, and create separate scenario-level splits.")
        return 2

    try:
        train_rows = load_teacher_records(TRAIN_DATA, "train")
        validation_rows = load_teacher_records(VALIDATION_DATA, "validation")
    except (OSError, ValueError) as exc:
        print(f"Teacher data validation failed: {exc}", file=sys.stderr)
        return 2
    train_groups = {row["split_group_id"] for row in train_rows}
    validation_groups = {row["split_group_id"] for row in validation_rows}
    overlap = train_groups & validation_groups
    if overlap:
        print(f"Train/validation source-group leakage found ({len(overlap)} overlapping groups); refusing to train.", file=sys.stderr)
        return 2

    print(f"Teacher examples: {len(train_rows)} train, {len(validation_rows)} validation; source groups are disjoint.")
    generated_count = sum(row.get("review_status") == "teacher_generated" for row in train_rows + validation_rows)
    if generated_count:
        auto_count = sum(row.get("review_mode") == "automatic_policy_acceptance_v1" for row in train_rows + validation_rows)
        print(f"Label provenance: {generated_count} machine-generated; {auto_count} accepted automatically by the documented rule.")
        print("These labels are not human-reviewed or independently verified MELON judgments.")
    print("\nHugging Face checkpoints are separate from Ollama models.")
    print("The selected checkpoint will download automatically if needed, after you approve below.")
    print("Review each model's license and disk use before continuing.")
    if input("Type DOWNLOAD AND TRAIN to allow checkpoint download and begin CPU LoRA training: ").strip() != "DOWNLOAD AND TRAIN":
        print("Stopped before downloading model weights or starting training.")
        return 0

    from melon_guard.training.peft_trainer import train_lora

    for label, model_id, slug in selected:
        print(f"\nStarting {label} on CPU. Keep this terminal open until the run finishes.")
        try:
            result = train_lora(
                model_id=model_id,
                train_path=TRAIN_DATA,
                validation_path=VALIDATION_DATA,
                output_dir=PROJECT_ROOT / "runs" / f"gmail_{slug}_lora",
            )
        except Exception as exc:
            print(f"Training failed for {model_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"Saved adapter to {result['adapter_dir']}")
        print(f"Validation metrics: {result['eval_metrics']}")
    print("\nFinished. Evaluate the adapter on untouched test scenarios before making quality or security claims.")
    return 0
