"""Train a text-only Qwen3-0.6B LoRA detector from prepared BrowseSafe++ train pairs."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "datasets/internet/prepared/browsesafepp_text_train.jsonl"
VALIDATION_FILE = ROOT / "datasets/internet/prepared/browsesafepp_text_validation.jsonl"
MANIFEST_FILE = ROOT / "datasets/internet/prepared/browsesafepp_text_manifest.json"
MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = ROOT / "runs/internet_qwen3_0_6b_lora"
MAX_LENGTH = 512


def load_records(path: Path, expected_split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            for key in ("example_id", "pair_id", "prompt", "target", "split", "source_split", "label"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Missing/non-text {key} at {path}:{line_number}")
            if row["split"] != expected_split or row["source_split"] != "train":
                raise ValueError(f"Unexpected split provenance at {path}:{line_number}")
            if row["label"] not in {"yes", "no"}:
                raise ValueError(f"Unexpected label {row['label']!r} at {path}:{line_number}")
            target = json.loads(row["target"])
            if target != {"injection_detected": row["label"] == "yes"}:
                raise ValueError(f"Target and label disagree at {path}:{line_number}")
            if row["example_id"] in ids:
                raise ValueError(f"Duplicate example_id {row['example_id']!r} in {path}")
            ids.add(row["example_id"])
            records.append(row)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


class CompletionOnlyDataset:
    """Train only on the short detector target; preserve both ends of long pages."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.items = []
        eos_id = tokenizer.eos_token_id
        page_marker = "Untrusted webpage (Markdown):\n"
        suffix_marker = "\n\nJSON response:\n"
        for row in rows:
            prompt = row["prompt"]
            if page_marker not in prompt or suffix_marker not in prompt:
                raise ValueError(f"Unexpected prompt boundaries for {row['example_id']}")
            prefix, rest = prompt.split(page_marker, 1)
            page, _suffix = rest.split(suffix_marker, 1)
            target_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                target_ids.append(eos_id)
            if not target_ids or len(target_ids) >= max_length:
                raise ValueError(f"Target too long or empty for {row['example_id']}")
            prefix_ids = tokenizer(prefix + page_marker, add_special_tokens=True)["input_ids"]
            suffix_ids = tokenizer(suffix_marker, add_special_tokens=False)["input_ids"]
            page_ids = tokenizer(page, add_special_tokens=False)["input_ids"]
            budget = max_length - len(prefix_ids) - len(suffix_ids) - len(target_ids)
            if budget < 1:
                raise ValueError(f"Prompt leaves no room for page text ({row['example_id']})")
            if len(page_ids) > budget:
                first = (budget + 1) // 2
                last = budget - first
                page_ids = page_ids[:first] + (page_ids[-last:] if last else [])
            prompt_ids = prefix_ids + page_ids + suffix_ids
            self.items.append({
                "input_ids": prompt_ids + target_ids,
                "labels": [-100] * len(prompt_ids) + target_ids,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(row["input_ids"]) for row in features)
        ids, masks, labels = [], [], []
        for row in features:
            pad = width - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.pad_token_id] * pad)
            masks.append([1] * len(row["input_ids"]) + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1, help="Full shard is much larger than Gmail; default is one CPU epoch")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_length < 128:
        parser.error("--epochs must be positive and --max-length at least 128")
    for path in (TRAIN_FILE, VALIDATION_FILE, MANIFEST_FILE):
        if not path.is_file():
            parser.error(f"Prepared BrowseSafe++ data is missing: {path}; run scripts/prepare_browsesafepp.py first")

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("test_data_used") is not False or manifest.get("teacher_traces_used") is not False:
        parser.error("Manifest does not certify train-only, direct-label preparation; refusing to train")
    train_rows = load_records(TRAIN_FILE, "train")
    validation_rows = load_records(VALIDATION_FILE, "validation")
    train_pairs = {row["pair_id"] for row in train_rows}
    validation_pairs = {row["pair_id"] for row in validation_rows}
    if train_pairs & validation_pairs:
        parser.error(f"Pair leakage: {len(train_pairs & validation_pairs)} pair IDs overlap")
    print(f"BrowseSafe++ text detector SFT: {len(train_rows)} train pages, {len(validation_rows)} validation pages.")
    print(f"Pairs are disjoint: {len(train_pairs)} train, {len(validation_pairs)} validation.")
    print(f"Labels: train={dict(Counter(row['label'] for row in train_rows))}; validation={dict(Counter(row['label'] for row in validation_rows))}.")
    print("Base model: Qwen/Qwen3-0.6B. This trains a separate adapter from the base, not from the Gmail adapter.")
    print("CPU training; one epoch by default. Longer page text is clipped to both beginning and end within the token budget.")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    torch.set_num_threads(max(1, int(os.environ.get("MELON_CPU_THREADS", str(os.cpu_count() or 1)))))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    ))
    model.print_trainable_parameters()
    train_data = CompletionOnlyDataset(train_rows, tokenizer, args.max_length)
    validation_data = CompletionOnlyDataset(validation_rows, tokenizer, args.max_length)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        use_cpu=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", eval_result)
    trainer.save_state()
    result = {
        "model_id": MODEL_ID,
        "adapter_dir": str(OUTPUT_DIR.resolve()),
        "dataset": manifest.get("dataset"),
        "manifest": str(MANIFEST_FILE.resolve()),
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "objective": "JSON injection-presence prediction (yes/no); not browser action generation",
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
    }
    (OUTPUT_DIR / "browsesafepp_training_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved separate BrowseSafe++ LoRA adapter to {OUTPUT_DIR.resolve()}")
    print(f"Validation loss: {eval_result.get('eval_loss')}")
    print("Next: evaluate accuracy/precision/recall/F1 on the held-out pair-grouped validation split before packaging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
