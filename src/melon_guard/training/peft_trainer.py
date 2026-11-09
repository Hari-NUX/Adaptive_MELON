"""CPU-first LoRA SFT for reviewed MELON teacher traces."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .data import format_student_example, load_teacher_records


class CompletionOnlyDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.items = []
        eos_id = tokenizer.eos_token_id
        for row in rows:
            prompt, completion = format_student_example(row)
            target_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                target_ids.append(eos_id)
            if not target_ids or len(target_ids) >= max_length:
                raise ValueError(f"Target too long or empty for scenario {row['scenario_id']}")
            context_marker = "Retrieved email/context:\n"
            suffix_marker = "\n\nChoose the tool action needed for the user task, then decide whether that action is safe.\nJSON response:\n"
            if context_marker not in prompt or suffix_marker not in prompt:
                raise ValueError("Student prompt formatting lost its context boundaries")
            head, remainder = prompt.split(context_marker, 1)
            context, _ = remainder.split(suffix_marker, 1)
            head_ids = tokenizer(head + context_marker, add_special_tokens=True)["input_ids"]
            suffix_ids = tokenizer(suffix_marker, add_special_tokens=False)["input_ids"]
            context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
            context_budget = max_length - len(target_ids) - len(head_ids) - len(suffix_ids)
            if context_budget < 1:
                raise ValueError(f"Task and target leave no room for email context ({row['scenario_id']})")
            if len(context_ids) > context_budget:
                beginning = (context_budget + 1) // 2
                ending = context_budget - beginning
                context_ids = context_ids[:beginning] + (context_ids[-ending:] if ending else [])
            prompt_ids = head_ids + context_ids + suffix_ids
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_lora(
    *,
    model_id: str,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    max_length: int = 512,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and save only a LoRA adapter; force CPU for predictable hardware use."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if max_length < 64 or epochs < 1:
        raise ValueError("max_length must be >= 64 and epochs must be positive")
    train_rows = load_teacher_records(train_path, "train")
    validation_rows = load_teacher_records(validation_path, "validation")
    trainable_cpu_threads = max(1, int(os.environ.get("MELON_CPU_THREADS", str(os.cpu_count() or 1))))
    torch.set_num_threads(trainable_cpu_threads)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_data = CompletionOnlyDataset(train_rows, tokenizer, max_length)
    validation_data = CompletionOnlyDataset(validation_rows, tokenizer, max_length)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
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
        logging_steps=1,
        report_to="none",
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", eval_result)
    trainer.save_state()
    return {
        "model_id": model_id,
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "adapter_dir": str(output_dir.resolve()),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
    }
