"""LoRA/QLoRA supervised fine-tuning for audited prompt/completion JSONL."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any


def _read_records(path: Path, expected_split: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected object in {path} line {line_number}")
            if not isinstance(record.get("prompt"), str) or not isinstance(record.get("completion"), str):
                raise ValueError(f"Expected prompt/completion strings in {path} line {line_number}")
            if not record["prompt"].strip() or not record["completion"].strip():
                raise ValueError(f"Empty prompt or completion in {path} line {line_number}")
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"Missing split metadata in {path} line {line_number}")
            actual = metadata.get("record_split")
            source_split = metadata.get("source_split")
            if actual != expected_split or source_split != "train":
                raise ValueError(
                    f"Split leakage guard: {path} line {line_number} has "
                    f"record_split={actual!r}, source_split={source_split!r}; "
                    f"expected record_split={expected_split!r} and source_split='train'"
                )
            records.append(record)
    if not records:
        raise ValueError(f"No training records found in {path}")
    return records


def train_peft(
    *,
    train_path: Path,
    validation_path: Path,
    model_name: str,
    output_dir: Path,
    method: str = "lora",
    max_seq_length: int = 2048,
    epochs: float = 3.0,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    seed: int = 42,
) -> None:
    """Fine-tune a causal LM; validation data is used for model selection only."""
    if method not in {"lora", "qlora"}:
        raise ValueError("method must be 'lora' or 'qlora'")
    if max_seq_length < 128:
        raise ValueError("max_seq_length must be at least 128")
    train_records = _read_records(train_path, "train")
    validation_records = _read_records(validation_path, "validation")

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "Install the model-training requirements and a PyTorch build for your device. "
            "See docs/installation.md."
        ) from exc

    cuda = torch.cuda.is_available()
    if method == "qlora" and not cuda:
        raise RuntimeError("QLoRA requires a compatible CUDA GPU; use --method lora on CPU.")
    if method == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("QLoRA requires a Transformers build with bitsandbytes support.") from exc
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization, device_map="auto", torch_dtype=compute_dtype
        )
    else:
        load_args = {}
        if cuda:
            load_args["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_args["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_args)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False

    if method == "qlora":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    class CompletionDataset(torch.utils.data.Dataset):
        def __init__(self, records):
            self.items = []
            for record in records:
                prompt_ids = tokenizer(record["prompt"], add_special_tokens=True)["input_ids"]
                answer_ids = tokenizer(record["completion"], add_special_tokens=False)["input_ids"]
                answer_ids = answer_ids[: max_seq_length - 2] + [tokenizer.eos_token_id]
                prompt_budget = max_seq_length - len(answer_ids)
                if prompt_budget <= 0:
                    raise ValueError("Completion leaves no tokens for the prompt")
                if len(prompt_ids) > prompt_budget:
                    head = max(1, prompt_budget // 2)
                    tail = prompt_budget - head
                    prompt_ids = prompt_ids[:head] + (prompt_ids[-tail:] if tail else [])
                input_ids = prompt_ids + answer_ids
                labels = [-100] * len(prompt_ids) + answer_ids
                self.items.append({"input_ids": input_ids, "labels": labels})

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    class CompletionCollator:
        def __call__(self, features):
            max_length = max(len(item["input_ids"]) for item in features)
            input_ids, labels, masks = [], [], []
            for item in features:
                pad = max_length - len(item["input_ids"])
                input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
                labels.append(item["labels"] + [-100] * pad)
                masks.append([1] * len(item["input_ids"]) + [0] * pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

    training_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=seed,
        data_seed=seed,
        gradient_checkpointing=True,
        fp16=bool(cuda and not torch.cuda.is_bf16_supported()),
        bf16=bool(cuda and torch.cuda.is_bf16_supported()),
    )
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    arguments = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=CompletionDataset(train_records),
        eval_dataset=CompletionDataset(validation_records),
        data_collator=CompletionCollator(),
    )
    trainer.train()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_metadata.json").write_text(json.dumps({
        "base_model": model_name,
        "method": method,
        "train_file": str(train_path),
        "validation_file": str(validation_path),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "max_seq_length": max_seq_length,
        "epochs": epochs,
        "seed": seed,
        "labels_masked_on_prompt": True,
    }, indent=2), encoding="utf-8")
