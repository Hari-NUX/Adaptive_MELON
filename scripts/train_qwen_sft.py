"""LoRA/QLoRA fine-tuning on prepared BIPIA train and validation JSONL."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from melon_guard.training.peft_trainer import train_peft  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path,
                        default=REPO_ROOT / "datasets" / "gmail" / "prepared" / "sft" / "train.jsonl")
    parser.add_argument("--validation-file", type=Path,
                        default=REPO_ROOT / "datasets" / "gmail" / "prepared" / "sft" / "validation.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--method", choices=("lora", "qlora"), default="lora")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "qwen3_0_6b_lora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_peft(
        train_path=args.train_file,
        validation_path=args.validation_file,
        model_name=args.model,
        output_dir=args.output_dir,
        method=args.method,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
