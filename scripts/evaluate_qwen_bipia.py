"""Compare base and LoRA Qwen answers on prepared BIPIA test examples."""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize(prediction).split()
    expected = normalize(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def evaluate_model(model_name: str, adapter: Path | None, rows: list[dict],
                   max_new_tokens: int, limit: int | None) -> tuple[dict, list[dict]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install PyTorch and Transformers before running Qwen evaluation.") from exc
    if adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Install PEFT to load the fine-tuned adapter.") from exc

    if limit is not None:
        rows = rows[:limit]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    grouped = defaultdict(list)
    with torch.inference_mode():
        for index, row in enumerate(rows):
            metadata = row["metadata"]
            if metadata.get("source_split") != "test" or metadata.get("record_split") != "test":
                raise ValueError(f"Test leakage guard failed at example {index}")
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
            if len(prompt_ids) > 2048:
                keep_head = 1024
                prompt_ids = prompt_ids[:keep_head] + prompt_ids[-(2048 - keep_head):]
            encoded = {"input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=device)}
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
            input_length = encoded["input_ids"].shape[1]
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            answer = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()
            reference = row["completion"]
            exact = normalize(answer) == normalize(reference)
            f1 = token_f1(answer, reference)
            label = metadata.get("label", "unspecified")
            result = {
                "index": index,
                "label": label,
                "prediction": answer,
                "reference": reference,
                "exact_match": exact,
                "token_f1": f1,
            }
            predictions.append(result)
            grouped[label].append(result)
    summary = {
        label: {
            "n": len(group),
            "exact_match": sum(item["exact_match"] for item in group) / max(1, len(group)),
            "mean_token_f1": sum(item["token_f1"] for item in group) / max(1, len(group)),
        }
        for label, group in sorted(grouped.items())
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, predictions


def load_test(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {})
            if metadata.get("source_split") != "test" or metadata.get("record_split") != "test":
                raise ValueError(f"Expected official test row at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No test rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path,
                        default=REPO_ROOT / "datasets" / "gmail" / "prepared" / "sft" / "test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "qwen3_bipia_eval")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, help="Optional smoke subset; omit for the full prepared test set")
    args = parser.parse_args()
    rows = load_test(args.test_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = [("base", None)]
    if args.adapter is not None:
        models.append(("lora", args.adapter))
    report = {
        "dataset": "Microsoft BIPIA EmailQA official test split",
        "test_file": str(args.test_file),
        "full_test_rows": len(rows),
        "evaluated_rows": min(len(rows), args.limit) if args.limit else len(rows),
        "task_metrics": "normalized exact match and token F1 against BIPIA ideal answers",
        "not_reported": "AgentDojo UA/ASR/BU; BIPIA answer metrics are not equivalent to MELON metrics",
        "models": {},
    }
    for name, adapter in models:
        summary, predictions = evaluate_model(args.model, adapter, rows, args.max_new_tokens, args.limit)
        report["models"][name] = summary
        prediction_path = args.output_dir / f"{name}_predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
