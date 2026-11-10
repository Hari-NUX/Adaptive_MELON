"""Interactively try the Gmail LoRA adapter; no email tools are connected."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_gmail_lora as evaluation


def read_email() -> str:
    print("Paste the email. Finish by entering END on its own line:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> int:
    adapter = evaluation.ADAPTER_DIR / "adapter_model.safetensors"
    if not adapter.is_file():
        print(f"Adapter not found: {adapter}", file=sys.stderr)
        return 2

    task = input("What should the email agent do? ").strip()
    if not task:
        print("Task cannot be empty.", file=sys.stderr)
        return 2
    context = read_email()
    if not context:
        print("Email text cannot be empty.", file=sys.stderr)
        return 2

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    threads = max(1, int(os.environ.get("MELON_CPU_THREADS", str(os.cpu_count() or 1))))
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(evaluation.MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        evaluation.MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, evaluation.ADAPTER_DIR)
    model.eval()

    row: dict[str, Any] = {"retrieved_context": context}
    prompt = evaluation.format_prompt(row, task)
    result = evaluation.generate(model, tokenizer, prompt)
    print("\nProposed output (display only; no action was run):")
    print(result["text"].strip())
    print(f"\nParsed structured decision: {result['parsed'] is not None}")
    print(f"Input tokens: {result['input_tokens']}; output tokens: {result['output_tokens']}")
    print(f"Generation time: {result['latency_seconds']:.2f} seconds")
    if result["parsed"] is None:
        print("The response did not match the expected action JSON schema.")
    else:
        print("Reminder: ALLOW/BLOCK is a model proposal, not an authorization or security guarantee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
