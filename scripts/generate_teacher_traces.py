"""Generate provisional original/masked traces using a local Ollama model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from melon_guard.teacher.melon_runner import generate_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "datasets/gmail/prepared/gmail_action_scenarios.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/gmail/prepared/gmail_teacher_traces_unreviewed.jsonl")
    parser.add_argument("--model", default="qwen3:1.7b", help="An Ollama model tag already installed locally")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0, help="0 means all scenarios; use a small positive value for a pilot")
    parser.add_argument("--resume", action="store_true", help="Append only scenarios whose trace IDs are not already in the output file")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Scenario file not found: {args.input}. Run build_gmail_action_scenarios.py first.")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    try:
        count = generate_file(
            args.input,
            args.output,
            args.model,
            base_url=args.base_url,
            timeout=args.timeout,
            limit=args.limit,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"Teacher trace generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {count} PROVISIONAL traces to {args.output}; review before training.")
    print("The email simulator is local-only. No real email was drafted, forwarded, or sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
