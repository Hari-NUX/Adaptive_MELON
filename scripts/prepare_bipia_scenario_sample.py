"""Build a small, train-only BIPIA email scenario sample for inspection.

This creates attack scenarios, not MELON teacher labels or SFT training data.
It follows BIPIA's start/end context insertion rules and keeps source hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {path}:{line_no}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            for key in ("context", "question", "ideal"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Missing/non-text {key} at {path}:{line_no}")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/gmail/raw"))
    parser.add_argument("--output", type=Path, default=Path("datasets/gmail/prepared/email_scenario_sample.jsonl"))
    parser.add_argument("--contexts", type=int, default=2, help="Number of source contexts to sample")
    parser.add_argument("--templates", type=int, default=2, help="Number of templates per category to sample")
    args = parser.parse_args()
    if args.contexts < 1 or args.templates < 1:
        parser.error("--contexts and --templates must be positive")

    train_path = args.raw_dir / "email_train.jsonl"
    attack_path = args.raw_dir / "text_attack_train.json"
    contexts = read_jsonl(train_path)
    attacks = json.loads(attack_path.read_text(encoding="utf-8"))
    if not isinstance(attacks, dict) or not attacks:
        raise ValueError("Expected non-empty category-to-template object in text_attack_train.json")

    selected_contexts = contexts[: args.contexts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as dest:
        for context in selected_contexts:
            context_id = digest(context["context"] + "\0" + context["question"])
            for category, templates in attacks.items():
                if not isinstance(templates, list):
                    raise ValueError(f"Attack category {category!r} must contain a list")
                for template_index, template in enumerate(templates[: args.templates]):
                    if not isinstance(template, str) or not template.strip():
                        raise ValueError(f"Empty/non-text template in {category!r} at index {template_index}")
                    template_id = f"{category}-{template_index}"
                    for position, poisoned in (
                        ("start", template + "\n" + context["context"]),
                        ("end", context["context"] + "\n" + template),
                    ):
                        row = {
                            "scenario_id": f"{context_id[:16]}:{template_id}:{position}",
                            "source": "BIPIA",
                            "source_split": "train",
                            "context_id": context_id,
                            "template_id": template_id,
                            "position": position,
                            "original_context": context["context"],
                            "poisoned_context": poisoned,
                            "question": context["question"],
                            "ideal": context["ideal"],
                            "provenance": {
                                "context_file": "email_train.jsonl",
                                "template_file": "text_attack_train.json",
                                "preprocessing": "bipia_scenario_sample_v1",
                            },
                            "label_status": "unlabeled_scenario_only",
                        }
                        dest.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
    print(f"Validated {len(contexts)} training contexts and {sum(len(v) for v in attacks.values())} training templates.")
    print(f"Wrote {written} train-only scenarios to {args.output}")
    print("No test files read; these scenarios have no MELON labels and are not SFT-ready.")


if __name__ == "__main__":
    main()
