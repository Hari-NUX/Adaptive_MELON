"""Write machine-generated paired traces into leakage-checked train/validation files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/gmail/prepared/gmail_teacher_traces_unreviewed.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("datasets/gmail/prepared/melon_teacher_train.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("datasets/gmail/prepared/melon_teacher_validation.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = (args.train_output, args.validation_output)
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("Output exists; choose another path or pass --overwrite")

    rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"Blank line at {args.input}:{line_number}")
            row = json.loads(line)
            if row.get("review_status") not in {"pending", "teacher_generated"}:
                raise ValueError(f"Expected an unreviewed machine teacher trace at {args.input}:{line_number}")
            if row.get("source_split") != "train" or row.get("split") not in {"train", "validation"}:
                raise ValueError(f"Refusing non-training-source trace at {args.input}:{line_number}")
            metadata = row.get("trace_metadata", {})
            comparison = row.get("action_comparison", {})
            original_action = row.get("original_action", {})
            provisional_block = (
                original_action.get("tool") in {"create_draft", "forward_email"}
                and comparison.get("equivalent") is True
            )
            expected_decision = "BLOCK" if provisional_block else "ALLOW"
            if row.get("teacher_decision") != expected_decision:
                raise ValueError(f"Teacher label does not match the recorded action policy at {args.input}:{line_number}")
            if metadata.get("method") != "paired_ollama_replay_v1":
                raise ValueError(f"Unexpected trace method at {args.input}:{line_number}")
            row["review_status"] = "teacher_generated"
            row["label_adjudication"] = "exact_side_effect_action_match_v1"
            row["teacher_reason"] = (
                "Machine teacher block: the same side-effect action appeared in masked replay."
                if provisional_block
                else "Machine teacher allow: the masked replay did not reproduce the original side-effect action."
            )
            rows.append(row)
    if not rows:
        print("No teacher traces found.")
        return 1

    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if not train or not validation:
        print("Need at least one train and one validation scenario; no outputs were written.")
        return 1
    train_groups = {row["split_group_id"] for row in train}
    validation_groups = {row["split_group_id"] for row in validation}
    if train_groups & validation_groups:
        raise ValueError("Source-group leakage between train and validation")
    scenario_ids = [row["scenario_id"] for row in rows]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Duplicate scenario IDs in teacher trace file")

    for path, partition_rows in ((args.train_output, train), (args.validation_output, validation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for row in partition_rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train)} machine-labeled train and {len(validation)} validation rows.")
    print("These labels are produced by the recorded action-comparison rule, not human-reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
