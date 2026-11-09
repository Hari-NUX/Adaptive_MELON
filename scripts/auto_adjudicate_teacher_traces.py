"""Apply the documented deterministic action-match rule to all provisional traces."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SIDE_EFFECT_TOOLS = {"create_draft", "forward_email"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/gmail/prepared/gmail_teacher_traces_unreviewed.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("datasets/gmail/prepared/melon_teacher_train.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("datasets/gmail/prepared/melon_teacher_validation.jsonl"))
    parser.add_argument("--overwrite", action="store_true", help="Replace the existing provisional partitions")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Trace file not found: {args.input}")
    outputs = (args.train_output, args.validation_output)
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("Train/validation outputs already exist; pass --overwrite to replace the provisional machine partitions")

    rows = read_jsonl(args.input)
    if not rows:
        print("No teacher traces found.")
        return 1
    ids: set[str] = set()
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    for line_number, row in enumerate(rows, 1):
        sid = row.get("scenario_id")
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError(f"Missing or duplicate scenario_id at {args.input}:{line_number}")
        ids.add(sid)
        if row.get("review_status") not in {"pending", "teacher_generated"}:
            raise ValueError(f"Not a provisional machine trace at {args.input}:{line_number}")
        if row.get("source_split") != "train" or row.get("split") not in {"train", "validation"}:
            raise ValueError(f"Refusing non-training-source trace at {args.input}:{line_number}")
        action = row.get("original_action", {})
        comparison = row.get("action_comparison", {})
        if not isinstance(action, dict) or not isinstance(comparison, dict) or not isinstance(comparison.get("equivalent"), bool):
            raise ValueError(f"Missing structured action comparison at {args.input}:{line_number}")
        block = action.get("tool") in SIDE_EFFECT_TOOLS and comparison["equivalent"]
        expected = "BLOCK" if block else "ALLOW"
        if row.get("teacher_decision") != expected:
            raise ValueError(f"Stored suggestion disagrees with the documented policy at {args.input}:{line_number}")
        row["review_status"] = "teacher_generated"
        row["label_adjudication"] = "exact_side_effect_action_match_v1"
        row["review_mode"] = "automatic_policy_acceptance_v1"
        row["human_reviewed"] = False
        row["teacher_reason"] = (
            "Automated policy label: original side-effect action was reproduced by the task-neutral replay."
            if block
            else "Automated policy label: the task-neutral replay did not reproduce the original side-effect action."
        )
        (train if row["split"] == "train" else validation).append(row)
        decisions[expected] += 1

    train_groups = {row.get("split_group_id") for row in train}
    validation_groups = {row.get("split_group_id") for row in validation}
    if None in train_groups or None in validation_groups or train_groups & validation_groups:
        raise ValueError("Missing split group or source-group leakage detected")
    if not train or not validation:
        print("Both train and validation partitions must contain records; no outputs written.")
        return 1

    for path, partition in ((args.train_output, train), (args.validation_output, validation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for row in partition:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Applied the recorded rule to {len(rows)} traces: {decisions['ALLOW']} ALLOW, {decisions['BLOCK']} BLOCK.")
    print(f"Wrote {len(train)} train and {len(validation)} validation records by source email.")
    print("These are automatically policy-labeled records, not human-reviewed labels or independently verified MELON judgments.")
    print("The interactive review-progress journal, if present, was left untouched and was not used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
