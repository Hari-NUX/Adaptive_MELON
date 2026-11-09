"""Interactively review provisional Gmail teacher traces with resumable progress."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank record at {path}:{line_number}")
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/gmail/prepared/gmail_teacher_traces_unreviewed.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("datasets/gmail/prepared/melon_teacher_train.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("datasets/gmail/prepared/melon_teacher_validation.jsonl"))
    parser.add_argument("--progress-file", type=Path, default=Path("datasets/gmail/prepared/gmail_teacher_review_progress.jsonl"))
    parser.add_argument("--overwrite", action="store_true", help="Replace provisional train/validation files after review completes")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Trace file not found: {args.input}")
    outputs = (args.train_output, args.validation_output)
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("Provisional train/validation files already exist; pass --overwrite to replace them after review completes")

    rows = read_jsonl(args.input)
    if not rows:
        print("No traces to review.")
        return 1
    scenario_ids = [row.get("scenario_id") for row in rows]
    if None in scenario_ids or len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Trace file has missing or duplicate scenario IDs")
    for line_number, row in enumerate(rows, 1):
        if row.get("review_status") not in {"pending", "teacher_generated"}:
            raise ValueError(f"Expected an unreviewed/teacher-generated trace at {args.input}:{line_number}")
        if row.get("split") not in {"train", "validation"} or row.get("source_split") != "train":
            raise ValueError(f"Refusing non-training-source trace at {args.input}:{line_number}")

    train_groups = {row["split_group_id"] for row in rows if row["split"] == "train"}
    validation_groups = {row["split_group_id"] for row in rows if row["split"] == "validation"}
    if train_groups & validation_groups:
        raise ValueError("Source-group leakage found between train and validation traces")

    prior: dict[str, dict[str, Any]] = {}
    if args.progress_file.exists():
        for progress in read_jsonl(args.progress_file):
            sid = progress.get("scenario_id")
            if sid not in set(scenario_ids):
                raise ValueError(f"Review progress refers to unknown scenario_id {sid!r}; move the progress file aside to start a new review")
            if sid in prior:
                raise ValueError(f"Duplicate review progress for scenario_id {sid!r}")
            if progress.get("status") not in {"approved", "skipped"}:
                raise ValueError(f"Invalid review progress for scenario_id {sid!r}")
            if progress.get("status") == "approved" and progress.get("decision") not in {"ALLOW", "BLOCK"}:
                raise ValueError(f"Invalid approved decision for scenario_id {sid!r}")
            prior[sid] = progress

    print(f"Loaded {len(rows)} provisional records; {len(prior)} already reviewed/skipped.")
    print("Label rule: BLOCK when an original simulated side-effect action is reproduced by the task-neutral replay; otherwise ALLOW.")
    print("A accepts the suggested label, L/B overrides it, S excludes the row, and Q saves progress and exits.")
    accepted: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    skipped = sum(p["status"] == "skipped" for p in prior.values())

    for index, row in enumerate(rows, 1):
        sid = row["scenario_id"]
        progress = prior.get(sid)
        if progress is None:
            print(f"\n[{index}/{len(rows)}] {sid} ({row['split']})")
            print(f"Source context: {row.get('source_context_id')} | scenario: {row.get('scenario_kind')} / {row.get('attack_id')}")
            print(f"User task: {row.get('user_task')}")
            print(f"Retrieved email/context:\n{row.get('retrieved_context', '')}")
            print(f"Original action: {json.dumps(row.get('original_action', {}), ensure_ascii=False)}")
            print(f"Additional original calls (not executed): {json.dumps(row.get('additional_original_tool_calls', []), ensure_ascii=False)}")
            print(f"Masked replay action: {json.dumps(row.get('masked_action', {}), ensure_ascii=False)}")
            print(f"Additional masked calls: {json.dumps(row.get('additional_masked_tool_calls', []), ensure_ascii=False)}")
            print(f"Action comparison: {json.dumps(row.get('action_comparison', {}), ensure_ascii=False)}")
            print(f"Simulated original outcome: {json.dumps(row.get('simulated_outcome', {}), ensure_ascii=False)}")
            print(f"Suggested label: {row.get('teacher_decision')} — {row.get('teacher_reason')}")
            choice = input("[A] accept suggestion, [L] set ALLOW, [B] set BLOCK, [S] exclude, [Q] save and quit: ").strip().lower()
            if choice == "q":
                print(f"Stopped. Progress is saved in {args.progress_file}; rerun this command to resume.")
                print("The current provisional train/validation files were not modified.")
                return 1
            if choice == "s":
                progress = {"scenario_id": sid, "status": "skipped", "decision": None, "reason": "Reviewer excluded this record."}
                skipped += 1
            elif choice == "a":
                progress = {"scenario_id": sid, "status": "approved", "decision": row["teacher_decision"], "reason": row["teacher_reason"]}
            elif choice in {"l", "b"}:
                decision = "ALLOW" if choice == "l" else "BLOCK"
                reason = input("Reviewer reason (required): ").strip()
                if not reason:
                    print("A reason is required. This row remains unreviewed; rerun to see it again.")
                    continue
                progress = {"scenario_id": sid, "status": "approved", "decision": decision, "reason": reason}
            else:
                print("Unrecognized choice. This row remains unreviewed; rerun to see it again.")
                continue

            args.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with args.progress_file.open("a", encoding="utf-8", newline="\n") as journal:
                journal.write(json.dumps(progress, ensure_ascii=False) + "\n")
                journal.flush()
            prior[sid] = progress

        if progress["status"] == "approved":
            reviewed = dict(row)
            reviewed["teacher_decision"] = progress["decision"]
            reviewed["teacher_reason"] = progress["reason"]
            reviewed["review_status"] = "approved"
            reviewed["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
            reviewed["label_adjudication"] = "human_review_of_paired_simulated_actions_v1"
            accepted[row["split"]].append(reviewed)

    if not accepted["train"] or not accepted["validation"]:
        print("Review reached the end, but at least one split has no approved records. No train/validation files were written.")
        return 1
    for path, split in ((args.train_output, "train"), (args.validation_output, "validation")):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for row in accepted[split]:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Review complete: approved {len(accepted['train'])} train and {len(accepted['validation'])} validation records; excluded {skipped}.")
    print("Approved files now replace the provisional machine-labeled partitions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
