"""Prepare clean and injected BIPIA EmailQA SFT records without split leakage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from melon_guard.data.loaders.bipia import build_email_sft_records  # noqa: E402


def write_records(path: Path, records: list[dict], split_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            if record["metadata"]["record_split"] != split_name:
                continue
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_split(records: list[dict], name: str) -> dict:
    selected = [r for r in records if r["metadata"]["record_split"] == name]
    return {
        "total": len(selected),
        "clean": sum(r["metadata"]["label"] == "clean" for r in selected),
        "injected": sum(r["metadata"]["label"] == "injected" for r in selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=REPO_ROOT / "datasets" / "gmail" / "raw")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "datasets" / "gmail" / "prepared" / "sft")
    parser.add_argument("--include-test", action="store_true",
                        help="Also prepare official test data for evaluation; never use it for training")
    args = parser.parse_args()

    train_records = build_email_sft_records(args.benchmark_root, "train")
    write_records(args.output_dir / "train.jsonl", train_records, "train")
    write_records(args.output_dir / "validation.jsonl", train_records, "validation")
    report = {
        "dataset": "Microsoft BIPIA EmailQA",
        "source_split_for_fitting": "train only",
        "train": count_split(train_records, "train"),
        "validation": count_split(train_records, "validation"),
        "official_test_prepared": False,
        "label_policy": "retain the original EmailQA answer; injected context remains untrusted",
    }
    if args.include_test:
        test_records = build_email_sft_records(args.benchmark_root, "test")
        write_records(args.output_dir / "test.jsonl", test_records, "test")
        report["official_test_prepared"] = True
        report["test"] = count_split(test_records, "test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
