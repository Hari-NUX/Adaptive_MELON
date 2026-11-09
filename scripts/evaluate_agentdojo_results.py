"""Aggregate per-case AgentDojo results into BU, UA, ASR, and benign FPR."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from melon_guard.evaluation.metrics import agent_defense_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSONL with one evaluated case per line")
    parser.add_argument("--group-by", nargs="*", default=["defense", "agent", "attack"])
    args = parser.parse_args()
    records = []
    with args.results.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at line {line_number}")
                records.append(value)
    if not records:
        raise ValueError("No result records were found")

    groups = defaultdict(list)
    for record in records:
        key = tuple((field, record.get(field, "unspecified")) for field in args.group_by)
        groups[key].append(record)
    report = {
        "source": str(args.results),
        "group_by": args.group_by,
        "results": [
            {**dict(key), **agent_defense_metrics(cases)}
            for key, cases in sorted(groups.items(), key=lambda item: str(item[0]))
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
