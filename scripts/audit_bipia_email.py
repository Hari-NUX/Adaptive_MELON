"""Read-only schema and provenance audit for all four local BIPIA email files."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_jsonl(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blank_lines: list[int] = []
    parse_errors: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                blank_lines.append(line_no)
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record root is not an object")
                rows.append(row)
            except Exception as exc:
                parse_errors.append({"line": line_no, "error": str(exc)})

    required = ("context", "question", "ideal")
    invalid_fields = []
    signatures: Counter[str] = Counter()
    explicit_ids = 0
    for index, row in enumerate(rows, 1):
        absent = [key for key in required if not isinstance(row.get(key), str) or not row[key].strip()]
        if absent:
            invalid_fields.append({"record": index, "fields": absent})
        if "id" in row or "uid" in row:
            explicit_ids += 1
        if isinstance(row.get("context"), str) and isinstance(row.get("question"), str):
            signatures[hashlib.sha256((row["context"] + "\0" + row["question"]).encode("utf-8")).hexdigest()] += 1
    duplicate_groups = sum(1 for count in signatures.values() if count > 1)
    return {
        "kind": "jsonl_email_contexts",
        "records_parsed": len(rows),
        "required_fields": list(required),
        "records_with_missing_or_empty_fields": invalid_fields,
        "blank_lines": blank_lines,
        "parse_errors": parse_errors,
        "duplicate_context_question_groups": duplicate_groups,
        "records_with_explicit_id_or_uid": explicit_ids,
        "sample_keys": sorted(rows[0].keys()) if rows else [],
    }


def audit_attacks(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"kind": "attack_templates", "parse_error": str(exc)}
    if not isinstance(obj, dict):
        return {"kind": "attack_templates", "schema_error": "root must be an object"}
    counts = {}
    for category, templates in obj.items():
        if not isinstance(category, str) or not isinstance(templates, list):
            errors.append(f"{category!r}: category must map to a list")
            continue
        counts[category] = len(templates)
        for index, template in enumerate(templates):
            if not isinstance(template, str) or not template.strip():
                errors.append(f"{category!r}[{index}]: template must be non-empty text")
    return {
        "kind": "attack_templates",
        "categories": len(obj),
        "template_count": sum(counts.values()),
        "templates_per_category": counts,
        "schema_errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/gmail/raw"))
    parser.add_argument("--output", type=Path, default=Path("datasets/gmail/prepared/bipia_audit.json"))
    args = parser.parse_args()
    files = (
        ("email_train.jsonl", audit_jsonl),
        ("email_test.jsonl", audit_jsonl),
        ("text_attack_train.json", audit_attacks),
        ("text_attack_test.json", audit_attacks),
    )
    report: dict[str, Any] = {
        "audit_version": "bipia_email_audit_v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Read-only audit. Preserve source text; do not discard injected content.",
        "files": {},
    }
    for filename, audit_fn in files:
        path = args.raw_dir / filename
        if not path.is_file():
            report["files"][filename] = {"missing": True}
            continue
        report["files"][filename] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            **audit_fn(path),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote read-only audit to {args.output}")
    for filename, details in report["files"].items():
        n = details.get("records_parsed", details.get("template_count", "?"))
        errs = len(details.get("parse_errors", [])) + len(details.get("records_with_missing_or_empty_fields", [])) + len(details.get("schema_errors", []))
        print(f"{filename}: count={n}, validation_issues={errs}, sha256={details.get('sha256', 'MISSING')}")


if __name__ == "__main__":
    main()
