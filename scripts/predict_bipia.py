"""Scan retrieved content with the trained local BIPIA window detector."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from melon_guard.inference.security_guard import TextWindowGuard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    parser.add_argument("--model", type=Path,
                        default=REPO_ROOT / "runs" / "bipia_detector" / "email_window_nb.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    content = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    result = TextWindowGuard(args.model, threshold=args.threshold).inspect(args.task, content)
    print(json.dumps({
        "label": "possible_injection" if result.flagged else "no_injection_detected",
        "score": round(result.score, 4),
        "threshold": result.threshold,
        "window_start": result.start,
        "excerpt": result.excerpt,
        "warning": "Naive Bayes score is not calibrated confidence; treat as a review signal.",
    }, indent=2))


if __name__ == "__main__":
    main()
