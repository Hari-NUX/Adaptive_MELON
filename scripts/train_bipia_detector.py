"""Train and evaluate the local BIPIA auxiliary text detector."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from melon_guard.data.loaders.bipia import build_email_examples  # noqa: E402
from melon_guard.training.local_detector import evaluate, save_model, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=REPO_ROOT / "datasets" / "gmail" / "raw",
                        help="BIPIA benchmark root or downloaded datasets/gmail/raw directory")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "bipia_detector")
    parser.add_argument("--window-chars", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_examples = build_email_examples(args.benchmark_root, "train", window_chars=args.window_chars, seed=args.seed)
    test_examples = build_email_examples(args.benchmark_root, "test", window_chars=args.window_chars, seed=args.seed)
    model = train(train_examples, window_chars=args.window_chars)
    model["training_split"] = "BIPIA EmailQA train"
    model["seed"] = args.seed
    metrics = evaluate(model, test_examples, threshold=args.threshold)
    report = {
        "dataset": "Microsoft BIPIA EmailQA",
        "dataset_source": "https://github.com/microsoft/BIPIA",
        "retrieved_on": date.today().isoformat(),
        "method": model["format"],
        "label_definition": "clean context window=0; inserted BIPIA attack-template window=1",
        "split": "official train and test contexts plus disjoint train and test attack-template files",
        "window_chars": args.window_chars,
        "seed": args.seed,
        "train_rows": len(train_examples),
        "test": metrics,
        "interpretation": [
            "These are text-window classification metrics, not MELON AgentDojo UA, ASR, or BU.",
            "BIPIA supplies context and attack templates, not MELON masked-run tool-action labels.",
            "Use the model as an auxiliary signal; it does not replace action-level checks or tool permissions.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, args.output_dir / "email_window_nb.json")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
