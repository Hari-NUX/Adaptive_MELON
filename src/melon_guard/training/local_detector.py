"""Dependency-free multinomial Naive Bayes detector for short text windows."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


def feature_counts(text: str) -> Counter[str]:
    lower = text.lower()
    words = re.findall(r"[a-z0-9_]+|[^\w\s]", lower)
    counts: Counter[str] = Counter("w:" + word for word in words)
    counts.update("b:" + a + " " + b for a, b in zip(words, words[1:]))
    task, _, content = lower.partition("\ncontent: ")
    task_words = set(re.findall(r"[a-z0-9_]+", task.removeprefix("task: ")))
    content_words = set(re.findall(r"[a-z0-9_]+", content))
    overlap = task_words & content_words
    ratio = len(overlap) / max(1, len(task_words))
    counts["r:overlap=" + ("low" if ratio < 0.08 else "mid" if ratio < 0.22 else "high")] += 1
    counts.update("x:" + word for word in overlap)
    compact = re.sub(r"\s+", " ", lower)
    for n in (3, 4, 5):
        counts.update(f"c{n}:" + compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
    return counts


def train(examples: Iterable, alpha: float = 0.15, window_chars: int = 200) -> dict:
    examples = list(examples)
    if not examples:
        raise ValueError("Cannot train on an empty dataset")
    class_docs = [0, 0]
    class_counts = [Counter(), Counter()]
    totals = [0, 0]
    document_frequency: Counter[str] = Counter()
    for example in examples:
        label = int(example.label)
        if label not in (0, 1):
            raise ValueError(f"Expected binary label, got {example.label!r}")
        counts = feature_counts(example.model_text())
        class_docs[label] += 1
        class_counts[label].update(counts)
        totals[label] += sum(counts.values())
        document_frequency.update(counts.keys())
    vocabulary = sorted(feature for feature, df in document_frequency.items() if df >= 2)
    if not vocabulary or not all(class_docs):
        raise ValueError("Training data must contain both classes and a non-empty vocabulary")
    denom = [totals[c] + alpha * len(vocabulary) for c in (0, 1)]
    return {
        "format": "melon-bipia-window-nb-v1",
        "window_chars": window_chars,
        "alpha": alpha,
        "vocab_size": len(vocabulary),
        "class_docs": class_docs,
        "denom": denom,
        "log_priors": [math.log(0.5), math.log(0.5)],
        "features": {f: [class_counts[c][f] for c in (0, 1)] for f in vocabulary},
        "training_rows": len(examples),
    }


def predict_score(model: dict, text: str) -> float:
    scores = list(model["log_priors"])
    alpha = model["alpha"]
    feature_table = model["features"]
    for feature, count in feature_counts(text).items():
        frequencies = feature_table.get(feature)
        if frequencies is None:
            continue
        for label in (0, 1):
            scores[label] += count * math.log((frequencies[label] + alpha) / model["denom"][label])
    pivot = max(scores)
    exponentials = [math.exp(max(-700.0, score - pivot)) for score in scores]
    return exponentials[1] / sum(exponentials)


def evaluate(model: dict, examples: Iterable, threshold: float = 0.5) -> dict:
    scored = [(predict_score(model, example.model_text()), int(example.label)) for example in examples]
    tp = sum(score >= threshold and label == 1 for score, label in scored)
    fp = sum(score >= threshold and label == 0 for score, label in scored)
    tn = sum(score < threshold and label == 0 for score, label in scored)
    fn = sum(score < threshold and label == 1 for score, label in scored)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    ordered = sorted(scored)
    positives = sum(label for _, label in scored)
    negatives = len(scored) - positives
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        mean_rank = ((index + 1) + end) / 2
        rank_sum += mean_rank * sum(label for _, label in ordered[index:end])
        index = end
    auc = (rank_sum - positives * (positives + 1) / 2) / max(1, positives * negatives)
    count = len(scored)
    always_attack_accuracy = positives / max(1, count)
    return {
        "n": count, "positive": positives, "negative": negatives,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / max(1, count),
        "always_attack_baseline_accuracy": always_attack_accuracy,
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": auc, "threshold": threshold,
    }


def save_model(model: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")


def load_model(path: Path) -> dict:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("format") != "melon-bipia-window-nb-v1":
        raise ValueError(f"Unsupported model format in {path}")
    return model
