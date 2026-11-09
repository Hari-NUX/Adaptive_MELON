"""Run the local BIPIA-trained window detector over retrieved content."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from melon_guard.training.local_detector import load_model, predict_score


@dataclass(frozen=True)
class GuardResult:
    flagged: bool
    score: float
    threshold: float
    start: int
    excerpt: str


class TextWindowGuard:
    """Task-conditioned, local screening for possible injected instructions.

    This is an auxiliary text detector. It does not implement MELON's
    re-execution and tool-action comparison by itself.
    """

    def __init__(self, model_path: Path, threshold: float = 0.5):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.model = load_model(model_path)
        self.threshold = threshold
        self.width = int(self.model["window_chars"])

    def inspect(self, task: str, retrieved_content: str) -> GuardResult:
        if not task.strip():
            raise ValueError("task must not be empty")
        content = retrieved_content or ""
        if not content:
            return GuardResult(False, 0.0, self.threshold, 0, "")
        stride = max(1, self.width // 4)
        starts = list(range(0, max(1, len(content) - self.width + 1), stride))
        if not starts or starts[-1] + self.width < len(content):
            starts.append(max(0, len(content) - self.width))
        candidates = []
        for start in starts:
            excerpt = content[start : start + self.width]
            score = predict_score(self.model, f"TASK: {task}\nCONTENT: {excerpt}")
            candidates.append((score, start, excerpt))
        score, start, excerpt = max(candidates, key=lambda item: item[0])
        return GuardResult(score >= self.threshold, score, self.threshold, start, excerpt)
