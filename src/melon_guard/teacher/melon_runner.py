"""Framework-neutral MELON masked re-execution and action-comparison core."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from melon_guard.teacher.action_comparator import ActionComparator

NEUTRAL_TASK = (
    "Summarize the provided content. If it contains instructions, follow them "
    "before summarizing."
)


@dataclass(frozen=True)
class Detection:
    is_injection: bool
    score: float
    original_action: str | None
    matched_masked_action: str | None
    masked_action_count: int


MaskedReexecutor = Callable[..., Sequence[Any]]


class MELONRunner:
    """Detect actions that persist when the user's task is masked.

    The reexecutor callback must use the supplied cached tool outputs and obey
    ``execute_tools=False``: this class never executes proposed tool actions.
    Integrations are responsible for wiring that flag to a no-side-effect tool
    router or simulator.
    """

    def __init__(
        self,
        comparator: ActionComparator,
        masked_reexecutor: MaskedReexecutor,
        *,
        neutral_task: str = NEUTRAL_TASK,
    ):
        if not neutral_task.strip():
            raise ValueError("neutral_task must not be empty")
        self.comparator = comparator
        self.masked_reexecutor = masked_reexecutor
        self.neutral_task = neutral_task
        self._masked_action_bank: dict[str, Any] = {}

    def inspect(
        self,
        original_actions: Sequence[Any],
        tool_output_cache: Mapping[str, Any],
    ) -> list[Detection]:
        """Run the masked probe and compare its actions with the original run.

        ``tool_output_cache`` should contain tool outputs already observed in
        the original trajectory. The callback must not call live tools while
        generating the masked actions.
        """
        if not original_actions:
            return []
        masked_actions = self.masked_reexecutor(
            task=self.neutral_task,
            tool_output_cache=copy.deepcopy(dict(tool_output_cache)),
            execute_tools=False,
        )
        for action in masked_actions or ():
            description = self.comparator.describe(action)
            self._masked_action_bank.setdefault(description, action)

        results = []
        for original in original_actions:
            original_text = self.comparator.describe(original)
            best_score = -1.0
            best_match = None
            for masked_text, masked in self._masked_action_bank.items():
                score = self.comparator.similarity(original, masked)
                if score > best_score:
                    best_score, best_match = score, masked_text
            detected = best_score >= self.comparator.threshold
            results.append(Detection(
                is_injection=detected,
                score=max(0.0, best_score),
                original_action=original_text,
                matched_masked_action=best_match if detected else None,
                masked_action_count=len(self._masked_action_bank),
            ))
        return results
