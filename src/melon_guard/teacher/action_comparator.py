"""Compare structured tool actions for the MELON masked re-execution check."""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

Embedder = Callable[[str], Sequence[float]]

_OMIT_GENERIC = {"body", "content", "html", "message", "text"}


def _get(value: Any, key: str, default=None):
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def action_description(action: Any) -> str:
    """Create a stable, focused description of a tool call.

    Volatile message bodies are omitted for send_email; only recipient fields
    are compared, following the paper's focused argument-comparison design.
    Sensitive action arguments such as recipient and amount remain included.
    """
    function = _get(action, "function")
    arguments = _get(action, "args")
    if function is None:
        function = _get(action, "name", _get(action, "tool"))
    if arguments is None:
        arguments = _get(action, "arguments", {})
    if isinstance(function, Mapping):
        if arguments in (None, {}):
            arguments = function.get("arguments", function.get("args", {}))
        function = function.get("name", function.get("function", ""))
    elif function is not None and not isinstance(function, str):
        if arguments in (None, {}):
            arguments = _get(function, "arguments", _get(function, "args", {}))
        function = _get(function, "name", _get(function, "function", ""))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("tool action arguments are not valid JSON") from exc
    if not isinstance(function, str) or not function.strip():
        raise ValueError("tool action must include a function/name")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise ValueError("tool action arguments must be a mapping")
    normalized_name = function.strip()
    if normalized_name == "send_email":
        keep = {"recipients", "to", "cc", "bcc"}
        selected = {key: value for key, value in arguments.items() if key.lower() in keep}
    elif normalized_name == "send_money":
        keep = {"recipient", "amount"}
        selected = {key: value for key, value in arguments.items() if key.lower() in keep}
    else:
        selected = {key: value for key, value in arguments.items() if key.lower() not in _OMIT_GENERIC}
    rendered = json.dumps(selected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"{normalized_name}({rendered})"


class ActionComparator:
    """Cosine similarity comparator backed by a caller-supplied embedder.

    For paper-aligned evaluation, supply the same embedding model and threshold
    used by the MELON experiment. The callback keeps API and local embedding
    providers out of the core package.
    """

    def __init__(self, embedder: Embedder, threshold: float = 0.8):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.embedder = embedder
        self.threshold = threshold
        self._vectors: dict[str, tuple[float, ...]] = {}

    def describe(self, action: Any) -> str:
        return action_description(action)

    def _embed(self, description: str) -> tuple[float, ...]:
        if description not in self._vectors:
            vector = tuple(float(value) for value in self.embedder(description))
            if not vector or any(not math.isfinite(value) for value in vector):
                raise ValueError("embedder must return a non-empty finite vector")
            self._vectors[description] = vector
        return self._vectors[description]

    def similarity(self, left: Any, right: Any) -> float:
        left_text = self.describe(left)
        right_text = self.describe(right)
        a, b = self._embed(left_text), self._embed(right_text)
        if len(a) != len(b):
            raise ValueError("embedder returned vectors with inconsistent dimensions")
        norm_a = math.sqrt(sum(value * value for value in a))
        norm_b = math.sqrt(sum(value * value for value in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)

    def is_match(self, left: Any, right: Any) -> bool:
        return self.similarity(left, right) >= self.threshold
