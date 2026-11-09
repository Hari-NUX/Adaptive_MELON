"""Optional OpenAI embedding provider for the MELON action comparator."""
from __future__ import annotations


class OpenAIEmbedder:
    """Callable adapter using the configured OPENAI_API_KEY environment value."""

    def __init__(self, model: str = "text-embedding-3-large"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install the optional dependency with: pip install -r requirements-melon-api.txt"
            ) from exc
        self.model = model
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding
