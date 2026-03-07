"""Anthropic model adapter for LLM QA evaluation.

Requires ANTHROPIC_API_KEY in the environment.
"""

import os
import time
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AnthropicModel:
    """Call Anthropic Messages API for each question."""
    model_id: str = "claude-haiku-4-5-20251001"
    name: str = ""
    max_tokens: int = 256
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = f"anthropic/{self.model_id}"

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
                )
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        client = self._get_client()
        responses = []
        for row in observations.itertuples():
            start = time.monotonic()
            response = client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": row.question}],
            )
            elapsed = time.monotonic() - start
            text = response.content[0].text.strip()
            responses.append({
                "question_id": row.question_id,
                "response": text,
                "latency": elapsed,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            })
        return pd.DataFrame(responses)
