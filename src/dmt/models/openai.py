"""OpenAI model adapter for LLM QA evaluation.

Requires OPENAI_API_KEY in the environment.
"""

import os
import time
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OpenAIModel:
    """Call OpenAI Chat Completions API for each question."""
    model_id: str = "gpt-4o"
    name: str = ""
    max_tokens: int = 256
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = f"openai/{self.model_id}"

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. "
                    "Set it with: export OPENAI_API_KEY='sk-...'"
                )
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        client = self._get_client()
        responses = []
        for row in observations.itertuples():
            start = time.monotonic()
            response = client.chat.completions.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": row.question}],
            )
            elapsed = time.monotonic() - start
            text = response.choices[0].message.content.strip()
            usage = response.usage
            responses.append({
                "question_id": row.question_id,
                "response": text,
                "latency": elapsed,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            })
        return pd.DataFrame(responses)
