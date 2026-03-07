"""Offline baseline models for LLM QA evaluation.

These run without API keys and serve as lower/upper bounds
for comparison with real LLM models.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class EchoModel:
    """Returns the question as the answer.  The worst possible baseline."""
    name: str = "Echo"

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "question_id": observations["question_id"],
            "response": observations["question"],
        })


@dataclass
class RandomModel:
    """Picks a random answer from the expected answers in the dataset."""
    name: str = "Random"
    seed: int = 42

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        pool = observations["expected"].values
        chosen = rng.choice(pool, size=len(observations), replace=True)
        return pd.DataFrame({
            "question_id": observations["question_id"],
            "response": chosen,
        })


@dataclass
class TemplateModel:
    """Rule-based pattern matching.  Gets arithmetic and simple factual right."""
    name: str = "Template"

    _KNOWN: dict = field(default_factory=lambda: {
        # Arithmetic patterns
        "2+2": "4", "2 + 2": "4",
        "3+5": "8", "3 + 5": "8",
        "10*10": "100", "10 * 10": "100",
        "100/4": "25", "100 / 4": "25",
        # Common factual
        "capital of france": "Paris",
        "capital of japan": "Tokyo",
        "capital of germany": "Berlin",
        "boiling point of water": "100",
        "speed of light": "299792458",
    })

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        responses = []
        for row in observations.itertuples():
            q = row.question.lower().strip().rstrip("?. ")
            answer = self._KNOWN.get(q)
            if answer is None:
                # Try substring matching
                for pattern, ans in self._KNOWN.items():
                    if pattern in q:
                        answer = ans
                        break
            if answer is None:
                answer = "I don't know"
            responses.append(answer)

        return pd.DataFrame({
            "question_id": observations["question_id"],
            "response": responses,
        })
