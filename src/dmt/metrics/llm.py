"""LLM-specific evaluation metrics.

These complement DMT's core RMSE/bias/skill_score with metrics
appropriate for evaluating language model outputs on QA tasks.
"""

import numpy as np
import pandas as pd


def exact_match(expected: np.ndarray, response: np.ndarray) -> float:
    """Fraction of responses that exactly match the expected answer.

    Comparison is case-insensitive and strips whitespace.
    """
    if len(expected) == 0:
        return 0.0
    matches = sum(
        str(e).strip().lower() == str(r).strip().lower()
        for e, r in zip(expected, response)
    )
    return float(matches / len(expected))


def fuzzy_match(expected: np.ndarray, response: np.ndarray) -> float:
    """Fraction of responses that contain the expected answer as a substring.

    More lenient than exact_match: if the model says "The answer is Paris"
    and the expected answer is "Paris", this counts as a match.
    """
    if len(expected) == 0:
        return 0.0
    matches = sum(
        str(e).strip().lower() in str(r).strip().lower()
        for e, r in zip(expected, response)
    )
    return float(matches / len(expected))


def mean_latency(latencies: np.ndarray) -> float:
    """Mean response latency in seconds.

    Only meaningful for API-based models that report latency.
    Returns 0.0 if no latency data.
    """
    if len(latencies) == 0:
        return 0.0
    return float(np.mean(latencies))


def compute_llm_metrics(
    expected: np.ndarray,
    response: np.ndarray,
    latencies: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all LLM metrics for a single model's responses.

    Parameters
    ----------
    expected : array of expected answers
    response : array of model responses
    latencies : optional array of per-question latencies (seconds)

    Returns dict with: exact_match, fuzzy_match, and optionally mean_latency.
    """
    result = {
        "exact_match": exact_match(expected, response),
        "fuzzy_match": fuzzy_match(expected, response),
    }
    if latencies is not None and len(latencies) > 0:
        result["mean_latency"] = mean_latency(latencies)
    return result
