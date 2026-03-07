"""Finance-specific evaluation metrics.

These complement DMT's core RMSE/bias/skill_score with metrics
standard in quantitative finance and risk management.
"""

import numpy as np
import pandas as pd


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean square error of return forecasts."""
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def directional_accuracy(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of days where the predicted direction matches actual.

    Direction = sign of return.  Days where the actual return is exactly
    zero are excluded (no direction to predict).
    """
    nonzero = observed != 0
    if not np.any(nonzero):
        return 0.0
    return float(np.mean(np.sign(observed[nonzero]) == np.sign(predicted[nonzero])))


def sharpe_ratio(returns: np.ndarray, annual_factor: float = 252.0) -> float:
    """Annualised Sharpe ratio (assuming zero risk-free rate).

    SR = mean(r) / std(r) * sqrt(252)
    """
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(annual_factor))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough decline in cumulative returns.

    Returns a non-negative number (the magnitude of the worst drawdown).
    """
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def var_95(returns: np.ndarray) -> float:
    """Value-at-Risk at the 95% confidence level.

    Returns the 5th percentile of the return distribution (a negative
    number for typical return series).  Interpretation: on 95% of days,
    the loss does not exceed this magnitude.
    """
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, 5))


def compute_finance_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Compute all finance metrics for a single model's predictions.

    Parameters
    ----------
    observed : array of actual returns
    predicted : array of predicted returns

    Returns dict with: rmse, directional_accuracy, sharpe_ratio,
    max_drawdown, var_95 (computed on forecast errors).
    """
    errors = observed - predicted
    return {
        "rmse": rmse(observed, predicted),
        "directional_accuracy": directional_accuracy(observed, predicted),
        "sharpe_observed": sharpe_ratio(observed),
        "max_drawdown_observed": max_drawdown(observed),
        "var_95_observed": var_95(observed),
        "sharpe_predicted": sharpe_ratio(predicted),
    }
