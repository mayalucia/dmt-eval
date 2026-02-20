"""Measurement functions for forecast verification.

Each measurement takes a merged DataFrame with 'temperature' (observed)
and 'predicted' columns, and returns a scalar metric value.
"""

import numpy as np
import pandas as pd


def rmse(df: pd.DataFrame) -> float:
    """Root Mean Square Error between observed and predicted."""
    return float(np.sqrt(np.mean((df["temperature"] - df["predicted"]) ** 2)))


def bias(df: pd.DataFrame) -> float:
    """Mean bias (predicted - observed).  Positive = warm bias."""
    return float(np.mean(df["predicted"] - df["temperature"]))


def skill_score(df: pd.DataFrame, reference_rmse: float) -> float:
    """Skill score relative to a reference forecast.

    SS = 1 - RMSE_model / RMSE_reference.
    Positive means the model beats the reference.
    """
    model_rmse = rmse(df)
    if reference_rmse == 0:
        return 0.0
    return float(1.0 - model_rmse / reference_rmse)


def compute_metrics(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_rmse: float | None = None,
) -> dict:
    """Compute all verification metrics for a single model.

    Parameters
    ----------
    observations : DataFrame with city, day, temperature, season
    predictions : DataFrame with city, day, predicted, season
    reference_rmse : RMSE of the reference forecast (for skill score)

    Returns a dict of metric_name -> value.
    """
    merged = observations.merge(predictions, on=["city", "day", "season"])
    result = {
        "rmse": rmse(merged),
        "bias": bias(merged),
    }
    if reference_rmse is not None:
        result["skill_score"] = skill_score(merged, reference_rmse)
    return result


def compute_metrics_by_group(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    group_by: str = "city",
    reference_rmse: float | None = None,
) -> pd.DataFrame:
    """Compute metrics broken down by a grouping variable.

    Returns a DataFrame with one row per group, columns for each metric.
    """
    merged = observations.merge(predictions, on=["city", "day", "season"])
    rows = []
    for group_val, group_df in merged.groupby(group_by):
        row = {group_by: group_val}
        row["rmse"] = rmse(group_df)
        row["bias"] = bias(group_df)
        if reference_rmse is not None:
            row["skill_score"] = skill_score(group_df, reference_rmse)
        row["n"] = len(group_df)
        rows.append(row)
    return pd.DataFrame(rows)
