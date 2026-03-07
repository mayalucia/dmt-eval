"""The Level 0 entry point: dmt.evaluate().

Domain-agnostic: the scenario descriptor tells the evaluator which
columns are which.  The evaluator is just the engine.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dmt.document.renderer import render_markdown


# ── Metrics (inlined for zero-dependency core) ──────────────────────────────

def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def _bias(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(predicted - observed))


def _skill_score(model_rmse: float, reference_rmse: float) -> float:
    if reference_rmse == 0:
        return 0.0
    return float(1.0 - model_rmse / reference_rmse)


# ── Scenario Descriptor ────────────────────────────────────────────────────

@dataclass
class Scenario:
    """Describes the shape of a validation scenario.

    This is how a domain tells DMT which columns matter.
    """
    # Column names
    observed_col: str = "observed"
    predicted_col: str = "predicted"
    entity_col: str = "entity"
    merge_on: list[str] = field(default_factory=lambda: ["entity", "step"])

    # Grouping for stratified analysis
    group_by: list[str] = field(default_factory=lambda: ["entity"])

    # Narrative templates
    domain_name: str = "model"
    observation_description: str = "observations"
    entity_description: str = "entities"


# ── Pre-built scenarios ────────────────────────────────────────────────────

WEATHER = Scenario(
    observed_col="temperature",
    predicted_col="predicted",
    entity_col="city",
    merge_on=["city", "day", "season"],
    group_by=["city", "season"],
    domain_name="weather prediction",
    observation_description="synthetic daily temperature observations",
    entity_description="European cities",
)

DRUG_EFFICACY = Scenario(
    observed_col="efficacy",
    predicted_col="predicted",
    entity_col="compound",
    merge_on=["compound", "dose"],
    group_by=["compound", "dose"],
    domain_name="drug efficacy prediction",
    observation_description="dose-response efficacy measurements",
    entity_description="pharmaceutical compounds",
)

EQUITY_FORECAST = Scenario(
    observed_col="return",
    predicted_col="predicted_return",
    entity_col="ticker",
    merge_on=["ticker", "date", "regime"],
    group_by=["ticker", "regime"],
    domain_name="equity return forecasting",
    observation_description="synthetic daily equity returns (GARCH-t)",
    entity_description="synthetic equities",
)


# ── The evaluator ──────────────────────────────────────────────────────────

def _is_numeric(arr: np.ndarray) -> bool:
    """Check if an array contains numeric data."""
    return np.issubdtype(arr.dtype, np.number)


def _exact_match(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of exact matches (case-insensitive, stripped)."""
    matches = sum(
        str(o).strip().lower() == str(p).strip().lower()
        for o, p in zip(observed, predicted)
    )
    return float(matches / len(observed)) if len(observed) > 0 else 0.0


def _fuzzy_match(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of responses containing the expected answer."""
    matches = sum(
        str(o).strip().lower() in str(p).strip().lower()
        for o, p in zip(observed, predicted)
    )
    return float(matches / len(observed)) if len(observed) > 0 else 0.0


def _compute_metrics(merged: pd.DataFrame, obs_col: str, pred_col: str,
                     reference_rmse: float | None = None) -> dict:
    """Compute metrics on a merged DataFrame.

    For numeric data: RMSE, bias, skill score.
    For string data: exact_match, fuzzy_match.
    """
    observed = merged[obs_col].values
    predicted = merged[pred_col].values

    if _is_numeric(observed) and _is_numeric(predicted):
        result = {
            "rmse": _rmse(observed, predicted),
            "bias": _bias(observed, predicted),
        }
        if reference_rmse is not None:
            result["skill_score"] = _skill_score(result["rmse"], reference_rmse)
    else:
        result = {
            "exact_match": _exact_match(observed, predicted),
            "fuzzy_match": _fuzzy_match(observed, predicted),
        }
    return result


def _compute_by_group(merged: pd.DataFrame, obs_col: str, pred_col: str,
                      group_col: str, reference_rmse: float | None = None
                      ) -> pd.DataFrame:
    """Compute metrics broken down by a grouping column."""
    rows = []
    for val, grp in merged.groupby(group_col):
        row = {group_col: val}
        row.update(_compute_metrics(grp, obs_col, pred_col, reference_rmse))
        row["n"] = len(grp)
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_models(models: list) -> list:
    """Resolve any string model specs to model objects."""
    resolved = []
    for m in models:
        if isinstance(m, str):
            from dmt.models import resolve
            resolved.append(resolve(m))
        else:
            resolved.append(m)
    return resolved


def evaluate(
    models: list,
    observations: pd.DataFrame,
    scenario: Scenario | None = None,
    reference_model=None,
    output_dir: str | Path = "./dmt_report",
    title: str = "Model Evaluation Report",
) -> Path:
    """Evaluate models against observations and produce a LabReport.

    Parameters
    ----------
    models : list of model objects or strings
        Each must have .name (str) and .predict(observations) -> DataFrame,
        or be a string model spec (e.g. "echo", "anthropic/claude-haiku-4-5-20251001").
    observations : DataFrame
        Ground truth.
    scenario : Scenario
        Describes column names, merge keys, and grouping.
        If None, attempts to auto-detect (falls back to WEATHER).
    reference_model : optional
        Baseline for skill scores.  If None, uses models[0].
    output_dir : path
        Where to write the report.
    title : str
        Report title.

    Returns the path to the generated report.
    """
    models = _resolve_models(models)
    if isinstance(reference_model, str):
        from dmt.models import resolve
        reference_model = resolve(reference_model)

    if scenario is None:
        scenario = WEATHER

    obs_col = scenario.observed_col
    pred_col = scenario.predicted_col
    merge_on = scenario.merge_on
    entity_col = scenario.entity_col

    reference = reference_model or models[0]
    ref_predictions = reference.predict(observations)
    ref_merged = observations.merge(ref_predictions, on=merge_on)

    # Detect whether this is a numeric or string-valued scenario
    numeric = (_is_numeric(ref_merged[obs_col].values)
               and _is_numeric(ref_merged[pred_col].values))

    reference_rmse = None
    if numeric:
        reference_rmse = _rmse(ref_merged[obs_col].values,
                               ref_merged[pred_col].values)

    # ── Run all models ──────────────────────────────────────────────────
    all_summary = []
    all_by_group = {}

    for model in models:
        predictions = model.predict(observations)
        merged = observations.merge(predictions, on=merge_on)
        summary = _compute_metrics(merged, obs_col, pred_col, reference_rmse)
        summary["model"] = model.name
        all_summary.append(summary)

        for group_col in scenario.group_by:
            if group_col not in all_by_group:
                all_by_group[group_col] = []
            by_g = _compute_by_group(merged, obs_col, pred_col,
                                     group_col, reference_rmse)
            by_g["model"] = model.name
            all_by_group[group_col].append(by_g)

    if numeric:
        metric_cols = ["model", "rmse", "bias", "skill_score"]
        primary_metric = "rmse"
        best_is_min = True
    else:
        metric_cols = ["model", "exact_match", "fuzzy_match"]
        primary_metric = "exact_match"
        best_is_min = False

    summary_df = pd.DataFrame(all_summary)[metric_cols]
    grouped_dfs = {
        col: pd.concat(frames, ignore_index=True)
        for col, frames in all_by_group.items()
    }

    # ── Assemble sections ───────────────────────────────────────────────
    sections = OrderedDict()

    n_entities = observations[entity_col].nunique()

    if numeric:
        metrics_description = (
            f"Models are compared using RMSE, bias, and skill score "
            f"(relative to {reference.name})."
        )
        methods_narrative = (
            "**Metrics**:\n\n"
            "- *RMSE*: Root mean square error.  Lower is better.\n"
            "- *Bias*: Mean (predicted - observed).  Zero is unbiased.\n"
            f"- *Skill Score*: 1 - RMSE_model / RMSE_{reference.name}.  "
            "Positive means the model beats the reference.\n\n"
            f"**Grouping**: Results are stratified by "
            + ", ".join(scenario.group_by) + "."
        )
    else:
        metrics_description = (
            "Models are compared using exact match and fuzzy match accuracy."
        )
        methods_narrative = (
            "**Metrics**:\n\n"
            "- *Exact Match*: Fraction of responses matching expected "
            "answer exactly (case-insensitive).  Higher is better.\n"
            "- *Fuzzy Match*: Fraction of responses containing the "
            "expected answer as a substring.  Higher is better.\n\n"
            f"**Grouping**: Results are stratified by "
            + ", ".join(scenario.group_by) + "."
        )

    sections["abstract"] = {
        "name": "Abstract",
        "narrative": (
            f"We evaluate {len(models)} {scenario.domain_name} models against "
            f"{scenario.observation_description} for "
            f"{n_entities} {scenario.entity_description}.  "
            f"{metrics_description}"
        ),
    }

    sections["introduction"] = {
        "name": "Introduction",
        "narrative": (
            f"This report compares {len(models)} models on a "
            f"{scenario.domain_name} task:\n\n"
            + "\n".join(f"- **{m.name}**" for m in models)
        ),
    }

    sections["methods"] = {
        "name": "Methods",
        "narrative": methods_narrative,
    }

    sections["results"] = {
        "name": "Results",
        "narrative": "### Overall Performance",
        "data": summary_df,
    }

    for group_col, gdf in grouped_dfs.items():
        label = f"results_by_{group_col}"
        sections[label] = {
            "name": f"Results by {group_col.replace('_', ' ').title()}",
            "narrative": (
                f"Performance broken down by {group_col}."
            ),
            "data": gdf,
        }

    # ── Discussion: find best model overall and per group ──────────────
    if best_is_min:
        best_idx = summary_df[primary_metric].idxmin()
    else:
        best_idx = summary_df[primary_metric].idxmax()
    best_overall = summary_df.loc[best_idx, "model"]

    qualifier = "lowest" if best_is_min else "highest"
    discussion_parts = [
        f"**{best_overall}** achieves the {qualifier} overall "
        f"{primary_metric}.\n",
    ]
    for group_col, gdf in grouped_dfs.items():
        for val in gdf[group_col].unique():
            subset = gdf[gdf[group_col] == val]
            if not subset.empty:
                if best_is_min:
                    best = subset.loc[subset[primary_metric].idxmin(), "model"]
                else:
                    best = subset.loc[subset[primary_metric].idxmax(), "model"]
                discussion_parts.append(
                    f"- *{group_col}={val}*: best is **{best}**")

    sections["discussion"] = {
        "name": "Discussion",
        "narrative": "\n".join(discussion_parts),
    }

    sections["conclusion"] = {
        "name": "Conclusion",
        "narrative": (
            f"This evaluation compared {len(models)} models on a "
            f"{scenario.domain_name} task, producing metrics with "
            f"stratification by {', '.join(scenario.group_by)}.  "
            f"The report was generated automatically by DMT."
        ),
    }

    return render_markdown(title, sections, output_dir)
