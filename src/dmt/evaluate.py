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


# ── The evaluator ──────────────────────────────────────────────────────────

def _compute_metrics(merged: pd.DataFrame, obs_col: str, pred_col: str,
                     reference_rmse: float | None = None) -> dict:
    """Compute RMSE, bias, skill score on a merged DataFrame."""
    observed = merged[obs_col].values
    predicted = merged[pred_col].values
    result = {
        "rmse": _rmse(observed, predicted),
        "bias": _bias(observed, predicted),
    }
    if reference_rmse is not None:
        result["skill_score"] = _skill_score(result["rmse"], reference_rmse)
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
    models : list of model objects
        Each must have .name (str) and .predict(observations) -> DataFrame.
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
    if scenario is None:
        scenario = WEATHER

    obs_col = scenario.observed_col
    pred_col = scenario.predicted_col
    merge_on = scenario.merge_on
    entity_col = scenario.entity_col

    reference = reference_model or models[0]
    ref_predictions = reference.predict(observations)
    ref_merged = observations.merge(ref_predictions, on=merge_on)
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

    summary_df = pd.DataFrame(all_summary)[["model", "rmse", "bias", "skill_score"]]
    grouped_dfs = {
        col: pd.concat(frames, ignore_index=True)
        for col, frames in all_by_group.items()
    }

    # ── Assemble sections ───────────────────────────────────────────────
    sections = OrderedDict()

    n_entities = observations[entity_col].nunique()
    sections["abstract"] = {
        "name": "Abstract",
        "narrative": (
            f"We evaluate {len(models)} {scenario.domain_name} models against "
            f"{scenario.observation_description} for "
            f"{n_entities} {scenario.entity_description}.  "
            f"Models are compared using RMSE, bias, and skill score "
            f"(relative to {reference.name})."
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
        "narrative": (
            "**Metrics**:\n\n"
            "- *RMSE*: Root mean square error.  Lower is better.\n"
            "- *Bias*: Mean (predicted - observed).  Zero is unbiased.\n"
            f"- *Skill Score*: 1 - RMSE_model / RMSE_{reference.name}.  "
            "Positive means the model beats the reference.\n\n"
            f"**Grouping**: Results are stratified by "
            + ", ".join(scenario.group_by) + "."
        ),
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
    best_overall = summary_df.loc[summary_df["rmse"].idxmin(), "model"]
    discussion_parts = [
        f"**{best_overall}** achieves the lowest overall RMSE.\n",
    ]
    for group_col, gdf in grouped_dfs.items():
        for val in gdf[group_col].unique():
            subset = gdf[gdf[group_col] == val]
            if not subset.empty:
                best = subset.loc[subset["rmse"].idxmin(), "model"]
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
