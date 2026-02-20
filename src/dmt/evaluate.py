"""The Level 0 entry point: dmt.evaluate().

Takes models + observations, runs measurements, produces a LabReport.
"""

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from dmt.measurement import compute_metrics, compute_metrics_by_group
from dmt.document.renderer import render_markdown


def evaluate(
    models: list,
    observations: pd.DataFrame,
    reference_model=None,
    group_by: str = "city",
    output_dir: str | Path = "./dmt_report",
    title: str = "Model Evaluation Report",
) -> Path:
    """Evaluate models against observations and produce a LabReport.

    Parameters
    ----------
    models : list of model objects
        Each must have .name (str) and .predict(observations) -> DataFrame.
    observations : DataFrame
        Ground truth with columns: city, day, temperature, season.
    reference_model : optional
        Model to use as baseline for skill scores.  If None, uses the
        first model in the list.
    group_by : str
        Column to break down metrics by (default: "city").
    output_dir : path
        Where to write the report.
    title : str
        Report title.

    Returns the path to the generated report.
    """
    reference = reference_model or models[0]
    ref_predictions = reference.predict(observations)
    ref_merged = observations.merge(ref_predictions, on=["city", "day", "season"])
    from dmt.measurement import rmse as _rmse
    reference_rmse = _rmse(ref_merged)

    # ── Run all models ──────────────────────────────────────────────────
    all_summary = []
    all_by_group = {}
    all_by_season = {}

    for model in models:
        predictions = model.predict(observations)
        summary = compute_metrics(observations, predictions, reference_rmse)
        summary["model"] = model.name
        all_summary.append(summary)

        by_group = compute_metrics_by_group(
            observations, predictions, group_by, reference_rmse)
        by_group["model"] = model.name
        all_by_group[model.name] = by_group

        by_season = compute_metrics_by_group(
            observations, predictions, "season", reference_rmse)
        by_season["model"] = model.name
        all_by_season[model.name] = by_season

    summary_df = pd.DataFrame(all_summary)[["model", "rmse", "bias", "skill_score"]]
    grouped_df = pd.concat(all_by_group.values(), ignore_index=True)
    season_df = pd.concat(all_by_season.values(), ignore_index=True)

    # ── Assemble the LabReport sections ─────────────────────────────────
    sections = OrderedDict()

    sections["abstract"] = {
        "name": "Abstract",
        "narrative": (
            f"We evaluate {len(models)} weather prediction models against "
            f"synthetic daily temperature observations for "
            f"{observations['city'].nunique()} European cities over "
            f"{observations['day'].nunique()} days.  "
            f"Models are compared using RMSE, bias, and skill score "
            f"(relative to {reference.name}).  "
            f"Results are broken down by {group_by} and by season."
        ),
    }

    sections["introduction"] = {
        "name": "Introduction",
        "narrative": (
            "Weather prediction is a canonical test bed for model comparison.  "
            "Given the same set of observations, different forecasting strategies "
            "trade off between capturing the seasonal cycle (climatological skill) "
            "and tracking day-to-day variability (persistence skill).  "
            "A useful model must outperform naive baselines on metrics that matter "
            "to the end user.\n\n"
            "This report compares:\n\n"
            + "\n".join(f"- **{m.name}**" for m in models)
        ),
    }

    sections["methods"] = {
        "name": "Methods",
        "narrative": (
            "**Observations**: Synthetic daily temperature generated from a "
            "sinusoidal annual cycle with AR(1) autocorrelated weather noise "
            "(rho=0.7).  City-specific parameters for mean temperature, "
            "seasonal amplitude, and noise standard deviation.\n\n"
            "**Metrics**:\n\n"
            "- *RMSE*: Root mean square error (degrees C).  Lower is better.\n"
            "- *Bias*: Mean (predicted - observed).  Zero is unbiased.\n"
            f"- *Skill Score*: 1 - RMSE_model / RMSE_{reference.name}.  "
            "Positive means the model beats the reference.\n\n"
            f"**Grouping**: Results are stratified by {group_by} and by season."
        ),
    }

    sections["results"] = {
        "name": "Results",
        "narrative": "### Overall Performance",
        "data": summary_df,
    }

    sections["results_by_city"] = {
        "name": f"Results by {group_by.title()}",
        "narrative": (
            f"Performance broken down by {group_by} reveals geographic "
            "variation in model skill."
        ),
        "data": grouped_df,
    }

    sections["results_by_season"] = {
        "name": "Results by Season",
        "narrative": (
            "Seasonal stratification tests whether model skill depends "
            "on the time of year."
        ),
        "data": season_df,
    }

    # ── Find best model per group ──────────────────────────────────────
    best_overall = summary_df.loc[summary_df["rmse"].idxmin(), "model"]
    discussion_parts = [
        f"**{best_overall}** achieves the lowest overall RMSE.\n",
    ]
    for season in ["winter", "spring", "summer", "autumn"]:
        season_data = season_df[season_df["season"] == season]
        if not season_data.empty:
            best = season_data.loc[season_data["rmse"].idxmin(), "model"]
            discussion_parts.append(f"- *{season.title()}*: best model is **{best}**")

    sections["discussion"] = {
        "name": "Discussion",
        "narrative": "\n".join(discussion_parts),
    }

    sections["conclusion"] = {
        "name": "Conclusion",
        "narrative": (
            f"The evaluation demonstrates DMT's end-to-end capability: "
            f"three models were compared against synthetic observations, "
            f"metrics computed with geographic and seasonal stratification, "
            f"and this structured report generated automatically.  "
            f"The same pattern applies to any domain — LLMs, drug discovery, "
            f"financial models — by swapping the adapter and metrics."
        ),
    }

    return render_markdown(title, sections, output_dir)
