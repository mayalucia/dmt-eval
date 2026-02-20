"""Simulated agent: follows the Drug Efficacy brief verbatim.

This script represents what a well-functioning AI agent would produce
when given the agent brief from Lesson 02.  It uses only the public
DMT API referenced in the brief.
"""

import sys
from pathlib import Path

# ── Step 0: The brief said these are our available imports ──────────────
from dmt.evaluate import evaluate, DRUG_EFFICACY
from dmt.scenario.drug_efficacy import (
    generate_observations,
    LinearModel,
    SigmoidModel,
    CalibratedModel,
)


def main(output_dir: str = "./agent_drug_report") -> dict:
    """Execute the agent brief and return results.

    Returns a dict with:
        report_path: Path to the generated report
        summary: The 3-sentence scientific summary
    """
    output_dir = Path(output_dir)

    # ── Step 1: Generate observations ─────────────────────────────────
    observations = generate_observations()

    # ── Step 2: Create model instances ────────────────────────────────
    linear = LinearModel()
    sigmoid = SigmoidModel()
    calibrated = CalibratedModel()

    # ── Step 3: Evaluate ──────────────────────────────────────────────
    report_path = evaluate(
        models=[linear, sigmoid, calibrated],
        observations=observations,
        scenario=DRUG_EFFICACY,
        reference_model=linear,
        output_dir=output_dir,
        title="Drug Efficacy Model Comparison",
    )

    # ── Step 4: Read report and write summary ─────────────────────────
    report_text = report_path.read_text()

    # Parse the Overall Performance table to find best/worst model.
    # We only read the table between "### Overall Performance" and
    # the next "##" heading — ignoring the per-group tables.
    best_model = None
    worst_model = None
    best_rmse = float("inf")
    worst_rmse = float("-inf")

    in_summary = False
    for line in report_text.split("\n"):
        if "Overall Performance" in line:
            in_summary = True
            continue
        if in_summary and line.startswith("## "):
            break
        if not in_summary:
            continue
        if "|" in line and "model" not in line.lower() and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2:
                try:
                    model_name = parts[0]
                    rmse = float(parts[1])
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model_name
                    if rmse > worst_rmse:
                        worst_rmse = rmse
                        worst_model = model_name
                except (ValueError, IndexError):
                    continue

    summary = (
        f"The {best_model} model achieves the lowest RMSE ({best_rmse:.2f}), "
        f"outperforming all other models on the drug efficacy prediction task. "
        f"The {worst_model} model performs worst because a linear assumption "
        f"fundamentally fails to capture the sigmoidal dose-response relationship "
        f"described by Hill equation kinetics. "
        f"This demonstrates that model structure must match the underlying biology — "
        f"even a miscalibrated sigmoid outperforms a well-fitted line."
    )

    # Write summary to file
    summary_path = output_dir / "agent_summary.txt"
    summary_path.write_text(summary)

    return {
        "report_path": str(report_path),
        "summary": summary,
        "best_model": best_model,
        "worst_model": worst_model,
    }


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./agent_drug_report"
    result = main(output_dir)
    print(f"Report: {result['report_path']}")
    print(f"Best model: {result['best_model']}")
    print(f"Worst model: {result['worst_model']}")
    print(f"\nSummary:\n{result['summary']}")
