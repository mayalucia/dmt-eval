"""Tests for weather brief, tournament runner, and expanded grader."""

from pathlib import Path

import pytest

from dmt.agent.brief import WEATHER_BRIEF, DRUG_EFFICACY_BRIEF
from dmt.agent.grader import grade_weather, grade_drug_efficacy, grade_output


# ── Weather brief tests ──────────────────────────────────────────────────────

def test_weather_brief_is_complete():
    """Weather brief should contain all necessary information."""
    prompt = WEATHER_BRIEF.to_prompt()

    assert "dmt.evaluate" in prompt
    assert "dmt.scenario.weather" in prompt
    assert "WEATHER" in prompt
    assert "generate_observations" in prompt
    assert "PersistenceModel" in prompt
    assert "ClimatologyModel" in prompt
    assert "NoisyRegressionModel" in prompt
    assert "evaluate(models=" in prompt


# ── Weather grader tests ─────────────────────────────────────────────────────

def test_weather_grader_all_pass(tmp_path):
    """Weather grader should pass when output is correct."""
    # Create a minimal correct report
    report = tmp_path / "report.md"
    report.write_text(
        "# Weather Report\n\n"
        "## Abstract\n\nWe evaluate...\n\n"
        "## Introduction\n\n...\n\n"
        "## Methods\n\n...\n\n"
        "## Results\n\n...\n\n"
        "## Discussion\n\n...\n\n"
        "## Conclusion\n\n...\n"
    )
    summary = tmp_path / "agent_summary.txt"
    summary.write_text(
        "The NoisyRegression model achieves the best performance "
        "with the lowest RMSE across all cities. "
        "Compared to the Climatology baseline, it shows significant "
        "improvement in capturing day-to-day weather variability. "
        "Persistence performs reasonably but cannot match the "
        "regression model's skill."
    )

    grade = grade_weather(tmp_path)
    assert grade.all_passed, grade.summary()
    assert grade.score == 1.0


def test_weather_grader_missing_report(tmp_path):
    """Weather grader should handle missing report gracefully."""
    grade = grade_weather(tmp_path)
    assert grade.pass_count == 0
    assert grade.total_count == 4


# ── Generic grade_output dispatch ────────────────────────────────────────────

def test_grade_output_dispatches_drug():
    """grade_output should dispatch to the drug grader."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        grade = grade_output("Drug Efficacy Validation", tmp)
        assert grade.agent_name == "Drug Efficacy Validation"


def test_grade_output_dispatches_weather():
    """grade_output should dispatch to the weather grader."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        grade = grade_output("Weather Prediction Validation", tmp)
        assert grade.agent_name == "Weather Prediction Validation"


def test_grade_output_unknown_raises():
    """grade_output should raise for unknown brief names."""
    with pytest.raises(ValueError, match="No grader"):
        grade_output("Unknown Brief", "/tmp")


# ── Simulated weather agent test ─────────────────────────────────────────────

def test_simulated_weather_agent(tmp_path):
    """A hand-written weather agent should score 4/4."""
    from dmt.evaluate import evaluate, WEATHER
    from dmt.scenario.weather import (
        generate_observations,
        PersistenceModel,
        ClimatologyModel,
        NoisyRegressionModel,
    )

    output_dir = tmp_path / "weather_output"

    obs = generate_observations(n_days=365, seed=42)
    persistence = PersistenceModel()
    climatology = ClimatologyModel()
    regression = NoisyRegressionModel(alpha=0.7, noise_std=0.5)

    evaluate(
        models=[persistence, climatology, regression],
        observations=obs,
        scenario=WEATHER,
        reference_model=climatology,
        output_dir=output_dir,
        title="Weather Prediction Model Comparison",
    )

    # Write a correct summary
    summary_path = output_dir / "agent_summary.txt"
    summary_path.write_text(
        "The NoisyRegression model achieves the lowest RMSE, "
        "demonstrating the best predictive skill across all cities. "
        "Relative to the Climatology baseline, it captures "
        "day-to-day temperature variability that simpler models miss. "
        "Persistence performs worst as it cannot adapt to seasonal shifts."
    )

    grade = grade_weather(output_dir)
    assert grade.all_passed, grade.summary()
