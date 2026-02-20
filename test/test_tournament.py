"""Tests for weather brief, tournament runner, and expanded grader.

Lesson 07: total_count updated from 4 to 5 (verdict_valid criterion added).
"""

import json
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
    # Use JSON verdict (primary path since Lesson 06)
    verdict = {
        "best_model": "NoisyRegressionModel",
        "best_reason": "lowest RMSE",
        "worst_model": "PersistenceModel",
        "worst_reason": "cannot adapt",
        "reference_model": "ClimatologyModel",
        "summary": "Regression is best, Climatology is baseline.",
    }
    (tmp_path / "agent_verdict.json").write_text(json.dumps(verdict))

    grade = grade_weather(tmp_path)
    assert grade.all_passed, grade.summary()
    assert grade.score == 1.0


def test_weather_grader_missing_report(tmp_path):
    """Weather grader should handle missing report gracefully."""
    grade = grade_weather(tmp_path)
    assert grade.pass_count == 0
    assert grade.total_count == 5  # report, sections, verdict_valid, best, reference


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
    """A hand-written weather agent should score 5/5 via JSON verdict."""
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

    # Write a correct JSON verdict
    verdict = {
        "best_model": "NoisyRegressionModel",
        "best_reason": "lowest RMSE across all cities",
        "worst_model": "PersistenceModel",
        "worst_reason": "cannot adapt to seasonal shifts",
        "reference_model": "ClimatologyModel",
        "summary": (
            "The NoisyRegression model achieves the lowest RMSE. "
            "Relative to the Climatology baseline, it captures "
            "day-to-day temperature variability that simpler models miss."
        ),
    }
    verdict_path = output_dir / "agent_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2))

    grade = grade_weather(output_dir)
    assert grade.all_passed, grade.summary()
