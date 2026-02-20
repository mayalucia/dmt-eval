"""Tests for Lesson 06: structured output (agent_verdict.json).

Tests the verdict schema, JSON-based grading, and prose fallback.
"""

import json
from pathlib import Path

import pytest

from dmt.agent.verdict import AgentVerdict, VERDICT_FILENAME
from dmt.agent.grader import grade_drug_efficacy, grade_weather, grade_output


# ── AgentVerdict tests ───────────────────────────────────────────────────

class TestAgentVerdict:

    def test_round_trip_json(self, tmp_path):
        """Verdict should survive save/load cycle."""
        v = AgentVerdict(
            best_model="CalibratedModel",
            best_reason="lowest RMSE",
            worst_model="LinearModel",
            worst_reason="fails on sigmoidal data",
            reference_model="LinearModel",
            summary="The CalibratedModel is best.",
        )
        v.save(tmp_path)
        loaded = AgentVerdict.load(tmp_path)

        assert loaded.best_model == v.best_model
        assert loaded.worst_model == v.worst_model
        assert loaded.reference_model == v.reference_model
        assert loaded.summary == v.summary

    def test_to_json_is_valid(self):
        """to_json should produce parseable JSON."""
        v = AgentVerdict(
            best_model="A", best_reason="r",
            worst_model="B", worst_reason="r",
            reference_model="C", summary="s",
        )
        data = json.loads(v.to_json())
        assert data["best_model"] == "A"
        assert data["worst_model"] == "B"

    def test_extra_field_preserved(self, tmp_path):
        """Extra domain-specific fields should survive round-trip."""
        v = AgentVerdict(
            best_model="A", best_reason="r",
            worst_model="B", worst_reason="r",
            reference_model="C", summary="s",
            extra={"rmse": 0.42, "domain": "weather"},
        )
        v.save(tmp_path)
        loaded = AgentVerdict.load(tmp_path)
        assert loaded.extra["rmse"] == 0.42
        assert loaded.extra["domain"] == "weather"

    def test_load_missing_raises(self, tmp_path):
        """Loading from a directory without verdict should raise."""
        with pytest.raises(FileNotFoundError):
            AgentVerdict.load(tmp_path)

    def test_verdict_filename_constant(self):
        """The filename should be agent_verdict.json."""
        assert VERDICT_FILENAME == "agent_verdict.json"


# ── JSON grading: Drug Efficacy ──────────────────────────────────────────

def _make_report(tmp_path: Path) -> None:
    """Create a minimal valid report.md."""
    (tmp_path / "report.md").write_text(
        "# Report\n\n"
        "## Abstract\n\ntext\n\n"
        "## Introduction\n\ntext\n\n"
        "## Methods\n\ntext\n\n"
        "## Results\n\ntext\n\n"
        "## Discussion\n\ntext\n\n"
        "## Conclusion\n\ntext\n"
    )


def test_drug_grader_json_all_pass(tmp_path):
    """Drug grader should pass with correct JSON verdict."""
    _make_report(tmp_path)
    verdict = {
        "best_model": "CalibratedModel",
        "best_reason": "lowest RMSE on dose-response",
        "worst_model": "LinearModel",
        "worst_reason": "fails on sigmoidal data",
        "reference_model": "LinearModel",
        "summary": "Calibrated is best, Linear is worst.",
    }
    (tmp_path / "agent_verdict.json").write_text(json.dumps(verdict))

    grade = grade_drug_efficacy(tmp_path)
    assert grade.all_passed, grade.summary()
    assert grade.score == 1.0
    # Verify it used JSON path (no "prose fallback" in detail)
    for c in grade.criteria:
        assert "prose fallback" not in c.detail


def test_drug_grader_json_wrong_best(tmp_path):
    """Drug grader should fail when JSON names the wrong best model."""
    _make_report(tmp_path)
    verdict = {
        "best_model": "LinearModel",
        "best_reason": "somehow",
        "worst_model": "LinearModel",
        "worst_reason": "actually this",
        "reference_model": "LinearModel",
        "summary": "Wrong.",
    }
    (tmp_path / "agent_verdict.json").write_text(json.dumps(verdict))

    grade = grade_drug_efficacy(tmp_path)
    best_criterion = next(c for c in grade.criteria if c.name == "identifies_best")
    assert not best_criterion.passed
    assert "expected Calibrated" in best_criterion.detail


def test_drug_grader_json_wrong_worst(tmp_path):
    """Drug grader should fail when JSON names the wrong worst model."""
    _make_report(tmp_path)
    verdict = {
        "best_model": "CalibratedModel",
        "best_reason": "good",
        "worst_model": "SigmoidModel",
        "worst_reason": "not really",
        "reference_model": "LinearModel",
        "summary": "Wrong worst.",
    }
    (tmp_path / "agent_verdict.json").write_text(json.dumps(verdict))

    grade = grade_drug_efficacy(tmp_path)
    worst_criterion = next(c for c in grade.criteria if c.name == "identifies_worst")
    assert not worst_criterion.passed
    assert "expected Linear" in worst_criterion.detail


# ── JSON grading: Weather ────────────────────────────────────────────────

def test_weather_grader_json_all_pass(tmp_path):
    """Weather grader should pass with correct JSON verdict."""
    _make_report(tmp_path)
    verdict = {
        "best_model": "NoisyRegressionModel",
        "best_reason": "lowest RMSE",
        "worst_model": "PersistenceModel",
        "worst_reason": "cannot adapt to seasonal shifts",
        "reference_model": "ClimatologyModel",
        "summary": "Regression is best, Climatology is baseline.",
    }
    (tmp_path / "agent_verdict.json").write_text(json.dumps(verdict))

    grade = grade_weather(tmp_path)
    assert grade.all_passed, grade.summary()
    assert grade.score == 1.0


def test_weather_grader_json_wrong_reference(tmp_path):
    """Weather grader should fail when reference model is wrong."""
    _make_report(tmp_path)
    verdict = {
        "best_model": "NoisyRegressionModel",
        "best_reason": "lowest RMSE",
        "worst_model": "PersistenceModel",
        "worst_reason": "bad",
        "reference_model": "PersistenceModel",
        "summary": "wrong ref.",
    }
    (tmp_path / "agent_verdict.json").write_text(json.dumps(verdict))

    grade = grade_weather(tmp_path)
    ref_criterion = next(c for c in grade.criteria if c.name == "identifies_reference")
    assert not ref_criterion.passed
    assert "expected Climatology" in ref_criterion.detail


# ── Prose fallback ───────────────────────────────────────────────────────

def test_drug_grader_prose_fallback(tmp_path):
    """Drug grader should fall back to prose when no JSON exists."""
    _make_report(tmp_path)
    (tmp_path / "agent_summary.txt").write_text(
        "The CalibratedModel achieves the best performance with lowest RMSE. "
        "The LinearModel fails to capture the sigmoidal dose-response."
    )

    grade = grade_drug_efficacy(tmp_path)
    assert grade.all_passed, grade.summary()
    # Verify it used the fallback path
    for c in grade.criteria:
        if c.name in ("identifies_best", "identifies_worst"):
            assert "prose fallback" in c.detail


def test_weather_grader_prose_fallback(tmp_path):
    """Weather grader should fall back to prose when no JSON exists."""
    _make_report(tmp_path)
    (tmp_path / "agent_summary.txt").write_text(
        "The NoisyRegression model achieves the best performance "
        "with the lowest RMSE across all cities. "
        "Compared to the Climatology baseline, it shows significant "
        "improvement in capturing day-to-day weather variability."
    )

    grade = grade_weather(tmp_path)
    assert grade.all_passed, grade.summary()
    for c in grade.criteria:
        if c.name in ("identifies_best", "identifies_reference"):
            assert "prose fallback" in c.detail


# ── Simulated agent end-to-end ───────────────────────────────────────────

def test_simulated_agent_writes_verdict(tmp_path):
    """The simulated agent should produce agent_verdict.json."""
    from dmt.agent.runner import run_agent

    script = Path(__file__).parent.parent / "scripts" / "simulated_agent.py"
    output_dir = tmp_path / "agent_output"
    result = run_agent(script, output_dir)

    assert result.success, f"Agent failed: {result.stderr}"
    assert result.verdict_exists, "No agent_verdict.json produced"

    verdict = AgentVerdict.load(output_dir)
    assert "calibrat" in verdict.best_model.lower()
    assert "linear" in verdict.worst_model.lower()


def test_simulated_agent_passes_json_grading(tmp_path):
    """The simulated agent should score 4/4 via JSON grading."""
    from dmt.agent.runner import run_agent

    script = Path(__file__).parent.parent / "scripts" / "simulated_agent.py"
    output_dir = tmp_path / "agent_output"
    result = run_agent(script, output_dir)
    assert result.success, f"Agent failed: {result.stderr}"

    grade = grade_drug_efficacy(output_dir)
    print("\n" + grade.summary())

    assert grade.all_passed, grade.summary()
    # Verify it used JSON path
    for c in grade.criteria:
        assert "prose fallback" not in c.detail
