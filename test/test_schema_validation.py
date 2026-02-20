"""Tests for Lesson 07: schema validation.

Tests the validator, load_validated, and grader integration with
malformed verdicts.
"""

import json
from pathlib import Path

import pytest

from dmt.agent.verdict import (
    AgentVerdict,
    ValidationResult,
    validate_verdict,
    REQUIRED_FIELDS,
    VERDICT_FILENAME,
)
from dmt.agent.grader import grade_drug_efficacy, grade_weather


# ── Helper ───────────────────────────────────────────────────────────────

def _complete_verdict() -> dict:
    """A valid verdict dict."""
    return {
        "best_model": "CalibratedModel",
        "best_reason": "lowest RMSE",
        "worst_model": "LinearModel",
        "worst_reason": "fails on sigmoidal data",
        "reference_model": "LinearModel",
        "summary": "Calibrated wins, Linear loses.",
    }


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


# ── validate_verdict tests ───────────────────────────────────────────────

class TestValidateVerdict:

    def test_valid_complete(self):
        """Complete verdict should validate."""
        result = validate_verdict(_complete_verdict())
        assert result.valid
        assert result.errors == []

    def test_missing_one_field(self):
        """Missing a single field should produce one error."""
        data = _complete_verdict()
        del data["worst_model"]
        result = validate_verdict(data)
        assert not result.valid
        assert len(result.errors) == 1
        assert "missing: worst_model" in result.errors[0]

    def test_missing_multiple_fields(self):
        """Missing multiple fields should produce multiple errors."""
        data = {"best_model": "X"}
        result = validate_verdict(data)
        assert not result.valid
        assert len(result.errors) == 5  # missing 5 of 6 required

    def test_empty_dict(self):
        """Empty dict should fail with 6 missing-field errors."""
        result = validate_verdict({})
        assert not result.valid
        assert len(result.errors) == 6

    def test_wrong_type_int(self):
        """Integer where string expected should fail."""
        data = _complete_verdict()
        data["best_model"] = 42
        result = validate_verdict(data)
        assert not result.valid
        assert any("expected str, got int" in e for e in result.errors)

    def test_wrong_type_none(self):
        """None where string expected should fail."""
        data = _complete_verdict()
        data["summary"] = None
        result = validate_verdict(data)
        assert not result.valid
        assert any("expected str, got NoneType" in e for e in result.errors)

    def test_wrong_type_list(self):
        """List where string expected should fail."""
        data = _complete_verdict()
        data["best_reason"] = ["a", "b"]
        result = validate_verdict(data)
        assert not result.valid
        assert any("expected str, got list" in e for e in result.errors)

    def test_empty_string(self):
        """Empty string should fail."""
        data = _complete_verdict()
        data["best_model"] = ""
        result = validate_verdict(data)
        assert not result.valid
        assert any("empty string" in e for e in result.errors)

    def test_whitespace_only(self):
        """Whitespace-only string should fail."""
        data = _complete_verdict()
        data["best_model"] = "   "
        result = validate_verdict(data)
        assert not result.valid
        assert any("empty string" in e for e in result.errors)

    def test_extra_fields_ok(self):
        """Extra fields should not cause validation failure."""
        data = _complete_verdict()
        data["extra"] = {"rmse": 0.42}
        data["unexpected_key"] = "fine"
        result = validate_verdict(data)
        assert result.valid

    def test_summary_is_human_readable(self):
        """summary() should produce a readable string."""
        data = _complete_verdict()
        del data["worst_model"]
        result = validate_verdict(data)
        s = result.summary()
        assert "invalid" in s
        assert "worst_model" in s

    def test_valid_summary(self):
        """Valid verdict summary should say 'valid'."""
        result = validate_verdict(_complete_verdict())
        assert result.summary() == "verdict valid"


# ── load_validated tests ─────────────────────────────────────────────────

class TestLoadValidated:

    def test_file_not_found(self, tmp_path):
        """Should return None + error when file doesn't exist."""
        verdict, result = AgentVerdict.load_validated(tmp_path)
        assert verdict is None
        assert not result.valid
        assert "file not found" in result.errors[0]

    def test_invalid_json(self, tmp_path):
        """Should return None + error for malformed JSON."""
        (tmp_path / VERDICT_FILENAME).write_text("not json {{{")
        verdict, result = AgentVerdict.load_validated(tmp_path)
        assert verdict is None
        assert not result.valid
        assert any("invalid JSON" in e for e in result.errors)

    def test_json_array_not_object(self, tmp_path):
        """JSON array instead of object should fail."""
        (tmp_path / VERDICT_FILENAME).write_text('[1, 2, 3]')
        verdict, result = AgentVerdict.load_validated(tmp_path)
        assert verdict is None
        assert not result.valid
        assert any("expected JSON object" in e for e in result.errors)

    def test_schema_invalid(self, tmp_path):
        """Valid JSON but missing fields should fail."""
        (tmp_path / VERDICT_FILENAME).write_text('{"best_model": "X"}')
        verdict, result = AgentVerdict.load_validated(tmp_path)
        assert verdict is None
        assert not result.valid

    def test_valid_verdict(self, tmp_path):
        """Valid verdict should return the AgentVerdict object."""
        data = _complete_verdict()
        (tmp_path / VERDICT_FILENAME).write_text(json.dumps(data))
        verdict, result = AgentVerdict.load_validated(tmp_path)
        assert verdict is not None
        assert result.valid
        assert verdict.best_model == "CalibratedModel"


# ── Grader integration: verdict_valid criterion ──────────────────────────

class TestGraderValidation:

    def test_valid_verdict_shows_criterion(self, tmp_path):
        """Valid verdict should produce a passing verdict_valid criterion."""
        _make_report(tmp_path)
        (tmp_path / VERDICT_FILENAME).write_text(json.dumps(_complete_verdict()))

        grade = grade_drug_efficacy(tmp_path)
        vc = next(c for c in grade.criteria if c.name == "verdict_valid")
        assert vc.passed
        assert "verdict valid" in vc.detail

    def test_invalid_verdict_fails_criterion(self, tmp_path):
        """Invalid verdict should fail verdict_valid and domain criteria."""
        _make_report(tmp_path)
        (tmp_path / VERDICT_FILENAME).write_text('{"best_model": "X"}')

        grade = grade_drug_efficacy(tmp_path)
        vc = next(c for c in grade.criteria if c.name == "verdict_valid")
        assert not vc.passed
        assert "missing" in vc.detail

        # Domain criteria should also fail with diagnostic
        best = next(c for c in grade.criteria if c.name == "identifies_best")
        assert not best.passed
        assert "verdict invalid" in best.detail

    def test_malformed_json_fails(self, tmp_path):
        """Non-JSON verdict file should fail verdict_valid."""
        _make_report(tmp_path)
        (tmp_path / VERDICT_FILENAME).write_text("This is prose, not JSON!")

        grade = grade_drug_efficacy(tmp_path)
        vc = next(c for c in grade.criteria if c.name == "verdict_valid")
        assert not vc.passed
        assert "invalid JSON" in vc.detail

    def test_prose_fallback_no_verdict_criterion(self, tmp_path):
        """Prose fallback should NOT produce a verdict_valid criterion."""
        _make_report(tmp_path)
        (tmp_path / "agent_summary.txt").write_text(
            "The CalibratedModel achieves the best performance with lowest RMSE. "
            "The LinearModel fails to capture the sigmoidal dose-response."
        )

        grade = grade_drug_efficacy(tmp_path)
        names = [c.name for c in grade.criteria]
        assert "verdict_valid" not in names
        assert grade.all_passed

    def test_weather_valid_verdict(self, tmp_path):
        """Weather grader should also validate verdict schema."""
        _make_report(tmp_path)
        verdict = {
            "best_model": "NoisyRegressionModel",
            "best_reason": "lowest RMSE",
            "worst_model": "PersistenceModel",
            "worst_reason": "bad",
            "reference_model": "ClimatologyModel",
            "summary": "Regression wins.",
        }
        (tmp_path / VERDICT_FILENAME).write_text(json.dumps(verdict))

        grade = grade_weather(tmp_path)
        vc = next(c for c in grade.criteria if c.name == "verdict_valid")
        assert vc.passed
        assert grade.all_passed

    def test_weather_invalid_verdict(self, tmp_path):
        """Weather grader should catch invalid verdict schema."""
        _make_report(tmp_path)
        (tmp_path / VERDICT_FILENAME).write_text('{"best_model": 42}')

        grade = grade_weather(tmp_path)
        vc = next(c for c in grade.criteria if c.name == "verdict_valid")
        assert not vc.passed

    def test_no_report_skips_all(self, tmp_path):
        """Missing report should skip all criteria including verdict_valid."""
        grade = grade_drug_efficacy(tmp_path)
        assert grade.pass_count == 0
        names = [c.name for c in grade.criteria]
        assert "verdict_valid" in names
        assert all(not c.passed for c in grade.criteria)

    def test_correct_criterion_count_json(self, tmp_path):
        """With JSON verdict: 5 criteria (report, sections, verdict_valid, best, worst)."""
        _make_report(tmp_path)
        (tmp_path / VERDICT_FILENAME).write_text(json.dumps(_complete_verdict()))
        grade = grade_drug_efficacy(tmp_path)
        assert grade.total_count == 5

    def test_correct_criterion_count_prose(self, tmp_path):
        """With prose fallback: 4 criteria (report, sections, best, worst)."""
        _make_report(tmp_path)
        (tmp_path / "agent_summary.txt").write_text(
            "The CalibratedModel is the best with lowest RMSE. "
            "LinearModel fails on the sigmoid."
        )
        grade = grade_drug_efficacy(tmp_path)
        assert grade.total_count == 4
