"""Tests for the MCP server tool functions.

These test the tool functions directly — no MCP transport or SDK
required. The functions are the same ones that get exposed via
the @mcp.tool() decorator.
"""

import pytest

from dmt.mcp_server import (
    dmt_evaluate,
    dmt_compare,
    dmt_list,
    dmt_report,
    KNOWN_SCENARIOS,
)


# ── dmt_evaluate ─────────────────────────────────────────────────────────────


class TestDmtEvaluate:
    """The evaluate tool produces a LabReport."""

    @pytest.mark.parametrize("scenario", KNOWN_SCENARIOS)
    def test_evaluate_each_scenario(self, scenario, tmp_path):
        """Every built-in scenario should produce a report."""
        report = dmt_evaluate(
            scenario=scenario,
            output_dir=str(tmp_path / f"{scenario}_report"),
        )
        assert isinstance(report, str)
        assert "## Abstract" in report
        assert "## Results" in report

    def test_evaluate_with_string_models(self, tmp_path):
        """String model specs should resolve."""
        report = dmt_evaluate(
            scenario="llm_qa",
            models=["echo", "template"],
            output_dir=str(tmp_path / "string_models"),
        )
        assert "Echo" in report
        assert "Template" in report

    def test_evaluate_with_title(self, tmp_path):
        report = dmt_evaluate(
            scenario="weather",
            output_dir=str(tmp_path / "titled"),
            title="Custom Title Report",
        )
        assert "Custom Title Report" in report

    def test_evaluate_unknown_scenario(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            dmt_evaluate(scenario="bogus")

    def test_evaluate_temp_dir(self):
        """When no output_dir, should use a temp directory."""
        report = dmt_evaluate(scenario="weather")
        assert isinstance(report, str)
        assert "## Abstract" in report


# ── dmt_compare ──────────────────────────────────────────────────────────────


class TestDmtCompare:
    """The compare tool produces a comparison report."""

    def test_compare_weather(self, tmp_path):
        report = dmt_compare(
            scenario="weather",
            output_dir=str(tmp_path / "compare"),
        )
        assert "## Results" in report
        assert "## Discussion" in report

    def test_compare_with_reference(self, tmp_path):
        report = dmt_compare(
            scenario="weather",
            reference="Climatology",
            output_dir=str(tmp_path / "compare_ref"),
        )
        assert "## Results" in report

    def test_compare_llm_qa(self, tmp_path):
        report = dmt_compare(
            scenario="llm_qa",
            output_dir=str(tmp_path / "compare_llm"),
        )
        assert "exact_match" in report.lower() or "Exact" in report


# ── dmt_list ─────────────────────────────────────────────────────────────────


class TestDmtList:
    """The list tool returns formatted info."""

    def test_list_scenarios(self):
        result = dmt_list("scenarios")
        assert "weather" in result
        assert "drug_efficacy" in result
        assert "equity_forecast" in result
        assert "llm_qa" in result

    def test_list_metrics(self):
        result = dmt_list("metrics")
        assert "rmse" in result
        assert "exact_match" in result

    def test_list_models(self):
        result = dmt_list("models")
        assert "echo" in result
        assert "anthropic" in result

    def test_list_unknown(self):
        result = dmt_list("bogus")
        assert "Unknown" in result


# ── dmt_report ───────────────────────────────────────────────────────────────


class TestDmtReport:
    """The report tool reads existing reports."""

    def test_read_existing_report(self, tmp_path):
        # First generate a report
        dmt_evaluate(
            scenario="weather",
            output_dir=str(tmp_path / "report_dir"),
        )
        # Then read it back
        result = dmt_report(str(tmp_path / "report_dir"))
        assert "## Abstract" in result

    def test_read_missing_report(self, tmp_path):
        result = dmt_report(str(tmp_path / "nonexistent"))
        assert "No report.md found" in result
