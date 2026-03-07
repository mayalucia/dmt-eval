"""Smoke tests for the DMT CLI."""

from pathlib import Path

from typer.testing import CliRunner

from dmt.cli import app

runner = CliRunner()


# ── version ───────────────────────────────────────────────────────────────────

def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "dmt-eval" in result.output


# ── list ──────────────────────────────────────────────────────────────────────

def test_list_scenarios():
    result = runner.invoke(app, ["list", "scenarios"])
    assert result.exit_code == 0
    assert "weather" in result.output
    assert "drug_efficacy" in result.output


def test_list_metrics():
    result = runner.invoke(app, ["list", "metrics"])
    assert result.exit_code == 0
    assert "rmse" in result.output
    assert "bias" in result.output


def test_list_unknown():
    result = runner.invoke(app, ["list", "bogus"])
    assert result.exit_code != 0


# ── eval ──────────────────────────────────────────────────────────────────────

def test_eval_weather(tmp_path):
    out = tmp_path / "weather_report"
    result = runner.invoke(app, ["eval", "--scenario", "weather", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "report.md").exists()
    report_text = (out / "report.md").read_text()
    assert "## Abstract" in report_text
    assert "## Results" in report_text
    assert "Persistence" in report_text


def test_eval_drug_efficacy(tmp_path):
    out = tmp_path / "drug_report"
    result = runner.invoke(
        app, ["eval", "--scenario", "drug_efficacy", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "report.md").exists()
    report_text = (out / "report.md").read_text()
    assert "## Abstract" in report_text
    assert "Linear" in report_text


def test_eval_unknown_scenario():
    result = runner.invoke(app, ["eval", "--scenario", "bogus"])
    assert result.exit_code != 0


def test_eval_custom_title(tmp_path):
    out = tmp_path / "titled_report"
    result = runner.invoke(
        app, [
            "eval", "--scenario", "weather",
            "--output", str(out),
            "--title", "My Custom Title",
        ],
    )
    assert result.exit_code == 0, result.output
    report_text = (out / "report.md").read_text()
    assert "# My Custom Title" in report_text


# ── compare ───────────────────────────────────────────────────────────────────

def test_compare_weather(tmp_path):
    out = tmp_path / "comparison"
    result = runner.invoke(
        app, ["compare", "--scenario", "weather", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "report.md").exists()


def test_compare_with_reference(tmp_path):
    out = tmp_path / "comparison_ref"
    result = runner.invoke(
        app, [
            "compare", "--scenario", "weather",
            "--reference", "Climatology",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    report_text = (out / "report.md").read_text()
    assert "Climatology" in report_text


def test_compare_filter_models(tmp_path):
    out = tmp_path / "comparison_filtered"
    result = runner.invoke(
        app, [
            "compare", "--scenario", "weather",
            "--models", "Persistence,Climatology",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    report_text = (out / "report.md").read_text()
    assert "Persistence" in report_text
    assert "Climatology" in report_text


# ── report ────────────────────────────────────────────────────────────────────

def test_report_rerender(tmp_path):
    # First generate a report
    out = tmp_path / "report_dir"
    runner.invoke(app, ["eval", "--scenario", "weather", "--output", str(out)])
    assert (out / "report.md").exists()

    # Re-render it
    result = runner.invoke(app, ["report", str(out), "--format", "md"])
    assert result.exit_code == 0, result.output
    assert "## Abstract" in result.output


def test_report_missing_dir(tmp_path):
    result = runner.invoke(app, ["report", str(tmp_path / "nonexistent")])
    assert result.exit_code != 0
