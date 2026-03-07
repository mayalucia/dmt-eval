"""DMT command-line interface.

dmt eval      — evaluate models against observations
dmt compare   — compare multiple models (tournament)
dmt report    — re-render an existing report
dmt list      — list available scenarios and metrics
"""

from pathlib import Path
from typing import Optional

import typer

from dmt import __version__

app = typer.Typer(
    name="dmt",
    help="DMT — Data, Models, Tests. Universal validation framework.",
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"dmt-eval {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=version_callback,
        is_eager=True, help="Show version and exit.",
    ),
):
    """DMT — Data, Models, Tests."""


# ── Scenario registry ────────────────────────────────────────────────────────

def _get_scenario(name: str):
    """Resolve a scenario name to (Scenario, observations, models)."""
    from dmt.evaluate import WEATHER, DRUG_EFFICACY

    if name == "weather":
        from dmt.scenario.weather import (
            generate_observations,
            PersistenceModel,
            ClimatologyModel,
            NoisyRegressionModel,
        )
        obs = generate_observations()
        models = [PersistenceModel(), ClimatologyModel(), NoisyRegressionModel()]
        return WEATHER, obs, models

    elif name == "drug_efficacy":
        from dmt.scenario.drug_efficacy import (
            generate_observations,
            LinearModel,
            SigmoidModel,
            CalibratedModel,
        )
        obs = generate_observations()
        models = [LinearModel(), SigmoidModel(), CalibratedModel()]
        return DRUG_EFFICACY, obs, models

    elif name == "equity_forecast":
        from dmt.evaluate import EQUITY_FORECAST
        from dmt.scenario.equity import (
            generate_returns,
            MeanModel,
            MomentumModel,
            VolatilityModel,
        )
        obs = generate_returns()
        models = [MeanModel(), MomentumModel(), VolatilityModel()]
        return EQUITY_FORECAST, obs, models

    elif name == "llm_qa":
        from dmt.scenario.llm_qa import LLM_QA, generate_dataset
        from dmt.models.baselines import EchoModel, RandomModel, TemplateModel
        obs = generate_dataset()
        models = [EchoModel(), RandomModel(), TemplateModel()]
        return LLM_QA, obs, models

    else:
        typer.echo(f"Unknown scenario: {name}", err=True)
        typer.echo(
            f"Available scenarios: {', '.join(_KNOWN_SCENARIOS)}", err=True,
        )
        raise typer.Exit(code=1)


_KNOWN_SCENARIOS = ["weather", "drug_efficacy", "equity_forecast", "llm_qa"]


# ── eval ──────────────────────────────────────────────────────────────────────

@app.command("eval")
def eval_cmd(
    scenario: str = typer.Option(
        ..., "--scenario", "-s", help="Scenario name (weather, drug_efficacy).",
    ),
    output: Path = typer.Option(
        Path("./dmt_report"), "--output", "-o", help="Output directory for the report.",
    ),
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Report title.",
    ),
):
    """Evaluate models against observations and produce a LabReport."""
    from dmt.evaluate import evaluate

    scenario_obj, obs, models = _get_scenario(scenario)
    report_title = title or f"{scenario.replace('_', ' ').title()} Model Evaluation"

    typer.echo(f"Evaluating {len(models)} models on '{scenario}' scenario...")
    report_path = evaluate(
        models=models,
        observations=obs,
        scenario=scenario_obj,
        output_dir=output,
        title=report_title,
    )
    typer.echo(f"Report written to {report_path}")


# ── compare ───────────────────────────────────────────────────────────────────

@app.command()
def compare(
    scenario: str = typer.Option(
        ..., "--scenario", "-s", help="Scenario name.",
    ),
    models_filter: Optional[str] = typer.Option(
        None, "--models", "-m",
        help="Comma-separated model names to include (default: all).",
    ),
    reference: Optional[str] = typer.Option(
        None, "--reference", "-r",
        help="Reference model name for skill scores.",
    ),
    output: Path = typer.Option(
        Path("./dmt_comparison"), "--output", "-o",
        help="Output directory for the comparison report.",
    ),
    title: Optional[str] = typer.Option(
        None, "--title", "-t", help="Report title.",
    ),
):
    """Compare multiple models on a scenario (tournament mode)."""
    from dmt.evaluate import evaluate

    scenario_obj, obs, all_models = _get_scenario(scenario)

    # Filter models if requested
    if models_filter:
        names = [n.strip() for n in models_filter.split(",")]
        filtered = [m for m in all_models if m.name in names]
        if not filtered:
            available = [m.name for m in all_models]
            typer.echo(
                f"No models matched {names}. Available: {available}", err=True,
            )
            raise typer.Exit(code=1)
        all_models = filtered

    # Find reference model
    ref_model = None
    if reference:
        for m in all_models:
            if m.name == reference:
                ref_model = m
                break
        if ref_model is None:
            available = [m.name for m in all_models]
            typer.echo(
                f"Reference model '{reference}' not found. Available: {available}",
                err=True,
            )
            raise typer.Exit(code=1)

    report_title = title or f"{scenario.replace('_', ' ').title()} Model Comparison"

    typer.echo(f"Comparing {len(all_models)} models on '{scenario}'...")
    report_path = evaluate(
        models=all_models,
        observations=obs,
        scenario=scenario_obj,
        reference_model=ref_model,
        output_dir=output,
        title=report_title,
    )
    typer.echo(f"Comparison report written to {report_path}")


# ── report ────────────────────────────────────────────────────────────────────

@app.command()
def report(
    report_dir: Path = typer.Argument(
        ..., help="Path to an existing report directory.",
    ),
    format: str = typer.Option(
        "md", "--format", "-f", help="Output format (md).",
    ),
):
    """Re-render an existing report directory."""
    report_md = report_dir / "report.md"
    if not report_md.exists():
        typer.echo(f"No report.md found in {report_dir}", err=True)
        raise typer.Exit(code=1)

    if format == "md":
        typer.echo(report_md.read_text())
    else:
        typer.echo(f"Format '{format}' not yet supported. Available: md", err=True)
        raise typer.Exit(code=1)


# ── list ──────────────────────────────────────────────────────────────────────

@app.command("list")
def list_cmd(
    what: str = typer.Argument(
        ..., help="What to list: scenarios, metrics.",
    ),
):
    """List available scenarios, metrics, or models."""
    if what == "scenarios":
        typer.echo("Available scenarios:")
        for s in _KNOWN_SCENARIOS:
            typer.echo(f"  {s}")
    elif what == "metrics":
        typer.echo("Built-in metrics:")
        typer.echo("  rmse    — Root Mean Square Error")
        typer.echo("  bias    — Mean prediction bias")
        typer.echo("  skill   — Skill score relative to reference")
    else:
        typer.echo(f"Unknown category: {what}", err=True)
        typer.echo("Available: scenarios, metrics", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
