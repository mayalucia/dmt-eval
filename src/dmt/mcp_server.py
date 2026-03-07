"""DMT MCP Server — validation as a cognitive tool for agents.

Exposes DMT capabilities as MCP tools so that AI agents can invoke
model evaluation, comparison, and reporting during their reasoning.

Requires: pip install dmt-eval[mcp]

Usage:
    dmt mcp serve            # stdio transport (Claude Code)
    dmt mcp serve --sse 8080 # SSE transport (remote agents)
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def _get_scenario(name: str):
    """Resolve scenario name to (Scenario, observations, models).

    Duplicates the CLI's _get_scenario but returns raw objects
    rather than printing errors — raises ValueError on unknown.
    """
    from dmt.evaluate import WEATHER, DRUG_EFFICACY, EQUITY_FORECAST

    if name == "weather":
        from dmt.scenario.weather import (
            generate_observations, PersistenceModel,
            ClimatologyModel, NoisyRegressionModel,
        )
        obs = generate_observations()
        models = [PersistenceModel(), ClimatologyModel(), NoisyRegressionModel()]
        return WEATHER, obs, models

    elif name == "drug_efficacy":
        from dmt.scenario.drug_efficacy import (
            generate_observations, LinearModel, SigmoidModel, CalibratedModel,
        )
        obs = generate_observations()
        models = [LinearModel(), SigmoidModel(), CalibratedModel()]
        return DRUG_EFFICACY, obs, models

    elif name == "equity_forecast":
        from dmt.scenario.equity import (
            generate_returns, MeanModel, MomentumModel, VolatilityModel,
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

    raise ValueError(
        f"Unknown scenario: {name}. "
        f"Available: weather, drug_efficacy, equity_forecast, llm_qa"
    )


KNOWN_SCENARIOS = ["weather", "drug_efficacy", "equity_forecast", "llm_qa"]


# ── Tool functions (callable without MCP transport for testing) ──────────────


def dmt_evaluate(
    scenario: str,
    models: list[str] | None = None,
    output_dir: str | None = None,
    title: str | None = None,
) -> str:
    """Evaluate models against observations and return a LabReport.

    Parameters
    ----------
    scenario : str
        Scenario name: weather, drug_efficacy, equity_forecast, llm_qa.
    models : list of str, optional
        Model specs (e.g. ["echo", "template"]). If None, uses
        built-in models for the scenario.
    output_dir : str, optional
        Output directory. If None, uses a temporary directory.
    title : str, optional
        Report title.

    Returns the Markdown report as a string.
    """
    from dmt.evaluate import evaluate

    scenario_obj, obs, default_models = _get_scenario(scenario)

    if models:
        from dmt.models import resolve
        model_objs = [resolve(m) for m in models]
    else:
        model_objs = default_models

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="dmt_report_")
    out = Path(output_dir)

    report_title = title or f"{scenario.replace('_', ' ').title()} Evaluation"

    report_path = evaluate(
        models=model_objs,
        observations=obs,
        scenario=scenario_obj,
        output_dir=out,
        title=report_title,
    )

    return report_path.read_text()


def dmt_compare(
    scenario: str,
    models: list[str] | None = None,
    reference: str | None = None,
    output_dir: str | None = None,
    title: str | None = None,
) -> str:
    """Compare multiple models on a scenario and return the report.

    Parameters
    ----------
    scenario : str
        Scenario name.
    models : list of str, optional
        Model specs to compare. If None, uses all built-in models.
    reference : str, optional
        Reference model name for skill scores.
    output_dir : str, optional
        Output directory.
    title : str, optional
        Report title.

    Returns the Markdown report as a string.
    """
    from dmt.evaluate import evaluate

    scenario_obj, obs, default_models = _get_scenario(scenario)

    if models:
        from dmt.models import resolve
        model_objs = [resolve(m) for m in models]
    else:
        model_objs = default_models

    ref_model = None
    if reference:
        for m in model_objs:
            if m.name == reference:
                ref_model = m
                break

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="dmt_compare_")
    out = Path(output_dir)

    report_title = title or f"{scenario.replace('_', ' ').title()} Comparison"

    report_path = evaluate(
        models=model_objs,
        observations=obs,
        scenario=scenario_obj,
        reference_model=ref_model,
        output_dir=out,
        title=report_title,
    )

    return report_path.read_text()


def dmt_list(what: str = "scenarios") -> str:
    """List available scenarios, metrics, or models.

    Parameters
    ----------
    what : str
        What to list: "scenarios", "metrics", or "models".

    Returns a formatted string.
    """
    if what == "scenarios":
        lines = ["Available scenarios:"]
        for s in KNOWN_SCENARIOS:
            lines.append(f"  - {s}")
        return "\n".join(lines)

    elif what == "metrics":
        return (
            "Built-in metrics:\n"
            "  Numeric scenarios: rmse, bias, skill_score\n"
            "  String scenarios: exact_match, fuzzy_match\n"
            "  Finance: rmse, directional_accuracy, sharpe_ratio, "
            "max_drawdown, var_95\n"
            "  LLM: exact_match, fuzzy_match, latency"
        )

    elif what == "models":
        return (
            "Built-in model specs (offline):\n"
            "  echo     — returns the input (worst baseline)\n"
            "  random   — random answer from pool\n"
            "  template — rule-based pattern matching\n"
            "\n"
            "API model specs (require keys):\n"
            "  anthropic/<model-id>  — e.g. anthropic/claude-haiku-4-5-20251001\n"
            "  openai/<model-id>    — e.g. openai/gpt-4o"
        )

    return f"Unknown category: {what}. Available: scenarios, metrics, models"


def dmt_report(report_dir: str) -> str:
    """Read and return an existing report.

    Parameters
    ----------
    report_dir : str
        Path to a report directory containing report.md.

    Returns the Markdown content.
    """
    report_path = Path(report_dir) / "report.md"
    if not report_path.exists():
        return f"No report.md found in {report_dir}"
    return report_path.read_text()


# ── MCP Server ───────────────────────────────────────────────────────────────


def create_server():
    """Create the FastMCP server with all tools registered."""
    from fastmcp import FastMCP

    mcp = FastMCP(name="dmt-eval")

    @mcp.tool()
    def evaluate(
        scenario: str,
        models: list[str] | None = None,
        output_dir: str | None = None,
        title: str | None = None,
    ) -> str:
        """Evaluate models against observations and produce a LabReport.

        Args:
            scenario: Scenario name (weather, drug_efficacy, equity_forecast, llm_qa)
            models: Model specs (e.g. ["echo", "template"]). If omitted, uses built-in models.
            output_dir: Output directory for report files. Uses temp dir if omitted.
            title: Report title.
        """
        return dmt_evaluate(scenario, models, output_dir, title)

    @mcp.tool()
    def compare(
        scenario: str,
        models: list[str] | None = None,
        reference: str | None = None,
        output_dir: str | None = None,
        title: str | None = None,
    ) -> str:
        """Compare multiple models on a scenario and return the report.

        Args:
            scenario: Scenario name (weather, drug_efficacy, equity_forecast, llm_qa)
            models: Model specs to compare. Uses all built-in models if omitted.
            reference: Reference model name for skill scores.
            output_dir: Output directory for report files.
            title: Report title.
        """
        return dmt_compare(scenario, models, reference, output_dir, title)

    @mcp.tool()
    def list_available(what: str = "scenarios") -> str:
        """List available scenarios, metrics, or models.

        Args:
            what: What to list — "scenarios", "metrics", or "models".
        """
        return dmt_list(what)

    @mcp.tool()
    def read_report(report_dir: str) -> str:
        """Read and return an existing DMT report.

        Args:
            report_dir: Path to a report directory containing report.md.
        """
        return dmt_report(report_dir)

    return mcp


def run_stdio():
    """Run the MCP server with stdio transport."""
    server = create_server()
    server.run()


def run_sse(port: int = 8080):
    """Run the MCP server with SSE transport."""
    server = create_server()
    server.run(transport="sse", port=port)
