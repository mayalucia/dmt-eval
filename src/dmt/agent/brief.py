"""Structured agent briefs.

A brief encodes:
- What imports the agent has access to
- What steps it must perform
- What success looks like (grading criteria)
"""

from dataclasses import dataclass, field


VERDICT_SCHEMA_DESCRIPTION = """\
Write a JSON file called agent_verdict.json to the output directory with these fields:
{
  "best_model": "<name of best model>",
  "best_reason": "<one sentence why>",
  "worst_model": "<name of worst model>",
  "worst_reason": "<one sentence why>",
  "reference_model": "<name of baseline model>",
  "summary": "<2-3 sentence scientific summary>"
}"""


@dataclass
class AgentBrief:
    """A machine-readable agent task specification."""

    # Identity
    name: str
    description: str

    # What the agent can import
    imports: list[str] = field(default_factory=list)

    # Step-by-step instructions
    steps: list[str] = field(default_factory=list)

    # Constraints
    constraints: dict[str, str] = field(default_factory=dict)

    # Success criteria (key -> human description)
    success_criteria: dict[str, str] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Render the brief as a text prompt an LLM agent would receive."""
        lines = [
            f"**AGENT BRIEF: {self.name}**\n",
            self.description + "\n",
            "**Available imports**:",
        ]
        for imp in self.imports:
            lines.append(f"- `{imp}`")
        lines.append("\n**Your task**:")
        for i, step in enumerate(self.steps, 1):
            lines.append(f"{i}. {step}")
        if self.constraints:
            lines.append("\n**Constraints**:")
            for key, val in self.constraints.items():
                lines.append(f"- {key}: {val}")
        if self.success_criteria:
            lines.append("\n**Success criteria**:")
            for key, val in self.success_criteria.items():
                lines.append(f"- {key}: {val}")
        return "\n".join(lines)


# ── Pre-built briefs ──────────────────────────────────────────────────────

DRUG_EFFICACY_BRIEF = AgentBrief(
    name="Drug Efficacy Validation",
    description=(
        "You are a scientific computing agent. Your task is to evaluate "
        "three drug efficacy models using the DMT validation framework."
    ),
    imports=[
        "from dmt.evaluate import evaluate, DRUG_EFFICACY",
        "from dmt.scenario.drug_efficacy import generate_observations, "
        "LinearModel, SigmoidModel, CalibratedModel",
    ],
    steps=[
        "Generate dose-response observations: obs = generate_observations()",
        "Create three model instances: LinearModel(), SigmoidModel(), CalibratedModel()",
        ("Call: evaluate(models=[linear, sigmoid, calibrated], "
         "observations=obs, scenario=DRUG_EFFICACY, "
         "reference_model=linear, output_dir=OUTPUT_DIR, "
         "title='Drug Efficacy Model Comparison')"),
        ("Read the generated report.md and write agent_verdict.json "
         "to the output directory (see verdict_format constraint)"),
    ],
    constraints={
        "reference_model": "Use LinearModel as the reference (baseline) model",
        "output_dir": "Use the path passed as sys.argv[1] (or default to ./agent_drug_report/)",
        "verdict_format": VERDICT_SCHEMA_DESCRIPTION,
    },
    success_criteria={
        "report_exists": "The report file exists at the output directory",
        "has_sections": (
            "The report contains Abstract, Methods, Results, Discussion, Conclusion"
        ),
        "identifies_best": "Verdict correctly identifies the Calibrated model as best",
        "identifies_worst": (
            "Verdict correctly identifies the Linear model as worst"
        ),
    },
)


WEATHER_BRIEF = AgentBrief(
    name="Weather Prediction Validation",
    description=(
        "You are a scientific computing agent. Your task is to evaluate "
        "three weather prediction models using the DMT validation framework."
    ),
    imports=[
        "from dmt.evaluate import evaluate, WEATHER",
        "from dmt.scenario.weather import generate_observations, "
        "PersistenceModel, ClimatologyModel, NoisyRegressionModel",
    ],
    steps=[
        "Generate weather observations: obs = generate_observations(n_days=365, seed=42)",
        ("Create three model instances: PersistenceModel(), "
         "ClimatologyModel(), NoisyRegressionModel(alpha=0.7, noise_std=0.5)"),
        ("Call: evaluate(models=[persistence, climatology, regression], "
         "observations=obs, scenario=WEATHER, "
         "reference_model=climatology, output_dir=OUTPUT_DIR, "
         "title='Weather Prediction Model Comparison')"),
        ("Read the generated report.md and write agent_verdict.json "
         "to the output directory (see verdict_format constraint)"),
    ],
    constraints={
        "reference_model": "Use ClimatologyModel as the reference (baseline) model",
        "output_dir": "Use the path passed as sys.argv[1] (or default to ./agent_weather_report/)",
        "verdict_format": VERDICT_SCHEMA_DESCRIPTION,
    },
    success_criteria={
        "report_exists": "The report file exists at the output directory",
        "has_sections": (
            "The report contains Abstract, Methods, Results, Discussion, Conclusion"
        ),
        "identifies_best": (
            "Verdict identifies NoisyRegression as the best model"
        ),
        "identifies_worst": (
            "Verdict identifies the worst-performing model"
        ),
    },
)
