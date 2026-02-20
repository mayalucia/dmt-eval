"""Grade an agent's output against success criteria."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CriterionResult:
    """Result of evaluating a single success criterion."""
    name: str
    passed: bool
    detail: str


@dataclass
class GradeReport:
    """Full grading report for an agent run."""
    agent_name: str
    criteria: list[CriterionResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.criteria)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.criteria if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.criteria)

    @property
    def score(self) -> float:
        if not self.criteria:
            return 0.0
        return self.pass_count / self.total_count

    def summary(self) -> str:
        lines = [
            f"Agent: {self.agent_name}",
            f"Score: {self.pass_count}/{self.total_count} "
            f"({self.score:.0%})",
            "",
        ]
        for c in self.criteria:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        return "\n".join(lines)


# ── Semantic keyword matching ──────────────────────────────────────────────

_POSITIVE_WORDS = frozenset({
    "best", "lowest", "superior", "outperform", "outperforms",
    "highest accuracy", "top", "winner", "strongest",
})

_NEGATIVE_WORDS = frozenset({
    "worst", "fails", "failure", "poor", "poorest",
    "highest rmse", "cannot capture", "inadequate",
})


def _text_contains_positive(text: str, entity: str) -> bool:
    """Check if text positively identifies entity as the best."""
    text = text.lower()
    entity = entity.lower()
    if entity not in text:
        return False
    return any(w in text for w in _POSITIVE_WORDS)


def _text_contains_negative(text: str, entity: str) -> bool:
    """Check if text negatively identifies entity's limitations."""
    text = text.lower()
    entity = entity.lower()
    if entity not in text:
        return False
    return any(w in text for w in _NEGATIVE_WORDS)


# ── Domain-specific graders ────────────────────────────────────────────────

def grade_drug_efficacy(output_dir: str | Path) -> GradeReport:
    """Grade an agent's drug efficacy validation output."""
    output_dir = Path(output_dir)
    report = GradeReport(agent_name="Drug Efficacy Validation")
    report_path = output_dir / "report.md"
    summary_path = output_dir / "agent_summary.txt"

    # ── Criterion 1: Report exists ────────────────────────────────────
    exists = report_path.exists()
    report.criteria.append(CriterionResult(
        name="report_exists",
        passed=exists,
        detail=str(report_path) if exists else "report.md not found",
    ))

    if not exists:
        for name in ["has_sections", "identifies_best", "identifies_worst"]:
            report.criteria.append(CriterionResult(
                name=name, passed=False, detail="skipped (no report)",
            ))
        return report

    report_text = report_path.read_text()

    # ── Criterion 2: Has required sections ────────────────────────────
    required = ["Abstract", "Methods", "Results", "Discussion", "Conclusion"]
    missing = [s for s in required if f"## {s}" not in report_text]
    report.criteria.append(CriterionResult(
        name="has_sections",
        passed=len(missing) == 0,
        detail="all present" if not missing else f"missing: {missing}",
    ))

    # ── Criterion 3: Identifies Calibrated as best ────────────────────
    summary_text = ""
    if summary_path.exists():
        summary_text = summary_path.read_text()

    calibrated_best = _text_contains_positive(summary_text, "calibrated")
    report.criteria.append(CriterionResult(
        name="identifies_best",
        passed=calibrated_best,
        detail=(
            "correctly identifies Calibrated" if calibrated_best
            else "did not identify Calibrated as best model"
        ),
    ))

    # ── Criterion 4: Notes Linear failure ─────────────────────────────
    linear_fails = _text_contains_negative(summary_text, "linear")
    # Also check for domain-specific reasoning
    summary_lower = summary_text.lower()
    if not linear_fails:
        linear_fails = (
            "linear" in summary_lower
            and ("sigmoid" in summary_lower or "hill" in summary_lower)
        )
    report.criteria.append(CriterionResult(
        name="identifies_worst",
        passed=linear_fails,
        detail=(
            "correctly notes Linear failure" if linear_fails
            else "did not explain Linear model's failure mode"
        ),
    ))

    return report


def grade_weather(output_dir: str | Path) -> GradeReport:
    """Grade an agent's weather prediction validation output."""
    output_dir = Path(output_dir)
    report = GradeReport(agent_name="Weather Prediction Validation")
    report_path = output_dir / "report.md"
    summary_path = output_dir / "agent_summary.txt"

    # ── Criterion 1: Report exists ────────────────────────────────────
    exists = report_path.exists()
    report.criteria.append(CriterionResult(
        name="report_exists",
        passed=exists,
        detail=str(report_path) if exists else "report.md not found",
    ))

    if not exists:
        for name in ["has_sections", "identifies_best", "identifies_reference"]:
            report.criteria.append(CriterionResult(
                name=name, passed=False, detail="skipped (no report)",
            ))
        return report

    report_text = report_path.read_text()

    # ── Criterion 2: Has required sections ────────────────────────────
    required = ["Abstract", "Methods", "Results", "Discussion", "Conclusion"]
    missing = [s for s in required if f"## {s}" not in report_text]
    report.criteria.append(CriterionResult(
        name="has_sections",
        passed=len(missing) == 0,
        detail="all present" if not missing else f"missing: {missing}",
    ))

    # ── Criterion 3: Identifies NoisyRegression as best ───────────────
    summary_text = ""
    if summary_path.exists():
        summary_text = summary_path.read_text()

    regression_best = _text_contains_positive(summary_text, "regression")
    # Also accept "NoisyRegression" as a variant
    if not regression_best:
        regression_best = _text_contains_positive(summary_text, "noisyregression")
    report.criteria.append(CriterionResult(
        name="identifies_best",
        passed=regression_best,
        detail=(
            "correctly identifies NoisyRegression" if regression_best
            else "did not identify NoisyRegression as best model"
        ),
    ))

    # ── Criterion 4: Mentions Climatology as baseline ─────────────────
    summary_lower = summary_text.lower()
    climatology_ref = (
        "climatology" in summary_lower
        and ("baseline" in summary_lower or "reference" in summary_lower
             or "benchmark" in summary_lower or "relative" in summary_lower
             or "compared" in summary_lower or "skill" in summary_lower)
    )
    report.criteria.append(CriterionResult(
        name="identifies_reference",
        passed=climatology_ref,
        detail=(
            "correctly references Climatology baseline" if climatology_ref
            else "did not mention Climatology as reference/baseline"
        ),
    ))

    return report


# ── Grader dispatch ────────────────────────────────────────────────────────

GRADERS = {
    "Drug Efficacy Validation": grade_drug_efficacy,
    "Weather Prediction Validation": grade_weather,
}


def grade_output(brief_name: str, output_dir: str | Path) -> GradeReport:
    """Grade agent output using the appropriate domain grader."""
    grader = GRADERS.get(brief_name)
    if grader is None:
        raise ValueError(
            f"No grader for brief '{brief_name}'. "
            f"Available: {list(GRADERS.keys())}"
        )
    return grader(output_dir)
