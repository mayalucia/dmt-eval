"""Grade an agent's output against success criteria.

Lesson 06: structured JSON verdict is the primary grading path.
Lesson 07: schema validation before grading.
Falls back to prose keyword matching if agent_verdict.json is absent.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from dmt.agent.verdict import validate_verdict, ValidationResult, VERDICT_FILENAME


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


# ── Verdict loading + validation ─────────────────────────────────────────

def _load_and_validate_verdict(output_dir: Path) -> tuple[dict | None, ValidationResult | None]:
    """Load and validate agent_verdict.json.

    Returns (parsed_dict, validation_result).
    - If file doesn't exist: (None, None) — triggers prose fallback.
    - If file exists but invalid JSON: (None, ValidationResult with errors).
    - If file exists, valid JSON, but bad schema: (dict, ValidationResult with errors).
    - If file exists and schema-valid: (dict, ValidationResult ok).
    """
    verdict_path = output_dir / VERDICT_FILENAME
    if not verdict_path.exists():
        return None, None

    try:
        data = json.loads(verdict_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None, ValidationResult(valid=False, errors=[f"invalid JSON: {e}"])

    if not isinstance(data, dict):
        return None, ValidationResult(
            valid=False,
            errors=[f"expected JSON object, got {type(data).__name__}"]
        )

    result = validate_verdict(data)
    return data, result


# ── Prose fallback (kept for backward compatibility) ─────────────────────

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


# ── Report section checker (shared) ─────────────────────────────────────

def _check_report_sections(report_text: str) -> CriterionResult:
    """Check that the report has all required sections."""
    required = ["Abstract", "Methods", "Results", "Discussion", "Conclusion"]
    missing = [s for s in required if f"## {s}" not in report_text]
    return CriterionResult(
        name="has_sections",
        passed=len(missing) == 0,
        detail="all present" if not missing else f"missing: {missing}",
    )


# ── Domain-specific graders ──────────────────────────────────────────────

def grade_drug_efficacy(output_dir: str | Path) -> GradeReport:
    """Grade an agent's drug efficacy validation output."""
    output_dir = Path(output_dir)
    report = GradeReport(agent_name="Drug Efficacy Validation")
    report_path = output_dir / "report.md"

    # ── Criterion 1: Report exists ────────────────────────────────────
    exists = report_path.exists()
    report.criteria.append(CriterionResult(
        name="report_exists",
        passed=exists,
        detail=str(report_path) if exists else "report.md not found",
    ))

    if not exists:
        for name in ["has_sections", "verdict_valid", "identifies_best", "identifies_worst"]:
            report.criteria.append(CriterionResult(
                name=name, passed=False, detail="skipped (no report)",
            ))
        return report

    report_text = report_path.read_text()

    # ── Criterion 2: Has required sections ────────────────────────────
    report.criteria.append(_check_report_sections(report_text))

    # ── Load and validate verdict ─────────────────────────────────────
    verdict, validation = _load_and_validate_verdict(output_dir)

    if validation is not None:
        # File existed — report validation result
        report.criteria.append(CriterionResult(
            name="verdict_valid",
            passed=validation.valid,
            detail=validation.summary(),
        ))

        if validation.valid:
            # ── Criterion 4: best_model == "CalibratedModel" ─────────
            best = verdict.get("best_model", "")
            calibrated_best = "calibrat" in best.lower()
            report.criteria.append(CriterionResult(
                name="identifies_best",
                passed=calibrated_best,
                detail=(
                    f"verdict.best_model={best!r}" if calibrated_best
                    else f"verdict.best_model={best!r} (expected Calibrated)"
                ),
            ))

            # ── Criterion 5: worst_model == "LinearModel" ────────────
            worst = verdict.get("worst_model", "")
            linear_worst = "linear" in worst.lower()
            report.criteria.append(CriterionResult(
                name="identifies_worst",
                passed=linear_worst,
                detail=(
                    f"verdict.worst_model={worst!r}" if linear_worst
                    else f"verdict.worst_model={worst!r} (expected Linear)"
                ),
            ))
        else:
            # Schema invalid — domain criteria auto-fail with diagnostic
            report.criteria.append(CriterionResult(
                name="identifies_best",
                passed=False,
                detail=f"skipped (verdict invalid: {validation.summary()})",
            ))
            report.criteria.append(CriterionResult(
                name="identifies_worst",
                passed=False,
                detail=f"skipped (verdict invalid: {validation.summary()})",
            ))
    else:
        # ── Prose fallback (no verdict file) ──────────────────────────
        summary_path = output_dir / "agent_summary.txt"
        summary_text = summary_path.read_text() if summary_path.exists() else ""

        calibrated_best = _text_contains_positive(summary_text, "calibrated")
        report.criteria.append(CriterionResult(
            name="identifies_best",
            passed=calibrated_best,
            detail=(
                "correctly identifies Calibrated (prose fallback)"
                if calibrated_best
                else "did not identify Calibrated as best model (prose fallback)"
            ),
        ))

        linear_fails = _text_contains_negative(summary_text, "linear")
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
                "correctly notes Linear failure (prose fallback)"
                if linear_fails
                else "did not identify Linear as worst (prose fallback)"
            ),
        ))

    return report


def grade_weather(output_dir: str | Path) -> GradeReport:
    """Grade an agent's weather prediction validation output."""
    output_dir = Path(output_dir)
    report = GradeReport(agent_name="Weather Prediction Validation")
    report_path = output_dir / "report.md"

    # ── Criterion 1: Report exists ────────────────────────────────────
    exists = report_path.exists()
    report.criteria.append(CriterionResult(
        name="report_exists",
        passed=exists,
        detail=str(report_path) if exists else "report.md not found",
    ))

    if not exists:
        for name in ["has_sections", "verdict_valid", "identifies_best", "identifies_reference"]:
            report.criteria.append(CriterionResult(
                name=name, passed=False, detail="skipped (no report)",
            ))
        return report

    report_text = report_path.read_text()

    # ── Criterion 2: Has required sections ────────────────────────────
    report.criteria.append(_check_report_sections(report_text))

    # ── Load and validate verdict ─────────────────────────────────────
    verdict, validation = _load_and_validate_verdict(output_dir)

    if validation is not None:
        # File existed — report validation result
        report.criteria.append(CriterionResult(
            name="verdict_valid",
            passed=validation.valid,
            detail=validation.summary(),
        ))

        if validation.valid:
            # ── Criterion 4: best_model contains "Regression" ─────────
            best = verdict.get("best_model", "")
            regression_best = "regression" in best.lower()
            report.criteria.append(CriterionResult(
                name="identifies_best",
                passed=regression_best,
                detail=(
                    f"verdict.best_model={best!r}" if regression_best
                    else f"verdict.best_model={best!r} (expected NoisyRegression)"
                ),
            ))

            # ── Criterion 5: reference_model contains "Climatology" ───
            ref = verdict.get("reference_model", "")
            climatology_ref = "climatology" in ref.lower()
            report.criteria.append(CriterionResult(
                name="identifies_reference",
                passed=climatology_ref,
                detail=(
                    f"verdict.reference_model={ref!r}" if climatology_ref
                    else f"verdict.reference_model={ref!r} (expected Climatology)"
                ),
            ))
        else:
            # Schema invalid — domain criteria auto-fail with diagnostic
            report.criteria.append(CriterionResult(
                name="identifies_best",
                passed=False,
                detail=f"skipped (verdict invalid: {validation.summary()})",
            ))
            report.criteria.append(CriterionResult(
                name="identifies_reference",
                passed=False,
                detail=f"skipped (verdict invalid: {validation.summary()})",
            ))
    else:
        # ── Prose fallback (no verdict file) ──────────────────────────
        summary_path = output_dir / "agent_summary.txt"
        summary_text = summary_path.read_text() if summary_path.exists() else ""

        regression_best = _text_contains_positive(summary_text, "regression")
        if not regression_best:
            regression_best = _text_contains_positive(summary_text, "noisyregression")
        report.criteria.append(CriterionResult(
            name="identifies_best",
            passed=regression_best,
            detail=(
                "correctly identifies NoisyRegression (prose fallback)"
                if regression_best
                else "did not identify NoisyRegression as best (prose fallback)"
            ),
        ))

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
                "correctly references Climatology baseline (prose fallback)"
                if climatology_ref
                else "did not mention Climatology as reference (prose fallback)"
            ),
        ))

    return report


# ── Grader dispatch ──────────────────────────────────────────────────────

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
