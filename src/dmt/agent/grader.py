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


def grade_drug_efficacy(output_dir: str | Path) -> GradeReport:
    """Grade an agent's drug efficacy validation output.

    Checks the four success criteria from the brief:
    1. Report file exists
    2. Report has required sections
    3. Summary identifies Calibrated as best
    4. Summary identifies Linear as worst / failing on sigmoidal data
    """
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
        # Can't grade further without the report
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
        summary_text = summary_path.read_text().lower()

    calibrated_best = (
        "calibrated" in summary_text
        and ("best" in summary_text or "lowest" in summary_text
             or "superior" in summary_text or "outperform" in summary_text
             or "highest accuracy" in summary_text)
    )
    report.criteria.append(CriterionResult(
        name="identifies_best",
        passed=calibrated_best,
        detail=(
            "correctly identifies Calibrated" if calibrated_best
            else "did not identify Calibrated as best model"
        ),
    ))

    # ── Criterion 4: Notes Linear failure on sigmoidal data ───────────
    linear_fails = (
        "linear" in summary_text
        and ("sigmoid" in summary_text or "hill" in summary_text
             or "fails" in summary_text or "worst" in summary_text)
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
