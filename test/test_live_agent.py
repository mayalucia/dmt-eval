"""End-to-end test: run the simulated agent, grade its output.

This is DMT testing itself: the framework is the model, the agent's
success/failure is the data, and the grading criteria are the test.
"""

from pathlib import Path

from dmt.agent.runner import run_agent
from dmt.agent.grader import grade_drug_efficacy


# Path to the simulated agent script (relative to repo root)
AGENT_SCRIPT = Path(__file__).parent.parent / "scripts" / "simulated_agent.py"


def test_simulated_agent_produces_report(tmp_path):
    """The simulated agent should produce a valid report."""
    result = run_agent(AGENT_SCRIPT, output_dir=tmp_path / "agent_output")

    assert result.success, (
        f"Agent script failed with return code {result.return_code}.\n"
        f"stderr: {result.stderr}"
    )
    assert result.report_exists, "Agent did not produce report.md"
    assert result.summary_exists, "Agent did not produce agent_summary.txt"


def test_simulated_agent_passes_all_criteria(tmp_path):
    """The simulated agent's output should pass all grading criteria."""
    output_dir = tmp_path / "agent_output"

    # Run the agent
    result = run_agent(AGENT_SCRIPT, output_dir=output_dir)
    assert result.success, f"Agent failed: {result.stderr}"

    # Grade the output
    grade = grade_drug_efficacy(output_dir)

    # Print the grade report for visibility
    print("\n" + grade.summary())

    # Assert all criteria pass
    for criterion in grade.criteria:
        assert criterion.passed, (
            f"Criterion '{criterion.name}' FAILED: {criterion.detail}"
        )

    assert grade.all_passed, (
        f"Agent scored {grade.pass_count}/{grade.total_count}"
    )


def test_grade_report_structure(tmp_path):
    """The grade report should have the expected structure."""
    output_dir = tmp_path / "agent_output"
    result = run_agent(AGENT_SCRIPT, output_dir=output_dir)
    assert result.success

    grade = grade_drug_efficacy(output_dir)

    assert grade.agent_name == "Drug Efficacy Validation"
    assert grade.total_count == 4
    assert grade.score == 1.0
    assert "PASS" in grade.summary()
    assert "FAIL" not in grade.summary()


def test_agent_brief_is_self_contained():
    """The brief alone should contain all information needed."""
    from dmt.agent.brief import DRUG_EFFICACY_BRIEF

    prompt = DRUG_EFFICACY_BRIEF.to_prompt()

    # The brief mentions all necessary imports
    assert "dmt.evaluate" in prompt
    assert "dmt.scenario.drug_efficacy" in prompt
    assert "evaluate" in prompt
    assert "DRUG_EFFICACY" in prompt

    # The brief has all four steps
    assert "generate_observations" in prompt
    assert "LinearModel" in prompt
    assert "evaluate(models=" in prompt
    assert "summary" in prompt.lower()

    # The brief has success criteria
    assert "report.md" in prompt or "report" in prompt.lower()
    assert "Calibrated" in prompt
