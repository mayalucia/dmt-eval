"""Live LLM agent tests.

These tests call the Anthropic API and cost real money.
Run with: uv run --extra llm --extra dev pytest test/test_llm_agent.py -v -m llm

Skip these in CI unless ANTHROPIC_API_KEY is set and --llm is passed.
"""

import os
from pathlib import Path

import pytest

# Skip entire module if no API key
pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    ),
]


@pytest.fixture
def agent_output(tmp_path):
    """Run the live agent once and return (llm_response, agent_result, output_dir)."""
    from dmt.agent.brief import DRUG_EFFICACY_BRIEF
    from dmt.agent.llm_runner import run_llm_agent

    output_dir = tmp_path / "llm_agent"
    llm_response, agent_result = run_llm_agent(
        brief=DRUG_EFFICACY_BRIEF,
        output_dir=output_dir,
    )
    return llm_response, agent_result, output_dir


def test_llm_produces_valid_code(agent_output):
    """Claude should produce syntactically valid Python."""
    llm_response, _, _ = agent_output
    code = llm_response.extracted_code

    # Should be valid Python
    compile(code, "<agent>", "exec")


def test_llm_agent_executes_successfully(agent_output):
    """The generated script should run without errors."""
    _, agent_result, _ = agent_output

    assert agent_result.success, (
        f"Agent script failed (exit code {agent_result.return_code}).\n"
        f"stderr: {agent_result.stderr}\n"
        f"stdout: {agent_result.stdout}"
    )


def test_llm_agent_produces_report(agent_output):
    """The agent should produce both report.md and agent_summary.txt."""
    _, agent_result, _ = agent_output

    if not agent_result.success:
        pytest.skip(f"Agent failed: {agent_result.stderr[:200]}")

    assert agent_result.report_exists, "No report.md produced"
    assert agent_result.summary_exists, "No agent_summary.txt produced"


def test_llm_agent_passes_grading(agent_output):
    """The live agent should pass all grading criteria."""
    _, agent_result, output_dir = agent_output

    if not agent_result.success:
        pytest.skip(f"Agent failed: {agent_result.stderr[:200]}")

    from dmt.agent.grader import grade_drug_efficacy
    grade = grade_drug_efficacy(output_dir)

    print(f"\n{grade.summary()}")

    for criterion in grade.criteria:
        assert criterion.passed, (
            f"Criterion '{criterion.name}' FAILED: {criterion.detail}"
        )
