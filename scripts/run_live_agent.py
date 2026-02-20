"""Run a live LLM agent on the drug efficacy brief and print the grade.

Usage:
    uv run --extra llm --extra dev python scripts/run_live_agent.py [output_dir]
"""

import sys
from pathlib import Path

from dmt.agent.brief import DRUG_EFFICACY_BRIEF
from dmt.agent.llm_runner import run_llm_agent
from dmt.agent.grader import grade_drug_efficacy


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./llm_agent_output")

    print(f"Sending brief to Claude...")
    print(f"Output directory: {output_dir}\n")

    try:
        llm_response, agent_result = run_llm_agent(
            brief=DRUG_EFFICACY_BRIEF,
            output_dir=output_dir,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Model: {llm_response.model}")
    print(f"Tokens: {llm_response.usage}")
    print(f"Agent exit code: {agent_result.return_code}")

    if agent_result.stderr:
        print(f"\nAgent stderr:\n{agent_result.stderr}")

    if agent_result.stdout:
        print(f"\nAgent stdout:\n{agent_result.stdout}")

    # Save the generated code for inspection
    workspace = output_dir / "_agent_workspace"
    print(f"\nGenerated script saved to: {workspace / 'agent_script.py'}")
    print(f"Raw LLM response saved to: {workspace / 'llm_raw_response.txt'}")

    # Grade
    print("\n" + "=" * 60)
    grade = grade_drug_efficacy(output_dir)
    print(grade.summary())
    print("=" * 60)

    # Show the agent's summary if it exists
    summary_path = output_dir / "agent_summary.txt"
    if summary_path.exists():
        print(f"\nAgent's scientific summary:\n{summary_path.read_text()}")

    return 0 if grade.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
