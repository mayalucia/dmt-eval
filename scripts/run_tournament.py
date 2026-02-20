"""Run a tournament across multiple models and briefs.

Usage:
    uv run --extra llm --extra dev python scripts/run_tournament.py
"""

import sys
from pathlib import Path

from dmt.agent.brief import DRUG_EFFICACY_BRIEF, WEATHER_BRIEF
from dmt.agent.tournament import run_tournament


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./tournament_output")

    # Models to test — adjust based on available API keys
    models = [
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
    ]

    briefs = [DRUG_EFFICACY_BRIEF, WEATHER_BRIEF]

    print("=" * 60)
    print("  DMT-Eval Tournament")
    print(f"  Models: {len(models)} | Briefs: {len(briefs)}")
    print("=" * 60)
    print()

    result = run_tournament(
        models=models,
        briefs=briefs,
        output_root=output_dir,
    )

    print()
    print("=" * 60)
    print("  LEADERBOARD")
    print("=" * 60)
    print()
    print(result.leaderboard())
    print()

    # Print detailed grade reports
    for entry in result.entries:
        if entry.grade_report:
            print(f"\n--- {entry.model} x {entry.brief_name} ---")
            print(entry.grade_report.summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
