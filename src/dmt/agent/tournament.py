"""Run a tournament: multiple LLMs compete on the same brief.

The tournament runner:
1. Takes a list of (model_id, provider) pairs and a brief
2. Sends the brief to each model
3. Executes the generated code
4. Grades each output
5. Returns a leaderboard
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dmt.agent.brief import AgentBrief
from dmt.agent.grader import GradeReport, grade_output
from dmt.agent.llm_runner import LLMResponse, run_llm_agent


@dataclass
class TournamentEntry:
    """Result from one contestant in the tournament."""
    model: str
    brief_name: str
    score: float
    pass_count: int
    total_count: int
    code_valid: bool
    execution_success: bool
    elapsed_seconds: float
    tokens_used: dict = field(default_factory=dict)
    grade_report: GradeReport | None = None
    error: str | None = None


@dataclass
class TournamentResult:
    """Complete tournament results."""
    entries: list[TournamentEntry] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for e in self.entries:
            rows.append({
                "model": e.model,
                "brief": e.brief_name,
                "score": f"{e.pass_count}/{e.total_count}",
                "pct": f"{e.score:.0%}",
                "code_valid": e.code_valid,
                "executes": e.execution_success,
                "time_s": f"{e.elapsed_seconds:.1f}",
                "error": e.error or "",
            })
        return pd.DataFrame(rows)

    def leaderboard(self) -> str:
        df = self.to_dataframe()
        return df.to_markdown(index=False)


def run_tournament(
    models: list[str],
    briefs: list[AgentBrief],
    output_root: str | Path = "./tournament_output",
    timeout: int = 60,
) -> TournamentResult:
    """Run a tournament: each model attempts each brief.

    Parameters
    ----------
    models : list of model IDs
        e.g. ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"]
    briefs : list of AgentBrief
        The tasks to attempt.
    output_root : path
        Base directory for all outputs.
    timeout : int
        Max seconds per agent execution.

    Returns TournamentResult with all entries.
    """
    output_root = Path(output_root)
    result = TournamentResult()

    for brief in briefs:
        for model in models:
            # Create a unique output directory per model+brief
            safe_model = model.replace("/", "_").replace(":", "_")
            safe_brief = brief.name.lower().replace(" ", "_")
            output_dir = output_root / f"{safe_brief}_{safe_model}"

            start = time.time()

            try:
                llm_response, agent_result = run_llm_agent(
                    brief=brief,
                    output_dir=output_dir,
                    model=model,
                    timeout=timeout,
                )
                elapsed = time.time() - start

                # Check code validity
                code_valid = True
                try:
                    compile(llm_response.extracted_code, "<agent>", "exec")
                except SyntaxError:
                    code_valid = False

                # Grade
                grade = grade_output(brief.name, output_dir)

                entry = TournamentEntry(
                    model=model,
                    brief_name=brief.name,
                    score=grade.score,
                    pass_count=grade.pass_count,
                    total_count=grade.total_count,
                    code_valid=code_valid,
                    execution_success=agent_result.success,
                    elapsed_seconds=elapsed,
                    tokens_used=llm_response.usage,
                    grade_report=grade,
                )

            except Exception as e:
                elapsed = time.time() - start
                entry = TournamentEntry(
                    model=model,
                    brief_name=brief.name,
                    score=0.0,
                    pass_count=0,
                    total_count=4,
                    code_valid=False,
                    execution_success=False,
                    elapsed_seconds=elapsed,
                    error=str(e),
                )

            result.entries.append(entry)
            # Print progress
            mark = "PASS" if entry.score == 1.0 else f"{entry.score:.0%}"
            print(f"  [{mark}] {model} x {brief.name}")

    return result
