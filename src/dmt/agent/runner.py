"""Run an agent script in a subprocess and capture its output."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentResult:
    """Captured result from running an agent."""
    return_code: int
    stdout: str
    stderr: str
    output_dir: Path

    @property
    def success(self) -> bool:
        return self.return_code == 0

    @property
    def report_path(self) -> Path:
        return self.output_dir / "report.md"

    @property
    def verdict_path(self) -> Path:
        return self.output_dir / "agent_verdict.json"

    @property
    def summary_path(self) -> Path:
        """Legacy: agent_summary.txt (pre-Lesson 06)."""
        return self.output_dir / "agent_summary.txt"

    @property
    def report_exists(self) -> bool:
        return self.report_path.exists()

    @property
    def verdict_exists(self) -> bool:
        return self.verdict_path.exists()

    @property
    def summary_exists(self) -> bool:
        """Legacy: check for agent_summary.txt."""
        return self.summary_path.exists()


def run_agent(script_path: str | Path, output_dir: str | Path,
              python: str | None = None,
              timeout: int = 60) -> AgentResult:
    """Execute an agent script in a subprocess.

    Parameters
    ----------
    script_path : path
        The agent script to run.
    output_dir : path
        Passed as the first argument to the script.
    python : str, optional
        Python interpreter.  If None, uses the current interpreter.
    timeout : int
        Maximum seconds to allow the agent to run.

    Returns an AgentResult with captured stdout/stderr.
    """
    script_path = Path(script_path)
    output_dir = Path(output_dir)
    python = python or sys.executable

    result = subprocess.run(
        [python, str(script_path), str(output_dir)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=script_path.parent.parent,  # repo root
    )

    return AgentResult(
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        output_dir=output_dir,
    )
