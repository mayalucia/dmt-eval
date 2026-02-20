"""Send an agent brief to an LLM and execute the generated code.

Currently supports Anthropic Claude.  The runner:
1. Sends the brief as a user message
2. Extracts a Python script from the response
3. Writes it to a temp file and runs it in a subprocess
4. Returns the AgentResult for grading
"""

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from dmt.agent.brief import AgentBrief
from dmt.agent.runner import AgentResult


@dataclass
class LLMResponse:
    """Raw response from the LLM."""
    model: str
    raw_text: str
    extracted_code: str
    usage: dict = field(default_factory=dict)


def _extract_python_code(text: str) -> str:
    """Extract Python code from a markdown-fenced response.

    Looks for ```python ... ``` blocks.  If multiple blocks exist,
    concatenates them (the agent may split imports from main logic).
    If no fenced blocks, assumes the entire response is code.
    """
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(blocks)
    # Fallback: try generic code fence
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(blocks)
    # Last resort: the whole thing
    return text


def call_claude(
    brief: AgentBrief,
    output_dir: str | Path,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
) -> LLMResponse:
    """Send the brief to Claude and get back a response.

    Requires ANTHROPIC_API_KEY in the environment.

    Raises
    ------
    RuntimeError
        If ANTHROPIC_API_KEY is not set or empty.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set or empty.\n"
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    output_dir = Path(output_dir)
    system_prompt = (
        "You are a scientific computing agent. "
        "Respond with a single, complete Python script that accomplishes "
        "the task described in the brief. The script must:\n"
        "- Be self-contained (all imports at the top)\n"
        "- Write outputs to the directory specified in the constraints\n"
        "- Write a file called 'agent_summary.txt' to the output directory "
        "containing a 3-sentence scientific summary\n"
        "- Use only the imports listed in the brief\n"
        "- Be executable with: python script.py <output_dir>\n\n"
        "Wrap your code in a ```python code fence."
    )

    user_message = (
        brief.to_prompt()
        + f"\n\nThe output directory is: {output_dir}\n"
        "Respond with only the Python script."
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text
    code = _extract_python_code(raw_text)

    return LLMResponse(
        model=model,
        raw_text=raw_text,
        extracted_code=code,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    )


def run_llm_agent(
    brief: AgentBrief,
    output_dir: str | Path,
    model: str = "claude-sonnet-4-20250514",
    repo_root: str | Path | None = None,
    max_tokens: int = 4096,
    timeout: int = 60,
) -> tuple[LLMResponse, AgentResult]:
    """Full pipeline: brief -> Claude -> code -> execute -> result.

    Parameters
    ----------
    brief : AgentBrief
        The task specification.
    output_dir : path
        Where the agent should write its outputs.
    model : str
        Claude model to use.
    repo_root : path, optional
        Working directory for execution (must have dmt importable).
        If None, uses the dmt-eval repo root.
    max_tokens : int
        Max tokens for Claude response.
    timeout : int
        Max seconds for the generated script to run.

    Returns (LLMResponse, AgentResult).
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the LLM's code
    llm_response = call_claude(brief, output_dir, model, max_tokens)

    # Write the code to a temp file
    script_dir = output_dir / "_agent_workspace"
    script_dir.mkdir(exist_ok=True)
    script_path = script_dir / "agent_script.py"
    script_path.write_text(llm_response.extracted_code)

    # Also save the raw response for debugging
    (script_dir / "llm_raw_response.txt").write_text(llm_response.raw_text)

    # Determine repo root for PYTHONPATH
    if repo_root is None:
        # Walk up from this file to find pyproject.toml
        candidate = Path(__file__).resolve().parent
        while candidate != candidate.parent:
            if (candidate / "pyproject.toml").exists():
                repo_root = candidate
                break
            candidate = candidate.parent
        else:
            repo_root = Path.cwd()
    repo_root = Path(repo_root)

    # Execute the script
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path

    result = subprocess.run(
        [sys.executable, str(script_path), str(output_dir)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(repo_root),
        env=env,
    )

    agent_result = AgentResult(
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        output_dir=output_dir,
    )

    return llm_response, agent_result
