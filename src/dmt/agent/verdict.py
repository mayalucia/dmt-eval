"""Structured agent verdict — the JSON output schema.

Instead of prose summaries, agents write a JSON file with explicit fields.
The grader checks field values directly — no keyword matching.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


VERDICT_FILENAME = "agent_verdict.json"


@dataclass
class AgentVerdict:
    """Structured verdict from an agent run.

    Fields
    ------
    best_model : str
        Name of the model the agent considers best.
    best_reason : str
        One-sentence justification.
    worst_model : str
        Name of the worst-performing model.
    worst_reason : str
        One-sentence explanation of the failure mode.
    reference_model : str
        The baseline / reference model used.
    summary : str
        A 2–3 sentence scientific summary (still useful for humans).
    extra : dict
        Domain-specific extras the grader may inspect.
    """
    best_model: str
    best_reason: str
    worst_model: str
    worst_reason: str
    reference_model: str
    summary: str
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(asdict(self), indent=2)

    def save(self, output_dir: str | Path) -> Path:
        """Write to agent_verdict.json in the given directory."""
        path = Path(output_dir) / VERDICT_FILENAME
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, output_dir: str | Path) -> "AgentVerdict":
        """Read from agent_verdict.json.

        Raises FileNotFoundError if the file doesn't exist.
        Raises json.JSONDecodeError or KeyError on malformed JSON.
        """
        path = Path(output_dir) / VERDICT_FILENAME
        data = json.loads(path.read_text())
        return cls(
            best_model=data["best_model"],
            best_reason=data["best_reason"],
            worst_model=data["worst_model"],
            worst_reason=data["worst_reason"],
            reference_model=data["reference_model"],
            summary=data["summary"],
            extra=data.get("extra", {}),
        )
