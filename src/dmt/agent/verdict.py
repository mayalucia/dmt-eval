"""Structured agent verdict — the JSON output schema.

Instead of prose summaries, agents write a JSON file with explicit fields.
The grader checks field values directly — no keyword matching.

Lesson 07: schema validation before grading.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


VERDICT_FILENAME = "agent_verdict.json"

# Required fields and their expected types
REQUIRED_FIELDS: dict[str, type] = {
    "best_model": str,
    "best_reason": str,
    "worst_model": str,
    "worst_reason": str,
    "reference_model": str,
    "summary": str,
}


@dataclass
class ValidationResult:
    """Result of validating a verdict against the schema."""
    valid: bool
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.valid:
            return "verdict valid"
        return "verdict invalid: " + "; ".join(self.errors)


def validate_verdict(data: dict) -> ValidationResult:
    """Validate a parsed verdict dict against the schema.

    Checks:
    - All required fields are present
    - All required fields are non-empty strings
    - No type violations

    Does NOT check domain correctness (e.g. whether best_model is
    actually the best).  That's the grader's job.
    """
    errors = []

    for field_name, expected_type in REQUIRED_FIELDS.items():
        if field_name not in data:
            errors.append(f"missing: {field_name}")
            continue

        value = data[field_name]
        if not isinstance(value, expected_type):
            actual = type(value).__name__
            errors.append(
                f"{field_name}: expected {expected_type.__name__}, got {actual}"
            )
        elif isinstance(value, str) and not value.strip():
            errors.append(f"{field_name}: empty string")

    return ValidationResult(valid=len(errors) == 0, errors=errors)


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

    @classmethod
    def load_validated(cls, output_dir: str | Path) -> tuple["AgentVerdict | None", ValidationResult]:
        """Load and validate in one step.

        Returns (verdict, validation_result).
        If the file doesn't exist or isn't valid JSON, returns (None, result_with_errors).
        If schema validation fails, returns (None, result_with_errors).
        If valid, returns (verdict, result_ok).
        """
        path = Path(output_dir) / VERDICT_FILENAME
        if not path.exists():
            return None, ValidationResult(valid=False, errors=["file not found"])
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            return None, ValidationResult(valid=False, errors=[f"invalid JSON: {e}"])

        if not isinstance(data, dict):
            return None, ValidationResult(
                valid=False, errors=[f"expected JSON object, got {type(data).__name__}"]
            )

        result = validate_verdict(data)
        if not result.valid:
            return None, result

        verdict = cls(
            best_model=data["best_model"],
            best_reason=data["best_reason"],
            worst_model=data["worst_model"],
            worst_reason=data["worst_reason"],
            reference_model=data["reference_model"],
            summary=data["summary"],
            extra=data.get("extra", {}),
        )
        return verdict, result
