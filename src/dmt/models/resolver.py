"""Resolve a model string spec to a callable model object.

Model specs:
    "echo"                              -> EchoModel
    "random"                            -> RandomModel
    "template"                          -> TemplateModel
    "anthropic/<model-id>"              -> AnthropicModel
    "openai/<model-id>"                 -> OpenAIModel

Every resolved model has .name (str) and .predict(observations) -> DataFrame,
satisfying the DMT model protocol used by evaluate().
"""

from __future__ import annotations


def resolve(spec: str):
    """Resolve a model string to a model object.

    Parameters
    ----------
    spec : str
        Model specification.  One of:
        - "echo", "random", "template" (offline baselines)
        - "anthropic/<model-id>" (Anthropic Messages API)
        - "openai/<model-id>" (OpenAI Chat Completions API)

    Returns a model object with .name and .predict().
    """
    spec = spec.strip()

    # Offline baselines
    if spec == "echo":
        from dmt.models.baselines import EchoModel
        return EchoModel()
    if spec == "random":
        from dmt.models.baselines import RandomModel
        return RandomModel()
    if spec == "template":
        from dmt.models.baselines import TemplateModel
        return TemplateModel()

    # Provider-qualified specs
    if "/" in spec:
        provider, model_id = spec.split("/", 1)

        if provider == "anthropic":
            from dmt.models.anthropic import AnthropicModel
            return AnthropicModel(model_id=model_id)

        if provider == "openai":
            from dmt.models.openai import OpenAIModel
            return OpenAIModel(model_id=model_id)

        raise ValueError(
            f"Unknown provider '{provider}' in model spec '{spec}'. "
            f"Supported: anthropic, openai"
        )

    raise ValueError(
        f"Unknown model spec '{spec}'. "
        f"Use 'echo', 'random', 'template', "
        f"'anthropic/<model-id>', or 'openai/<model-id>'."
    )
