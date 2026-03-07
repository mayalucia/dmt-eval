"""Model resolution: string specs to callable model objects.

Usage::

    from dmt.models import resolve

    model = resolve("echo")              # offline baseline
    model = resolve("anthropic/claude-haiku-4-5-20251001")  # API
    model = resolve("openai/gpt-4o")     # API
"""

from dmt.models.resolver import resolve

__all__ = ["resolve"]
