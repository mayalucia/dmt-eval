"""
Validated plugin registry.

A PluginRegistry is a dict[str, Callable] that optionally validates
entries against an Interface. It also provides a @register decorator
for class-based plugin definitions.

Brain-Score uses plain dicts as registries:
    model_registry: Dict[str, Callable[[], BrainModel]] = {}

This replaces that with a dict that knows what it holds and can
validate entries at registration time.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Type

from .interface import Interface


class PluginRegistry(dict):
    """A registry dict that optionally validates against an Interface.

    Usage::

        model_registry = PluginRegistry('model', BrainModelInterface)

        # Decorator registration (preferred)
        @model_registry.register('alexnet')
        class AlexNetPlugin:
            @classmethod
            def build(cls):
                return ModelCommitment(...)

        # Direct assignment (backward compatible)
        model_registry['alexnet'] = lambda: ModelCommitment(...)

        # Load
        model = model_registry.load('alexnet')
    """

    def __init__(self, name: str, interface: Optional[Type[Interface]] = None):
        super().__init__()
        self.name = name
        self.interface = interface

    def register(self, identifier: str):
        """Class decorator: register a plugin class under identifier.

        The class must define a classmethod ``build()`` that returns
        the plugin instance. If an interface is set, the class is
        validated against it at decoration time.

        ::

            @model_registry.register('alexnet')
            class AlexNetPlugin:
                @classmethod
                def build(cls):
                    return ModelCommitment(...)
        """
        def decorator(cls):
            # Validate the class against the interface if one is set.
            # We validate the class itself (must have build()), not the
            # build output (that would require calling the factory).
            if self.interface is not None:
                self.interface.register(identifier, cls)

            # Store the factory function
            if hasattr(cls, 'build') and callable(getattr(cls, 'build')):
                self[identifier] = cls.build
            else:
                raise TypeError(
                    f"Plugin class {cls.__name__} must define a "
                    f"classmethod build()"
                )
            return cls
        return decorator

    def register_factory(self, identifier: str, factory: Callable):
        """Register a bare factory function (lambda or function).

        This is the backward-compatible path: no class, no validation.
        Equivalent to Brain-Score's model_registry['id'] = lambda: ...
        """
        self[identifier] = factory

    def load(self, identifier: str) -> Any:
        """Load a plugin by identifier: call its factory."""
        if identifier not in self:
            raise KeyError(
                f"No plugin '{identifier}' in {self.name} registry. "
                f"Available: {sorted(self.keys())}"
            )
        return self[identifier]()

    def validate_output(self, identifier: str) -> list[str]:
        """Build a plugin and validate the output against the interface.

        Returns list of missing methods (empty if compliant).
        This is expensive (calls the factory) — use for testing, not
        on every load.
        """
        if self.interface is None:
            return []
        obj = self.load(identifier)
        return self.interface.validate(obj)

    def __repr__(self):
        entries = ', '.join(sorted(self.keys()))
        return f"PluginRegistry('{self.name}', [{entries}])"
