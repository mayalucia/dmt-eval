"""
Interface declaration using __init_subclass__ instead of metaclasses.

Inspired by DMT's InterfaceMeta, but avoids MRO conflicts and fixes
the shared-registry bug. Any non-dunder method in an Interface subclass
body becomes a required method. Implementations are validated at
registration time, not at runtime.

DMT (Blue Brain Project) used a custom metaclass (InterfaceMeta) that:
- Scanned the class body for non-dunder methods → __requiredmethods__
- Maintained __implementation_registry__ (but as a shared mutable list — bug)
- Validated implementations only up to the first missing method — bug

This version:
- Uses __init_subclass__ (no metaclass, no MRO conflicts)
- Each Interface subclass gets its own registry
- validate() reports ALL missing methods
"""

from __future__ import annotations

from typing import ClassVar


class Interface:
    """Base class for declaring structural interfaces.

    Subclass this to define a contract. Every non-dunder, non-private
    callable in the subclass body becomes a required method::

        class BrainModelInterface(Interface):
            def look_at(self, stimuli): ...
            def start_recording(self, target, time_bins): ...

    Then validate implementations::

        missing = BrainModelInterface.validate(MyModel)
        # [] if compliant, ['look_at', ...] if not
    """

    __required_methods__: ClassVar[set[str]] = set()
    __implementation_registry__: ClassVar[dict[str, type]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Capture all non-dunder, non-private callables as required methods.
        # Properties count — they define the interface contract too.
        cls.__required_methods__ = {
            name for name, val in vars(cls).items()
            if not name.startswith('_')
            and (callable(val) or isinstance(val, (property, classmethod, staticmethod)))
        }
        # Each subclass gets its own registry — not shared with parent.
        cls.__implementation_registry__ = {}

    @classmethod
    def validate(cls, impl) -> list[str]:
        """Return list of method names required by this Interface
        but missing from impl (a class or instance).

        Returns an empty list if impl is fully compliant.
        """
        return sorted([
            m for m in cls.__required_methods__
            if not hasattr(impl, m)
        ])

    @classmethod
    def is_implemented_by(cls, impl) -> bool:
        """True if impl has all required methods."""
        return len(cls.validate(impl)) == 0

    @classmethod
    def register(cls, name: str, impl_class: type):
        """Validate impl_class and register it under name.

        Raises TypeError if impl_class is missing required methods.
        """
        missing = cls.validate(impl_class)
        if missing:
            raise TypeError(
                f"{impl_class.__name__} does not implement {cls.__name__}. "
                f"Missing: {missing}"
            )
        cls.__implementation_registry__[name] = impl_class

    @classmethod
    def implementations(cls) -> dict[str, type]:
        """Return all registered implementations."""
        return dict(cls.__implementation_registry__)

    @classmethod
    def implementation_guide(cls) -> str:
        """Human-readable guide: what methods must an implementation provide?"""
        lines = [f"To implement {cls.__name__}, provide these methods:"]
        for method_name in sorted(cls.__required_methods__):
            method = getattr(cls, method_name, None)
            doc = getattr(method, '__doc__', None) or '(no docstring)'
            lines.append(f"  - {method_name}: {doc.strip()}")
        return '\n'.join(lines)
