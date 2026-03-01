"""
Adapter decorators: @adapts and @implements.

DMT used two decorators to wire adapters:
  @adapts(ModelClass)        — marks what model type this adapter handles
  @implements(Interface)     — validates the adapter implements the interface

Both are parametric class decorators that annotate the class and
optionally register it in the interface's implementation registry.

This version simplifies DMT's approach:
- No ABCMeta.register() (unnecessary for Brain-Score's use case)
- @implements validates ALL missing methods (DMT only reported the first)
- Clean attribute initialization (no try/except AttributeError pattern)
"""

from __future__ import annotations

from typing import Type

from .interface import Interface


def adapts(*model_classes: type):
    """Mark an adapter class as adapting one or more model classes.

    Usage::

        @adapts(torch.nn.Module)
        class PytorchAdapter:
            ...

    Or multiple models::

        @adapts(AlexNet, ResNet)
        class VisionModelAdapter:
            ...

    The decorated class gets ``__adapted_models__``: a dict mapping
    class name → class.
    """
    def decorator(adapter_cls):
        if not hasattr(adapter_cls, '__adapted_models__'):
            adapter_cls.__adapted_models__ = {}
        for model_class in model_classes:
            adapter_cls.__adapted_models__[model_class.__name__] = model_class
        return adapter_cls
    return decorator


def implements(interface_cls: Type[Interface], name: str = None):
    """Validate and register a class as implementing an Interface.

    Validates at decoration time that all required methods are present.
    Raises TypeError immediately if any are missing — fail fast, not
    at scoring time.

    Usage::

        @implements(BrainModelInterface)
        class MyModel:
            def look_at(self, stimuli): ...
            def start_recording(self, target, time_bins): ...
            ...

    With explicit registration name::

        @implements(BrainModelInterface, name='my_model')
        class MyModel:
            ...

    The decorated class gets ``__implemented_interfaces__``: a dict
    mapping interface name → interface class.
    """
    def decorator(impl_cls):
        # Validate
        missing = interface_cls.validate(impl_cls)
        if missing:
            guide = interface_cls.implementation_guide()
            raise TypeError(
                f"{impl_cls.__name__} does not implement "
                f"{interface_cls.__name__}.\n"
                f"Missing methods: {missing}\n\n"
                f"{guide}"
            )

        # Annotate the class
        if not hasattr(impl_cls, '__implemented_interfaces__'):
            impl_cls.__implemented_interfaces__ = {}
        impl_cls.__implemented_interfaces__[interface_cls.__name__] = interface_cls

        # Register in the interface's registry
        reg_name = name or impl_cls.__name__
        interface_cls.register(reg_name, impl_cls)

        return impl_cls
    return decorator


def get_adapted_models(adapter_cls) -> dict[str, type]:
    """Return the models this adapter handles."""
    return getattr(adapter_cls, '__adapted_models__', {})


def get_implemented_interfaces(impl_cls) -> dict[str, Type[Interface]]:
    """Return the interfaces this class implements."""
    return getattr(impl_cls, '__implemented_interfaces__', {})
