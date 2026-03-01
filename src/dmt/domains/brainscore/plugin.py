"""
Plugin base classes with auto-registration via __init_subclass__.

Brain-Score discovers plugins by text-searching __init__.py files for
literal strings like model_registry['alexnet']. This replaces that
with Python's own class machinery: subclass ModelPlugin with an
identifier= argument and you're registered.

The old way (Brain-Score):
    model_registry['alexnet'] = lambda: ModelCommitment(...)

The new way:
    class AlexNetPlugin(ModelPlugin, identifier='alexnet'):
        @classmethod
        def build(cls):
            return ModelCommitment(...)

Both produce a factory in the registry. The new way additionally:
- Is discoverable by normal Python import (no text search)
- Can validate that build() exists at class definition time
- Can validate the build output at test time
"""

from __future__ import annotations

from .registry import PluginRegistry
from .brain_model import BrainModelInterface, BenchmarkInterface


# Module-level registries — the equivalents of Brain-Score's globals.
model_registry = PluginRegistry('model', BrainModelInterface)
benchmark_registry = PluginRegistry('benchmark', BenchmarkInterface)
metric_registry = PluginRegistry('metric')
data_registry = PluginRegistry('data')
stimulus_set_registry = PluginRegistry('stimulus_set')


class ModelPlugin:
    """Base class for model plugins. Subclassing auto-registers.

    Usage::

        class AlexNetPlugin(ModelPlugin, identifier='alexnet'):
            @classmethod
            def build(cls):
                from .model import get_model, get_layers
                return ModelCommitment(
                    identifier='alexnet',
                    activations_model=get_model('alexnet'),
                    layers=get_layers('alexnet'))

    The class is registered in model_registry at definition time.
    build() is called lazily when the model is actually loaded.
    """

    def __init_subclass__(cls, identifier: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if identifier is not None:
            # Check that the subclass defines its OWN build(), not just
            # inherits the base class placeholder.
            if 'build' not in vars(cls):
                raise TypeError(
                    f"{cls.__name__} must override build() classmethod"
                )
            model_registry[identifier] = cls.build
            cls._identifier = identifier

    @classmethod
    def build(cls):
        """Override this to construct and return a BrainModel."""
        raise NotImplementedError(
            f"{cls.__name__}.build() not implemented"
        )


class BenchmarkPlugin:
    """Base class for benchmark plugins.

    Usage::

        class MajajHong2015ITPlugin(BenchmarkPlugin,
                                     identifier='dicarlo.MajajHong2015public.IT-pls'):
            @classmethod
            def build(cls):
                return NeuralBenchmark(...)
    """

    def __init_subclass__(cls, identifier: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if identifier is not None:
            if 'build' not in vars(cls):
                raise TypeError(
                    f"{cls.__name__} must override build() classmethod"
                )
            benchmark_registry[identifier] = cls.build
            cls._identifier = identifier

    @classmethod
    def build(cls):
        """Override this to construct and return a Benchmark."""
        raise NotImplementedError(
            f"{cls.__name__}.build() not implemented"
        )


class MetricPlugin:
    """Base class for metric plugins."""

    def __init_subclass__(cls, identifier: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if identifier is not None:
            if 'build' not in vars(cls):
                raise TypeError(
                    f"{cls.__name__} must override build() classmethod"
                )
            metric_registry[identifier] = cls.build
            cls._identifier = identifier

    @classmethod
    def build(cls):
        """Override this to construct and return a Metric."""
        raise NotImplementedError(
            f"{cls.__name__}.build() not implemented"
        )


class DataPlugin:
    """Base class for data assembly plugins."""

    def __init_subclass__(cls, identifier: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if identifier is not None:
            if 'build' not in vars(cls):
                raise TypeError(
                    f"{cls.__name__} must override build() classmethod"
                )
            data_registry[identifier] = cls.build
            cls._identifier = identifier

    @classmethod
    def build(cls):
        raise NotImplementedError(
            f"{cls.__name__}.build() not implemented"
        )
