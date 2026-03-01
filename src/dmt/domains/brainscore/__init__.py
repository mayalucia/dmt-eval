"""
brain-score-dmt: DMT-style plugin infrastructure for Brain-Score.

A lightweight metaprogramming layer inspired by DMT (Digital Model Testing,
Blue Brain Project), designed to wrap over Brain-Score's existing plugin
system with proper interface enforcement, validated registration, and
Python-native discovery.

Key improvements over Brain-Score's current system:
- Interface enforcement: BrainModelInterface validates at registration, not runtime
- Validated registration: @implements and PluginRegistry.register() check compliance
- Auto-discovery: __init_subclass__ replaces text search on __init__.py files
- Bug fixes over DMT: per-interface registries, report ALL missing methods

Key improvements over DMT's metaclass approach:
- __init_subclass__ instead of InterfaceMeta: no MRO conflicts
- No compound metaclass (AIMeta): simpler, no ClassAttributeMeta dependency
- No runtime proxy synthesis (adapted()): unnecessary for Brain-Score's API
- No Field descriptors: Brain-Score uses xarray for data validation

Usage::

    from brain_score_dmt import (
        Interface,
        PluginRegistry,
        ModelPlugin,
        BenchmarkPlugin,
        BrainModelInterface,
        BenchmarkInterface,
        adapts,
        implements,
    )

    # Define a model plugin (auto-registers)
    class MyModel(ModelPlugin, identifier='my_model'):
        @classmethod
        def build(cls):
            return ModelCommitment(...)

    # Use the registry
    from brain_score_dmt.plugin import model_registry
    model = model_registry.load('my_model')

    # Bridge to Brain-Score
    from brain_score_dmt.compat import sync_to_brainscore
    from brainscore_vision import model_registry as bs_registry
    sync_to_brainscore(model_registry, bs_registry)
"""

from .interface import Interface
from .registry import PluginRegistry
from .adapter import adapts, implements
from .brain_model import BrainModelInterface, BenchmarkInterface, MetricInterface
from .plugin import (
    ModelPlugin,
    BenchmarkPlugin,
    MetricPlugin,
    DataPlugin,
    model_registry,
    benchmark_registry,
    metric_registry,
    data_registry,
    stimulus_set_registry,
)

__all__ = [
    # Core
    'Interface',
    'PluginRegistry',
    # Decorators
    'adapts',
    'implements',
    # Brain-Score interfaces
    'BrainModelInterface',
    'BenchmarkInterface',
    'MetricInterface',
    # Plugin base classes
    'ModelPlugin',
    'BenchmarkPlugin',
    'MetricPlugin',
    'DataPlugin',
    # Registries
    'model_registry',
    'benchmark_registry',
    'metric_registry',
    'data_registry',
    'stimulus_set_registry',
]
