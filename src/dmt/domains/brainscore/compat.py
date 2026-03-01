"""
Compatibility bridge between brain-score-dmt and Brain-Score's existing
plain-dict registries.

Two directions:
1. DMT → Brain-Score: push validated registrations into Brain-Score's dicts
2. Brain-Score → DMT: wrap existing Brain-Score factories with validation

This allows incremental adoption: new plugins use the DMT-style system,
existing plugins continue to work through the compatibility layer.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict

from .interface import Interface
from .brain_model import BrainModelInterface, BenchmarkInterface
from .registry import PluginRegistry


def sync_to_brainscore(
    plugin_registry: PluginRegistry,
    brainscore_registry: Dict[str, Callable]
):
    """Push DMT-style registrations into Brain-Score's plain dicts.

    Entries already in brainscore_registry are not overwritten.

    Usage::

        from brainscore_vision import model_registry as bs_model_registry
        from brain_score_dmt.plugin import model_registry as dmt_model_registry

        sync_to_brainscore(dmt_model_registry, bs_model_registry)
        # Now brainscore_vision.load_model('alexnet') works with
        # plugins registered via the DMT system.
    """
    for identifier, factory in plugin_registry.items():
        if identifier not in brainscore_registry:
            brainscore_registry[identifier] = factory


def sync_from_brainscore(
    brainscore_registry: Dict[str, Callable],
    plugin_registry: PluginRegistry
):
    """Import Brain-Score plain-dict entries into a DMT PluginRegistry.

    Entries already in plugin_registry are not overwritten.
    No validation is performed (legacy entries may not conform).

    Usage::

        sync_from_brainscore(bs_model_registry, dmt_model_registry)
    """
    for identifier, factory in brainscore_registry.items():
        if identifier not in plugin_registry:
            plugin_registry.register_factory(identifier, factory)


def validate_brainscore_model(factory: Callable, identifier: str) -> Any:
    """Call a Brain-Score model factory and validate the output.

    Returns the model if valid. Warns if methods are missing.

    Usage::

        model = validate_brainscore_model(
            model_registry['alexnet'], 'alexnet')
    """
    model = factory()
    missing = BrainModelInterface.validate(model)
    if missing:
        warnings.warn(
            f"Model '{identifier}' is missing BrainModelInterface methods: "
            f"{missing}. It may fail on some benchmarks.",
            stacklevel=2
        )
    return model


def validate_brainscore_benchmark(factory: Callable, identifier: str) -> Any:
    """Call a Brain-Score benchmark factory and validate the output."""
    benchmark = factory()
    missing = BenchmarkInterface.validate(benchmark)
    if missing:
        warnings.warn(
            f"Benchmark '{identifier}' is missing BenchmarkInterface methods: "
            f"{missing}.",
            stacklevel=2
        )
    return benchmark


def audit_registry(
    brainscore_registry: Dict[str, Callable],
    interface: type[Interface]
) -> dict[str, list[str]]:
    """Audit an entire Brain-Score registry for interface compliance.

    Calls every factory and validates the output. Returns a dict
    mapping identifier → list of missing methods (empty if compliant).

    This is expensive (instantiates every plugin). Use for CI/testing.

    Usage::

        issues = audit_registry(model_registry, BrainModelInterface)
        for name, missing in issues.items():
            if missing:
                print(f"{name}: missing {missing}")
    """
    results = {}
    for identifier, factory in brainscore_registry.items():
        try:
            obj = factory()
            results[identifier] = interface.validate(obj)
        except Exception as e:
            results[identifier] = [f"ERROR: {e}"]
    return results
