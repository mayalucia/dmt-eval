"""
Brain-Score interfaces expressed as proper Interface declarations.

Brain-Score defines BrainModel, Benchmark, and Metric as informal
interfaces (ABC with no enforcement). Here we formalize them: any
class claiming to be a BrainModel must actually have look_at(),
start_recording(), etc. — verified at registration time, not at
runtime failure.
"""

from __future__ import annotations

from .interface import Interface


class BrainModelInterface(Interface):
    """The contract every model must fulfill for Brain-Score benchmarks.

    Mirrors brainscore_vision.model_interface.BrainModel, but enforced.
    """

    def identifier(self) -> str:
        """Unique model name, e.g. 'alexnet'."""
        ...

    def visual_degrees(self) -> int:
        """How many visual degrees the model's input subtends."""
        ...

    def start_task(self, task, fitting_stimuli):
        """Configure the model for a behavioral task."""
        ...

    def start_recording(self, recording_target, time_bins):
        """Configure the model to record from a brain region."""
        ...

    def look_at(self, stimuli, number_of_trials=1):
        """Present stimuli, return NeuroidAssembly or BehavioralAssembly."""
        ...


class BenchmarkInterface(Interface):
    """The contract every benchmark must fulfill.

    Mirrors brainscore_core.benchmarks.Benchmark.
    """

    def identifier(self) -> str:
        """Benchmark identifier, e.g. 'dicarlo.MajajHong2015public.IT-pls'."""
        ...

    def ceiling(self):
        """Data reliability estimate (Score)."""
        ...


class MetricInterface(Interface):
    """The contract every metric must fulfill.

    Mirrors brainscore_core.metrics.Metric.
    """
    pass
    # Note: Metric's contract is __call__(assembly1, assembly2) -> Score.
    # __call__ is a dunder, so it's not captured by Interface.__init_subclass__.
    # This is a deliberate design choice: __call__ is universal, not specific
    # to the Metric contract. Validation of __call__ signature would require
    # inspect.signature, which is fragile with C extensions and wrappers.
    # For Metric, the interface is effectively "be callable with two assemblies."
