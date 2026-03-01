"""
Example: MajajHong2015 benchmark plugin in the DMT-style system.

Current Brain-Score:
    # brainscore_vision/benchmarks/majajhong2015/__init__.py
    benchmark_registry['dicarlo.MajajHong2015public.IT-pls'] = (
        lambda: load_assembly_and_build_benchmark('IT'))

New style:
    class MajajHong2015IT(BenchmarkPlugin,
                           identifier='dicarlo.MajajHong2015public.IT-pls'):
        @classmethod
        def build(cls):
            return NeuralBenchmark(...)

One plugin class per benchmark variant. Multiple variants from the
same dataset are sibling classes (not one lambda-generating loop).
"""

from brain_score_dmt.plugin import BenchmarkPlugin


# --- Stub types ---

class Score:
    """Stub for brainscore_core.metrics.Score"""
    def __init__(self, value, error=0.0):
        self.value = value
        self.error = error

    def __float__(self):
        return self.value


class NeuralBenchmark:
    """Stub for brainscore_vision.benchmark_helpers.neural_common.NeuralBenchmark"""
    def __init__(self, identifier, assembly, similarity_metric,
                 ceiling, visual_degrees=8, number_of_trials=1,
                 parent='neural', version=3, bibtex=None):
        self._identifier = identifier
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        self._ceiling = ceiling
        self._visual_degrees = visual_degrees
        self._parent = parent
        self._version = version

    @property
    def identifier(self):
        return self._identifier

    @property
    def ceiling(self):
        return self._ceiling

    def __call__(self, candidate):
        # In real code: run the benchmark protocol
        return Score(0.588, error=0.007)


# --- Plugin definitions ---

class MajajHong2015V4(BenchmarkPlugin,
                       identifier='dicarlo.MajajHong2015public.V4-pls'):
    """V4 neural predictivity benchmark, public data."""

    @classmethod
    def build(cls):
        return NeuralBenchmark(
            identifier='dicarlo.MajajHong2015public.V4-pls',
            assembly=None,  # would be: load_assembly('MajajHong2015.public', 'V4')
            similarity_metric=None,  # would be: load_metric('pls')
            ceiling=Score(0.892),
            visual_degrees=8,
            number_of_trials=1,
            parent='neural',
            version=3)


class MajajHong2015IT(BenchmarkPlugin,
                       identifier='dicarlo.MajajHong2015public.IT-pls'):
    """IT neural predictivity benchmark, public data."""

    @classmethod
    def build(cls):
        return NeuralBenchmark(
            identifier='dicarlo.MajajHong2015public.IT-pls',
            assembly=None,  # would be: load_assembly('MajajHong2015.public', 'IT')
            similarity_metric=None,  # would be: load_metric('pls')
            ceiling=Score(0.817),
            visual_degrees=8,
            number_of_trials=1,
            parent='neural',
            version=3)


if __name__ == '__main__':
    from brain_score_dmt.plugin import benchmark_registry

    print("Registered benchmarks:")
    for name in sorted(benchmark_registry.keys()):
        bench = benchmark_registry.load(name)
        print(f"  {name}: ceiling={float(bench.ceiling):.3f}")
