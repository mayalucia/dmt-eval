"""Tests for ModelPlugin and BenchmarkPlugin auto-registration."""

import pytest

# We need fresh registries per test, so we import and manipulate them.
from brain_score_dmt import plugin as plugin_module


class TestModelPlugin:
    def setup_method(self):
        """Clear model_registry before each test."""
        plugin_module.model_registry.clear()

    def test_auto_registration(self):
        class MyModel(plugin_module.ModelPlugin, identifier='test_model'):
            @classmethod
            def build(cls):
                return type('FakeModel', (), {
                    'identifier': property(lambda self: 'test_model'),
                    'visual_degrees': lambda self: 8,
                    'start_task': lambda self, t, f=None: None,
                    'start_recording': lambda self, r, t=None: None,
                    'look_at': lambda self, s, n=1: None,
                })()

        assert 'test_model' in plugin_module.model_registry
        model = plugin_module.model_registry.load('test_model')
        assert model.identifier == 'test_model'

    def test_missing_build_raises(self):
        with pytest.raises(TypeError, match="must override build"):
            class BadModel(plugin_module.ModelPlugin, identifier='bad'):
                pass

    def test_no_identifier_no_registration(self):
        """Subclassing without identifier= does not register."""
        class AbstractModel(plugin_module.ModelPlugin):
            @classmethod
            def build(cls):
                return None

        assert len(plugin_module.model_registry) == 0

    def test_identifier_stored(self):
        class Tagged(plugin_module.ModelPlugin, identifier='tagged'):
            @classmethod
            def build(cls):
                return None

        assert Tagged._identifier == 'tagged'


class TestBenchmarkPlugin:
    def setup_method(self):
        plugin_module.benchmark_registry.clear()

    def test_auto_registration(self):
        class MyBench(plugin_module.BenchmarkPlugin,
                       identifier='test.bench.IT-pls'):
            @classmethod
            def build(cls):
                return type('FakeBench', (), {
                    'identifier': property(lambda self: 'test.bench.IT-pls'),
                    'ceiling': property(lambda self: 0.82),
                    '__call__': lambda self, candidate: 0.59,
                })()

        assert 'test.bench.IT-pls' in plugin_module.benchmark_registry
        bench = plugin_module.benchmark_registry.load('test.bench.IT-pls')
        assert bench.identifier == 'test.bench.IT-pls'

    def test_multiple_benchmarks_same_dataset(self):
        """Two plugins from the same dataset (V4 and IT) coexist."""

        class V4Bench(plugin_module.BenchmarkPlugin,
                       identifier='test.V4-pls'):
            @classmethod
            def build(cls):
                return type('V4', (), {
                    'identifier': property(lambda s: 'test.V4-pls'),
                    'ceiling': property(lambda s: 0.89),
                })()

        class ITBench(plugin_module.BenchmarkPlugin,
                       identifier='test.IT-pls'):
            @classmethod
            def build(cls):
                return type('IT', (), {
                    'identifier': property(lambda s: 'test.IT-pls'),
                    'ceiling': property(lambda s: 0.82),
                })()

        assert len(plugin_module.benchmark_registry) == 2
        assert 'test.V4-pls' in plugin_module.benchmark_registry
        assert 'test.IT-pls' in plugin_module.benchmark_registry


class TestExamplesIntegration:
    """Test that the example plugins register correctly."""

    def setup_method(self):
        plugin_module.model_registry.clear()
        plugin_module.benchmark_registry.clear()

    def test_alexnet_example(self):
        # Importing the module triggers registration
        from brain_score_dmt.examples import alexnet  # noqa: F401

        assert 'alexnet' in plugin_module.model_registry
        model = plugin_module.model_registry.load('alexnet')
        assert model.identifier == 'alexnet'
        assert model.visual_degrees() == 8

    def test_benchmark_example(self):
        from brain_score_dmt.examples import benchmark  # noqa: F401

        assert 'dicarlo.MajajHong2015public.IT-pls' in plugin_module.benchmark_registry
        assert 'dicarlo.MajajHong2015public.V4-pls' in plugin_module.benchmark_registry

        it_bench = plugin_module.benchmark_registry.load(
            'dicarlo.MajajHong2015public.IT-pls')
        assert float(it_bench.ceiling) == pytest.approx(0.817)
