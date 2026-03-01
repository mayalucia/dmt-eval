"""Tests for PluginRegistry."""

import pytest
from brain_score_dmt.interface import Interface
from brain_score_dmt.registry import PluginRegistry


class MathInterface(Interface):
    def compute(self, x): ...


class GoodPlugin:
    def compute(self, x):
        return x * 2

    @classmethod
    def build(cls):
        return cls()


class BadPlugin:
    """Missing compute()."""
    @classmethod
    def build(cls):
        return cls()


class TestPluginRegistry:
    def test_direct_assignment(self):
        reg = PluginRegistry('test')
        reg['foo'] = lambda: 42
        assert reg.load('foo') == 42

    def test_register_factory(self):
        reg = PluginRegistry('test')
        reg.register_factory('bar', lambda: 'hello')
        assert reg.load('bar') == 'hello'

    def test_load_missing_raises(self):
        reg = PluginRegistry('test')
        with pytest.raises(KeyError, match="No plugin 'missing'"):
            reg.load('missing')

    def test_repr(self):
        reg = PluginRegistry('model')
        reg['a'] = lambda: None
        reg['b'] = lambda: None
        assert 'model' in repr(reg)
        assert 'a' in repr(reg)


class TestDecoratorRegistration:
    def test_register_with_interface(self):
        reg = PluginRegistry('test', MathInterface)

        @reg.register('good')
        class Plugin:
            def compute(self, x):
                return x

            @classmethod
            def build(cls):
                return cls()

        assert 'good' in reg
        obj = reg.load('good')
        assert obj.compute(5) == 5

    def test_register_missing_interface_method_raises(self):
        reg = PluginRegistry('test', MathInterface)

        with pytest.raises(TypeError, match="Missing.*compute"):
            @reg.register('bad')
            class Plugin:
                @classmethod
                def build(cls):
                    return cls()

    def test_register_missing_build_raises(self):
        reg = PluginRegistry('test')

        with pytest.raises(TypeError, match="must define.*build"):
            @reg.register('no_build')
            class Plugin:
                pass


class TestValidateOutput:
    def test_validate_output_compliant(self):
        reg = PluginRegistry('test', MathInterface)
        reg.register_factory('good', GoodPlugin.build)
        assert reg.validate_output('good') == []

    def test_validate_output_noncompliant(self):
        reg = PluginRegistry('test', MathInterface)
        reg.register_factory('bad', BadPlugin.build)
        missing = reg.validate_output('bad')
        assert 'compute' in missing

    def test_validate_output_no_interface(self):
        reg = PluginRegistry('test')  # no interface
        reg.register_factory('any', lambda: object())
        assert reg.validate_output('any') == []
