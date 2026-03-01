"""Tests for the Brain-Score compatibility bridge."""

import pytest
import warnings

from brain_score_dmt.compat import (
    sync_to_brainscore,
    sync_from_brainscore,
    validate_brainscore_model,
    validate_brainscore_benchmark,
    audit_registry,
)
from brain_score_dmt.brain_model import BrainModelInterface, BenchmarkInterface
from brain_score_dmt.registry import PluginRegistry


# --- Stub models ---

class CompleteBrainModel:
    """Has all BrainModelInterface methods."""
    @property
    def identifier(self):
        return 'complete'

    def visual_degrees(self):
        return 8

    def start_task(self, task, fitting_stimuli=None):
        pass

    def start_recording(self, recording_target, time_bins=None):
        pass

    def look_at(self, stimuli, number_of_trials=1):
        pass


class IncompleteBrainModel:
    """Missing start_recording and start_task."""
    @property
    def identifier(self):
        return 'incomplete'

    def look_at(self, stimuli, number_of_trials=1):
        pass


class CompleteBenchmark:
    @property
    def identifier(self):
        return 'test.bench'

    @property
    def ceiling(self):
        return 0.82


# --- Tests ---

class TestSyncToBrainscore:
    def test_pushes_entries(self):
        dmt_reg = PluginRegistry('model')
        dmt_reg['foo'] = lambda: 42

        bs_reg = {}
        sync_to_brainscore(dmt_reg, bs_reg)

        assert 'foo' in bs_reg
        assert bs_reg['foo']() == 42

    def test_does_not_overwrite_existing(self):
        dmt_reg = PluginRegistry('model')
        dmt_reg['foo'] = lambda: 'new'

        bs_reg = {'foo': lambda: 'existing'}
        sync_to_brainscore(dmt_reg, bs_reg)

        assert bs_reg['foo']() == 'existing'


class TestSyncFromBrainscore:
    def test_imports_entries(self):
        bs_reg = {'bar': lambda: 99}
        dmt_reg = PluginRegistry('model')

        sync_from_brainscore(bs_reg, dmt_reg)

        assert 'bar' in dmt_reg
        assert dmt_reg.load('bar') == 99

    def test_does_not_overwrite_existing(self):
        bs_reg = {'bar': lambda: 'old'}
        dmt_reg = PluginRegistry('model')
        dmt_reg['bar'] = lambda: 'new'

        sync_from_brainscore(bs_reg, dmt_reg)

        assert dmt_reg.load('bar') == 'new'


class TestValidateBrainscoreModel:
    def test_valid_model_returns(self):
        model = validate_brainscore_model(
            CompleteBrainModel, 'complete')
        assert model.identifier == 'complete'

    def test_incomplete_model_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = validate_brainscore_model(
                IncompleteBrainModel, 'incomplete')
            assert len(w) == 1
            assert 'missing BrainModelInterface methods' in str(w[0].message)
            assert 'start_recording' in str(w[0].message)


class TestValidateBrainscoreBenchmark:
    def test_valid_benchmark(self):
        bench = validate_brainscore_benchmark(
            CompleteBenchmark, 'test.bench')
        assert bench.identifier == 'test.bench'


class TestAuditRegistry:
    def test_audit_finds_issues(self):
        bs_reg = {
            'good': CompleteBrainModel,
            'bad': IncompleteBrainModel,
        }
        results = audit_registry(bs_reg, BrainModelInterface)

        assert results['good'] == []
        assert len(results['bad']) > 0
        assert 'start_recording' in results['bad']

    def test_audit_catches_exceptions(self):
        def broken_factory():
            raise RuntimeError("Model load failed")

        bs_reg = {'broken': broken_factory}
        results = audit_registry(bs_reg, BrainModelInterface)

        assert len(results['broken']) == 1
        assert 'ERROR' in results['broken'][0]
