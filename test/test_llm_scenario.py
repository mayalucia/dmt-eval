"""Tests for the LLM QA evaluation scenario.

Three categories:
1. Model resolution (string -> model object)
2. QA dataset and baseline models
3. LLM metrics
4. End-to-end pipeline (evaluate with string specs)
5. CLI integration

API-calling tests are marked @pytest.mark.llm and skipped without keys.
"""

import numpy as np
import pandas as pd
import pytest

from dmt.models import resolve
from dmt.models.baselines import EchoModel, RandomModel, TemplateModel
from dmt.scenario.llm_qa import LLM_QA, generate_dataset
from dmt.metrics.llm import exact_match, fuzzy_match, compute_llm_metrics
from dmt.evaluate import evaluate


# ── Model Resolution ─────────────────────────────────────────────────────────


class TestResolver:
    """String specs resolve to model objects."""

    def test_resolve_echo(self):
        m = resolve("echo")
        assert m.name == "Echo"
        assert hasattr(m, "predict")

    def test_resolve_random(self):
        m = resolve("random")
        assert m.name == "Random"

    def test_resolve_template(self):
        m = resolve("template")
        assert m.name == "Template"

    def test_resolve_anthropic(self):
        m = resolve("anthropic/claude-haiku-4-5-20251001")
        assert m.name == "anthropic/claude-haiku-4-5-20251001"
        assert m.model_id == "claude-haiku-4-5-20251001"

    def test_resolve_openai(self):
        m = resolve("openai/gpt-4o")
        assert m.name == "openai/gpt-4o"
        assert m.model_id == "gpt-4o"

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model spec"):
            resolve("bogus")

    def test_resolve_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            resolve("google/gemini-pro")


# ── QA Dataset ───────────────────────────────────────────────────────────────


class TestDataset:
    """The built-in QA dataset has the right structure."""

    @pytest.fixture
    def dataset(self):
        return generate_dataset()

    def test_columns(self, dataset):
        required = {"question_id", "question", "expected", "category", "difficulty"}
        assert required == set(dataset.columns)

    def test_size(self, dataset):
        assert len(dataset) >= 35  # at least 35 QA pairs

    def test_categories(self, dataset):
        cats = set(dataset["category"].unique())
        assert {"factual", "reasoning", "coding", "creative"}.issubset(cats)

    def test_difficulties(self, dataset):
        diffs = set(dataset["difficulty"].unique())
        assert {"easy", "medium", "hard"}.issubset(diffs)

    def test_unique_ids(self, dataset):
        assert dataset["question_id"].is_unique


# ── Baseline Models ──────────────────────────────────────────────────────────


class TestBaselineModels:
    """Offline baselines satisfy the model protocol."""

    @pytest.fixture
    def dataset(self):
        return generate_dataset()

    @pytest.fixture(params=[EchoModel, RandomModel, TemplateModel])
    def model(self, request):
        return request.param()

    def test_has_name(self, model):
        assert isinstance(model.name, str)
        assert len(model.name) > 0

    def test_predict_returns_dataframe(self, model, dataset):
        result = model.predict(dataset)
        assert isinstance(result, pd.DataFrame)

    def test_predict_has_required_columns(self, model, dataset):
        result = model.predict(dataset)
        assert "question_id" in result.columns
        assert "response" in result.columns

    def test_predict_same_length(self, model, dataset):
        result = model.predict(dataset)
        assert len(result) == len(dataset)

    def test_echo_returns_question(self, dataset):
        result = EchoModel().predict(dataset)
        merged = dataset.merge(result, on="question_id")
        # Echo returns the question itself
        assert (merged["question"] == merged["response"]).all()

    def test_template_gets_arithmetic(self, dataset):
        """Template model should get simple arithmetic right."""
        result = TemplateModel().predict(dataset)
        merged = dataset.merge(result, on="question_id")
        # Check "What is 2 + 2?"
        r01 = merged[merged["question_id"] == "r01"]
        assert r01.iloc[0]["response"] == "4"


# ── LLM Metrics ──────────────────────────────────────────────────────────────


class TestLLMMetrics:
    """Metric functions on known inputs."""

    def test_exact_match_perfect(self):
        expected = np.array(["Paris", "Tokyo", "Berlin"])
        response = np.array(["Paris", "Tokyo", "Berlin"])
        assert exact_match(expected, response) == 1.0

    def test_exact_match_case_insensitive(self):
        expected = np.array(["Paris", "Tokyo"])
        response = np.array(["paris", "TOKYO"])
        assert exact_match(expected, response) == 1.0

    def test_exact_match_none(self):
        expected = np.array(["Paris", "Tokyo"])
        response = np.array(["London", "Beijing"])
        assert exact_match(expected, response) == 0.0

    def test_fuzzy_match_substring(self):
        expected = np.array(["Paris"])
        response = np.array(["The capital of France is Paris."])
        assert fuzzy_match(expected, response) == 1.0

    def test_fuzzy_match_no_match(self):
        expected = np.array(["Paris"])
        response = np.array(["The capital is London"])
        assert fuzzy_match(expected, response) == 0.0

    def test_compute_llm_metrics(self):
        expected = np.array(["4", "Paris", "100"])
        response = np.array(["4", "The answer is Paris", "99"])
        metrics = compute_llm_metrics(expected, response)
        assert "exact_match" in metrics
        assert "fuzzy_match" in metrics
        # "4" exact matches, "Paris" doesn't exact-match but fuzzy-matches
        assert metrics["exact_match"] == pytest.approx(1 / 3)
        assert metrics["fuzzy_match"] == pytest.approx(2 / 3)


# ── End-to-End Pipeline ──────────────────────────────────────────────────────


class TestEndToEnd:
    """Full pipeline with offline baselines."""

    def test_evaluate_with_model_objects(self, tmp_path):
        dataset = generate_dataset()
        models = [EchoModel(), RandomModel(), TemplateModel()]

        report_path = evaluate(
            models=models,
            observations=dataset,
            scenario=LLM_QA,
            output_dir=tmp_path / "llm_report",
            title="LLM QA Evaluation",
        )

        assert report_path.exists()
        report = report_path.read_text()
        assert "# LLM QA Evaluation" in report
        assert "Echo" in report
        assert "Random" in report
        assert "Template" in report

    def test_evaluate_with_string_specs(self, tmp_path):
        """String model specs are resolved automatically."""
        dataset = generate_dataset()

        report_path = evaluate(
            models=["echo", "template"],
            observations=dataset,
            scenario=LLM_QA,
            output_dir=tmp_path / "llm_string_report",
            title="LLM String Spec Test",
        )

        assert report_path.exists()
        report = report_path.read_text()
        assert "Echo" in report
        assert "Template" in report

    def test_stratified_by_category_and_difficulty(self, tmp_path):
        dataset = generate_dataset()
        models = [EchoModel(), TemplateModel()]

        output_dir = tmp_path / "llm_stratified"
        evaluate(
            models=models,
            observations=dataset,
            scenario=LLM_QA,
            output_dir=output_dir,
            title="Stratified Test",
        )

        assert (output_dir / "results.csv").exists()
        assert (output_dir / "results_by_category.csv").exists()
        assert (output_dir / "results_by_difficulty.csv").exists()

    def test_template_beats_echo(self):
        """Template model should have better metrics than Echo."""
        dataset = generate_dataset()
        echo_pred = EchoModel().predict(dataset)
        template_pred = TemplateModel().predict(dataset)

        echo_merged = dataset.merge(echo_pred, on="question_id")
        template_merged = dataset.merge(template_pred, on="question_id")

        echo_em = exact_match(
            echo_merged["expected"].values,
            echo_merged["response"].values,
        )
        template_em = exact_match(
            template_merged["expected"].values,
            template_merged["response"].values,
        )
        # Template knows some answers; Echo never matches
        assert template_em > echo_em


# ── API-based tests (skipped without keys) ───────────────────────────────────

import os

@pytest.mark.llm
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestAnthropicModel:
    """Live API tests — require ANTHROPIC_API_KEY."""

    def test_resolve_and_predict(self):
        model = resolve("anthropic/claude-haiku-4-5-20251001")
        # Use a tiny dataset
        dataset = pd.DataFrame([{
            "question_id": "t01",
            "question": "What is 2 + 2?",
            "expected": "4",
            "category": "reasoning",
            "difficulty": "easy",
        }])
        result = model.predict(dataset)
        assert len(result) == 1
        assert "response" in result.columns
        assert "latency" in result.columns
