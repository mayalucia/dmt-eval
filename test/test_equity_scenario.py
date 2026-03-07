"""Tests for the equity forecasting scenario.

Three categories:
1. Synthetic data properties (fat tails, vol clustering)
2. Model protocol compliance
3. End-to-end pipeline (evaluate -> report)
"""

import numpy as np
import pandas as pd
import pytest

from dmt.scenario.equity import (
    generate_returns,
    MeanModel,
    MomentumModel,
    VolatilityModel,
    TICKERS,
)
from dmt.metrics.finance import (
    directional_accuracy,
    sharpe_ratio,
    max_drawdown,
    var_95,
    compute_finance_metrics,
)
from dmt.evaluate import evaluate, EQUITY_FORECAST


# ── Synthetic Data Properties ────────────────────────────────────────────────


class TestSyntheticData:
    """Verify the generated returns exhibit stylised facts."""

    @pytest.fixture
    def observations(self):
        return generate_returns()

    def test_shape(self, observations):
        """4 tickers x 1000 days = 4000 rows."""
        assert len(observations) == 4 * 1000
        assert set(observations.columns) == {
            "ticker", "date", "return", "log_return",
            "cumulative_return", "regime",
        }

    def test_tickers_present(self, observations):
        assert set(observations["ticker"].unique()) == set(TICKERS.keys())

    def test_fat_tails(self, observations):
        """Excess kurtosis should be > 0 (heavier than Gaussian)."""
        for ticker in observations["ticker"].unique():
            returns = observations[observations["ticker"] == ticker]["return"].values
            n = len(returns)
            mean = np.mean(returns)
            m2 = np.mean((returns - mean) ** 2)
            m4 = np.mean((returns - mean) ** 4)
            kurtosis = m4 / (m2 ** 2) if m2 > 0 else 0
            # Gaussian kurtosis = 3; excess kurtosis > 0 means fat tails
            assert kurtosis > 3, (
                f"{ticker}: kurtosis {kurtosis:.2f} should exceed 3 (Gaussian)"
            )

    def test_volatility_clustering(self, observations):
        """Squared returns should be autocorrelated (lag-1 > 0)."""
        for ticker in observations["ticker"].unique():
            returns = observations[observations["ticker"] == ticker]["return"].values
            sq = returns ** 2
            # Lag-1 autocorrelation of squared returns
            acf1 = np.corrcoef(sq[:-1], sq[1:])[0, 1]
            assert acf1 > 0.0, (
                f"{ticker}: squared-return autocorrelation {acf1:.3f} "
                f"should be positive (volatility clustering)"
            )

    def test_returns_roughly_uncorrelated(self, observations):
        """Raw returns should have weak autocorrelation."""
        for ticker in observations["ticker"].unique():
            returns = observations[observations["ticker"] == ticker]["return"].values
            acf1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            # Not a hard statistical test — just a sanity check
            assert abs(acf1) < 0.15, (
                f"{ticker}: return autocorrelation {acf1:.3f} is too large"
            )

    def test_regimes_present(self, observations):
        """Each ticker should have at least bull regime."""
        for ticker in observations["ticker"].unique():
            regimes = observations[observations["ticker"] == ticker]["regime"].unique()
            assert "bull" in regimes


# ── Model Protocol ───────────────────────────────────────────────────────────


class TestModelProtocol:
    """Every model must have .name and .predict() -> DataFrame."""

    @pytest.fixture
    def observations(self):
        return generate_returns(n_days=100)

    @pytest.fixture(params=[MeanModel, MomentumModel, VolatilityModel])
    def model(self, request):
        return request.param()

    def test_has_name(self, model):
        assert isinstance(model.name, str)
        assert len(model.name) > 0

    def test_predict_returns_dataframe(self, model, observations):
        result = model.predict(observations)
        assert isinstance(result, pd.DataFrame)

    def test_predict_has_required_columns(self, model, observations):
        result = model.predict(observations)
        required = {"ticker", "date", "predicted_return", "regime"}
        assert required.issubset(set(result.columns)), (
            f"Missing columns: {required - set(result.columns)}"
        )

    def test_predict_same_length(self, model, observations):
        result = model.predict(observations)
        assert len(result) == len(observations)


# ── Finance Metrics ──────────────────────────────────────────────────────────


class TestFinanceMetrics:
    """Verify metric functions on known inputs."""

    def test_directional_accuracy_perfect(self):
        obs = np.array([1.0, -1.0, 1.0, -1.0])
        pred = np.array([0.5, -0.3, 2.0, -0.1])
        assert directional_accuracy(obs, pred) == 1.0

    def test_directional_accuracy_zero(self):
        obs = np.array([1.0, -1.0, 1.0, -1.0])
        pred = np.array([-0.5, 0.3, -2.0, 0.1])
        assert directional_accuracy(obs, pred) == 0.0

    def test_directional_accuracy_half(self):
        obs = np.array([1.0, -1.0, 1.0, -1.0])
        pred = np.array([0.5, 0.3, -2.0, -0.1])  # match on 0,3; mismatch on 1,2
        assert directional_accuracy(obs, pred) == 0.5

    def test_sharpe_ratio_positive(self):
        # Constant positive returns => infinite Sharpe? No, std > 0 in practice
        returns = np.array([0.01, 0.02, 0.01, 0.03, 0.01])
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_sharpe_ratio_zero_returns(self):
        returns = np.zeros(10)
        assert sharpe_ratio(returns) == 0.0

    def test_max_drawdown_no_loss(self):
        # Monotonically increasing cumulative returns => no drawdown
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        assert max_drawdown(returns) == 0.0

    def test_max_drawdown_known(self):
        # Up 3, then down 5 => drawdown of 5
        returns = np.array([1.0, 1.0, 1.0, -2.0, -2.0, -1.0])
        dd = max_drawdown(returns)
        assert dd == pytest.approx(5.0)

    def test_var_95_gaussian(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 1, size=10000)
        v = var_95(returns)
        # 5th percentile of N(0,1) is about -1.645
        assert -2.0 < v < -1.3


# ── End-to-End Pipeline ──────────────────────────────────────────────────────


class TestEndToEnd:
    """Full pipeline: generate -> models -> evaluate -> report."""

    def test_evaluate_produces_report(self, tmp_path):
        obs = generate_returns(n_days=200)

        models = [MeanModel(), MomentumModel(), VolatilityModel()]

        report_path = evaluate(
            models=models,
            observations=obs,
            scenario=EQUITY_FORECAST,
            output_dir=tmp_path / "equity_report",
            title="Equity Return Forecast Comparison",
        )

        assert report_path.exists()
        report = report_path.read_text()

        # Standard sections
        assert "# Equity Return Forecast Comparison" in report
        assert "## Abstract" in report
        assert "## Introduction" in report
        assert "## Methods" in report
        assert "## Results" in report
        assert "## Discussion" in report
        assert "## Conclusion" in report

        # All models mentioned
        assert "HistoricalMean" in report
        assert "Momentum" in report
        assert "VolRegime" in report

    def test_stratified_csvs(self, tmp_path):
        obs = generate_returns(n_days=200)
        models = [MeanModel(), MomentumModel(), VolatilityModel()]

        output_dir = tmp_path / "equity_report"
        evaluate(
            models=models,
            observations=obs,
            scenario=EQUITY_FORECAST,
            output_dir=output_dir,
            title="Equity Report",
        )

        assert (output_dir / "results.csv").exists()
        assert (output_dir / "results_by_ticker.csv").exists()
        assert (output_dir / "results_by_regime.csv").exists()

    def test_finance_metrics_on_scenario(self):
        """Compute finance metrics alongside standard evaluation."""
        obs = generate_returns(n_days=500)
        model = MeanModel()
        predictions = model.predict(obs)

        merged = obs.merge(predictions, on=["ticker", "date", "regime"])
        for ticker in obs["ticker"].unique():
            tk = merged[merged["ticker"] == ticker]
            metrics = compute_finance_metrics(
                tk["return"].values,
                tk["predicted_return"].values,
            )
            assert "rmse" in metrics
            assert "directional_accuracy" in metrics
            assert 0 <= metrics["directional_accuracy"] <= 1
            assert metrics["rmse"] >= 0
