"""End-to-end test: generate weather scenario, evaluate models, produce report."""

from pathlib import Path
import shutil

from dmt.scenario.weather import (
    generate_observations,
    PersistenceModel,
    ClimatologyModel,
    NoisyRegressionModel,
)
from dmt.adapter import adapt
from dmt.evaluate import evaluate


def test_weather_scenario(tmp_path):
    """Full pipeline: observations -> models -> adapter -> evaluate -> report."""
    # 1. Generate synthetic observations
    obs = generate_observations(n_days=365, seed=42)
    assert len(obs) == 5 * 365  # 5 cities, 365 days
    assert set(obs.columns) == {"city", "day", "temperature", "season"}

    # 2. Create models
    persistence = PersistenceModel()
    climatology = ClimatologyModel()
    regression = NoisyRegressionModel(alpha=0.7, noise_std=0.5)

    # 3. Verify adapter protocol
    for model in [persistence, climatology, regression]:
        adapted = adapt(model)
        assert adapted.name

    # 4. Run evaluation
    report_dir = tmp_path / "weather_report"
    report_path = evaluate(
        models=[persistence, climatology, regression],
        observations=obs,
        reference_model=climatology,
        output_dir=report_dir,
        title="Weather Prediction Model Comparison",
    )

    # 5. Verify report was generated
    assert report_path.exists()
    report_text = report_path.read_text()

    # 6. Verify report structure
    assert "# Weather Prediction Model Comparison" in report_text
    assert "## Abstract" in report_text
    assert "## Introduction" in report_text
    assert "## Methods" in report_text
    assert "## Results" in report_text
    assert "## Discussion" in report_text
    assert "## Conclusion" in report_text

    # 7. Verify all models appear in results
    assert "Persistence" in report_text
    assert "Climatology" in report_text
    assert "NoisyRegression" in report_text

    # 8. Verify CSV data was saved
    assert (report_dir / "results.csv").exists()
    assert (report_dir / "results_by_city.csv").exists()
    assert (report_dir / "results_by_season.csv").exists()


def test_regression_beats_persistence():
    """The AR(1) blend should outperform naive persistence overall."""
    obs = generate_observations(n_days=365, seed=42)

    from dmt.measurement import compute_metrics
    persistence = PersistenceModel()
    regression = NoisyRegressionModel(alpha=0.7, noise_std=0.5)

    p_metrics = compute_metrics(obs, persistence.predict(obs))
    r_metrics = compute_metrics(obs, regression.predict(obs))

    assert r_metrics["rmse"] < p_metrics["rmse"], (
        f"Regression RMSE ({r_metrics['rmse']:.2f}) should be less than "
        f"Persistence RMSE ({p_metrics['rmse']:.2f})"
    )


def test_climatology_is_unbiased():
    """Climatology predictions should have near-zero bias."""
    obs = generate_observations(n_days=365, seed=42)

    from dmt.measurement import compute_metrics
    climatology = ClimatologyModel()
    metrics = compute_metrics(obs, climatology.predict(obs))

    assert abs(metrics["bias"]) < 0.5, (
        f"Climatology bias ({metrics['bias']:.3f}) should be near zero"
    )
