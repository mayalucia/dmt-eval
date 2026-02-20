"""Regression test: weather scenario still works after evaluate() refactor."""

from dmt.scenario.weather import (
    generate_observations,
    PersistenceModel,
    ClimatologyModel,
    NoisyRegressionModel,
)
from dmt.evaluate import evaluate, WEATHER


def test_weather_still_works(tmp_path):
    """Weather scenario produces correct report with refactored evaluator."""
    obs = generate_observations(n_days=365, seed=42)

    report_path = evaluate(
        models=[PersistenceModel(), ClimatologyModel(),
                NoisyRegressionModel(alpha=0.7, noise_std=0.5)],
        observations=obs,
        scenario=WEATHER,
        reference_model=ClimatologyModel(),
        output_dir=tmp_path / "weather_regression",
        title="Weather Regression Test",
    )

    report_text = report_path.read_text()

    assert "## Abstract" in report_text
    assert "## Results" in report_text
    assert "Persistence" in report_text
    assert "Climatology" in report_text
    assert "NoisyRegression" in report_text
    assert "weather prediction" in report_text
