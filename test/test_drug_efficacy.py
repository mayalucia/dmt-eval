"""End-to-end test: drug efficacy scenario using the domain-agnostic evaluator."""

from pathlib import Path

from dmt.scenario.drug_efficacy import (
    generate_observations,
    LinearModel,
    SigmoidModel,
    CalibratedModel,
)
from dmt.evaluate import evaluate, DRUG_EFFICACY


def test_drug_efficacy_scenario(tmp_path):
    """Full pipeline: dose-response data -> models -> evaluate -> report."""
    obs = generate_observations()
    assert "compound" in obs.columns
    assert "dose" in obs.columns
    assert "efficacy" in obs.columns

    linear = LinearModel()
    sigmoid = SigmoidModel()
    calibrated = CalibratedModel()

    report_dir = tmp_path / "drug_report"
    report_path = evaluate(
        models=[linear, sigmoid, calibrated],
        observations=obs,
        scenario=DRUG_EFFICACY,
        reference_model=linear,
        output_dir=report_dir,
        title="Drug Efficacy Model Comparison",
    )

    assert report_path.exists()
    report_text = report_path.read_text()

    # Report structure
    assert "# Drug Efficacy Model Comparison" in report_text
    assert "## Abstract" in report_text
    assert "## Methods" in report_text
    assert "## Results" in report_text
    assert "## Discussion" in report_text

    # All models present
    assert "Linear" in report_text
    assert "Sigmoid(miscalibrated)" in report_text
    assert "Calibrated" in report_text

    # Domain-specific terms from the scenario descriptor
    assert "drug efficacy prediction" in report_text
    assert "pharmaceutical compounds" in report_text


def test_calibrated_beats_linear():
    """The calibrated Hill model should crush the linear model."""
    obs = generate_observations()

    from dmt.evaluate import _rmse
    linear = LinearModel()
    calibrated = CalibratedModel()

    l_pred = linear.predict(obs)
    c_pred = calibrated.predict(obs)

    l_merged = obs.merge(l_pred, on=["compound", "dose"])
    c_merged = obs.merge(c_pred, on=["compound", "dose"])

    l_rmse = _rmse(l_merged["efficacy"].values, l_merged["predicted"].values)
    c_rmse = _rmse(c_merged["efficacy"].values, c_merged["predicted"].values)

    assert c_rmse < l_rmse, (
        f"Calibrated RMSE ({c_rmse:.2f}) should be less than "
        f"Linear RMSE ({l_rmse:.2f})"
    )
