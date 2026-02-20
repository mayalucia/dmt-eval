"""Synthetic dose-response data and toy pharmacological models.

The ground truth follows Hill equation kinetics.  Three models of
increasing sophistication attempt to predict efficacy at each dose.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ── Compound parameters ──────────────────────────────────────────────────────

COMPOUNDS = {
    "AlphaBlock": {"ic50": 5.0,  "e_max": 95.0, "hill_n": 1.5},
    "BetaCure":   {"ic50": 15.0, "e_max": 85.0, "hill_n": 2.5},
    "GammaLite":  {"ic50": 8.0,  "e_max": 70.0, "hill_n": 1.0},
}


def hill_equation(dose: float, ic50: float, e_max: float, hill_n: float) -> float:
    """Standard Hill equation for dose-response."""
    if dose <= 0:
        return 0.0
    return e_max * (dose ** hill_n) / (dose ** hill_n + ic50 ** hill_n)


def generate_observations(
    compounds: dict | None = None,
    doses: list[float] | None = None,
    n_replicates: int = 5,
    noise_std: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic dose-response observations.

    Returns a DataFrame with columns: compound, dose, efficacy, replicate.
    """
    rng = np.random.default_rng(seed)
    compounds = compounds or COMPOUNDS
    if doses is None:
        doses = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    rows = []
    for compound, params in compounds.items():
        for dose in doses:
            true_effect = hill_equation(dose, **params)
            for rep in range(n_replicates):
                observed = true_effect + rng.normal(0, noise_std)
                observed = max(0.0, min(100.0, observed))  # clamp to [0, 100]
                rows.append({
                    "compound": compound,
                    "dose": dose,
                    "efficacy": float(observed),
                    "replicate": rep,
                })
    return pd.DataFrame(rows)


# ── Toy Models ───────────────────────────────────────────────────────────────

@dataclass
class LinearModel:
    """Assumes efficacy is linear in dose.  The worst possible model."""
    name: str = "Linear"
    slope: float = 0.8
    intercept: float = 5.0

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in observations.iterrows():
            pred = min(100.0, self.intercept + self.slope * row["dose"])
            rows.append({
                "compound": row["compound"],
                "dose": row["dose"],
                "predicted": float(pred),
            })
        return pd.DataFrame(rows)


@dataclass
class SigmoidModel:
    """Hill equation with wrong parameters (trained on different data)."""
    name: str = "Sigmoid(miscalibrated)"
    # Deliberately offset from true values
    ic50_offset: float = 3.0
    e_max_offset: float = -10.0
    hill_n_override: float = 1.0
    compounds: dict = field(default_factory=lambda: COMPOUNDS)

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in observations.iterrows():
            params = self.compounds[row["compound"]]
            pred = hill_equation(
                row["dose"],
                ic50=params["ic50"] + self.ic50_offset,
                e_max=params["e_max"] + self.e_max_offset,
                hill_n=self.hill_n_override,
            )
            rows.append({
                "compound": row["compound"],
                "dose": row["dose"],
                "predicted": float(pred),
            })
        return pd.DataFrame(rows)


@dataclass
class CalibratedModel:
    """Hill equation with near-correct parameters + small noise."""
    name: str = "Calibrated"
    noise_std: float = 1.5
    compounds: dict = field(default_factory=lambda: COMPOUNDS)
    seed: int = 99

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        rows = []
        for _, row in observations.iterrows():
            params = self.compounds[row["compound"]]
            pred = hill_equation(row["dose"], **params) + rng.normal(0, self.noise_std)
            pred = max(0.0, min(100.0, pred))
            rows.append({
                "compound": row["compound"],
                "dose": row["dose"],
                "predicted": float(pred),
            })
        return pd.DataFrame(rows)
