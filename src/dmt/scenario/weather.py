"""Synthetic weather observations and toy prediction models.

The ground truth is a sinusoidal annual cycle with AR(1) weather noise.
Three models of increasing sophistication attempt to predict it.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# ── City parameters ──────────────────────────────────────────────────────────

CITIES = {
    "Zurich":    {"mean": 9.3,  "amplitude": 10.5, "phase": 15,  "noise_std": 3.2},
    "Madrid":    {"mean": 14.5, "amplitude": 10.0, "phase": 20,  "noise_std": 3.5},
    "Stockholm": {"mean": 6.5,  "amplitude": 13.0, "phase": 10,  "noise_std": 3.8},
    "Athens":    {"mean": 18.0, "amplitude": 9.0,  "phase": 25,  "noise_std": 2.8},
    "London":    {"mean": 11.0, "amplitude": 7.5,  "phase": 18,  "noise_std": 2.5},
}


def generate_observations(
    cities: dict | None = None,
    n_days: int = 365,
    ar1_rho: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily temperature observations.

    Returns a DataFrame with columns: city, day, temperature, season.
    """
    rng = np.random.default_rng(seed)
    cities = cities or CITIES
    rows = []
    for city, params in cities.items():
        # Seasonal cycle
        days = np.arange(n_days)
        seasonal = (params["mean"]
                    + params["amplitude"]
                    * np.sin(2 * np.pi * (days - params["phase"]) / 365))
        # AR(1) weather noise
        noise = np.zeros(n_days)
        noise[0] = rng.normal(0, params["noise_std"])
        for t in range(1, n_days):
            noise[t] = (ar1_rho * noise[t - 1]
                        + np.sqrt(1 - ar1_rho**2)
                        * rng.normal(0, params["noise_std"]))
        temperature = seasonal + noise
        season = pd.cut(
            days % 365,
            bins=[0, 90, 181, 273, 365],
            labels=["winter", "spring", "summer", "autumn"],
            right=False,
        )
        for d in days:
            rows.append({
                "city": city,
                "day": int(d),
                "temperature": float(temperature[d]),
                "season": season[d],
            })
    return pd.DataFrame(rows)


# ── Toy Models ───────────────────────────────────────────────────────────────

@dataclass
class PersistenceModel:
    """Tomorrow = today.  The simplest baseline."""
    name: str = "Persistence"

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Predict next-day temperature as current temperature."""
        predictions = []
        for city in observations["city"].unique():
            city_obs = observations[observations["city"] == city].sort_values("day")
            temps = city_obs["temperature"].values
            # Shift: prediction for day d+1 is observation at day d
            pred_temps = np.concatenate([[temps[0]], temps[:-1]])
            for i, row in enumerate(city_obs.itertuples()):
                predictions.append({
                    "city": city,
                    "day": row.day,
                    "predicted": float(pred_temps[i]),
                    "season": row.season,
                })
        return pd.DataFrame(predictions)


@dataclass
class ClimatologyModel:
    """Tomorrow = historical average for this calendar day."""
    name: str = "Climatology"
    cities: dict = field(default_factory=lambda: CITIES)

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Predict temperature as the seasonal climatological value."""
        predictions = []
        for city in observations["city"].unique():
            params = self.cities[city]
            city_obs = observations[observations["city"] == city].sort_values("day")
            for row in city_obs.itertuples():
                clim = (params["mean"]
                        + params["amplitude"]
                        * np.sin(2 * np.pi * (row.day - params["phase"]) / 365))
                predictions.append({
                    "city": city,
                    "day": row.day,
                    "predicted": float(clim),
                    "season": row.season,
                })
        return pd.DataFrame(predictions)


@dataclass
class NoisyRegressionModel:
    """AR(1) blend: alpha * today + (1-alpha) * climatology + noise."""
    name: str = "NoisyRegression"
    alpha: float = 0.6
    noise_std: float = 1.0
    cities: dict = field(default_factory=lambda: CITIES)
    seed: int = 123

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Predict via AR(1) blending of persistence and climatology."""
        rng = np.random.default_rng(self.seed)
        predictions = []
        for city in observations["city"].unique():
            params = self.cities[city]
            city_obs = observations[observations["city"] == city].sort_values("day")
            temps = city_obs["temperature"].values
            for i, row in enumerate(city_obs.itertuples()):
                clim = (params["mean"]
                        + params["amplitude"]
                        * np.sin(2 * np.pi * (row.day - params["phase"]) / 365))
                if i == 0:
                    pred = clim
                else:
                    pred = (self.alpha * temps[i - 1]
                            + (1 - self.alpha) * clim
                            + rng.normal(0, self.noise_std))
                predictions.append({
                    "city": city,
                    "day": row.day,
                    "predicted": float(pred),
                    "season": row.season,
                })
        return pd.DataFrame(predictions)
