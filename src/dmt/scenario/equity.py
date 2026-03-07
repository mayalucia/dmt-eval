"""Synthetic equity returns and baseline forecasting models.

The ground truth is a GARCH(1,1) process with Student-t innovations,
reproducing the stylised facts of real financial time series:
uncorrelated returns, volatility clustering, and fat tails.

Three models of increasing sophistication attempt to forecast returns.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


# ── Ticker parameters ────────────────────────────────────────────────────────

TICKERS = {
    "ALPHA": {
        "mu": 0.0004,    # daily drift (~10% annual)
        "omega": 1e-6,   # base variance
        "alpha": 0.05,   # ARCH: shock impact
        "beta": 0.90,    # GARCH: variance persistence
        "nu": 7,         # Student-t degrees of freedom
    },
    "BETA": {
        "mu": 0.0001,    # low drift (~2.5% annual)
        "omega": 5e-6,   # higher base variance
        "alpha": 0.10,   # more reactive to shocks
        "beta": 0.85,    # high persistence
        "nu": 6,         # heavier tails
    },
    "GAMMA": {
        "mu": 0.0008,    # high drift (~20% annual) — but with regime switches
        "omega": 2e-6,
        "alpha": 0.07,
        "beta": 0.88,
        "nu": 8,         # lighter tails
        "regime_switch": True,
    },
    "DELTA": {
        "mu": 0.0002,    # modest drift
        "omega": 3e-6,   # elevated base variance
        "alpha": 0.12,   # very reactive
        "beta": 0.83,    # moderate persistence
        "nu": 4,         # very heavy tails — crisis-prone
    },
}


def _garch_simulate(
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    nu: float,
    n: int,
    rng: np.random.Generator,
    regime_switch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a GARCH(1,1) process with Student-t innovations.

    Returns (returns, conditional_variances).
    """
    # Unconditional variance as starting point
    sigma2_uncond = omega / (1 - alpha - beta)
    sigma2 = np.empty(n)
    returns = np.empty(n)

    sigma2[0] = sigma2_uncond
    eps = rng.standard_t(df=nu, size=n)

    # For regime-switching ticker: flip drift sign mid-series
    mu_t = np.full(n, mu)
    if regime_switch:
        # Bear regime in middle third
        third = n // 3
        mu_t[third:2 * third] = -abs(mu) * 0.5

    returns[0] = mu_t[0] + np.sqrt(sigma2[0]) * eps[0]

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = mu_t[t] + np.sqrt(sigma2[t]) * eps[t]

    return returns, sigma2


def _assign_regimes(
    cumulative_returns: np.ndarray,
    conditional_variance: np.ndarray,
) -> np.ndarray:
    """Label each day as bull, bear, or crisis.

    - crisis: conditional volatility > 2x median
    - bear: cumulative return declining (20-day rolling negative)
    - bull: everything else
    """
    n = len(cumulative_returns)
    regimes = np.full(n, "bull", dtype=object)

    vol = np.sqrt(conditional_variance)
    median_vol = np.median(vol)

    # Rolling 20-day return
    window = min(20, n)
    for t in range(window, n):
        rolling_ret = cumulative_returns[t] - cumulative_returns[t - window]
        if vol[t] > 2 * median_vol:
            regimes[t] = "crisis"
        elif rolling_ret < 0:
            regimes[t] = "bear"

    return regimes


def generate_returns(
    tickers: dict | None = None,
    n_days: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily equity returns.

    Uses a GARCH(1,1) process with Student-t innovations for each ticker.
    Returns a DataFrame with columns:
        ticker, date, return, log_return, cumulative_return, regime
    """
    rng = np.random.default_rng(seed)
    tickers = tickers or TICKERS
    rows = []

    for ticker, params in tickers.items():
        returns, sigma2 = _garch_simulate(
            mu=params["mu"],
            omega=params["omega"],
            alpha=params["alpha"],
            beta=params["beta"],
            nu=params["nu"],
            n=n_days,
            rng=rng,
            regime_switch=params.get("regime_switch", False),
        )

        cumulative = np.cumsum(returns)
        regimes = _assign_regimes(cumulative, sigma2)

        for d in range(n_days):
            rows.append({
                "ticker": ticker,
                "date": d,
                "return": float(returns[d]),
                "log_return": float(returns[d]),  # for small returns, approx equal
                "cumulative_return": float(cumulative[d]),
                "regime": regimes[d],
            })

    return pd.DataFrame(rows)

# ── Baseline Models ──────────────────────────────────────────────────────────


@dataclass
class MeanModel:
    """Predict tomorrow's return as the expanding-window historical mean.

    The efficient market baseline: if returns are unpredictable,
    the best forecast is the long-run average.
    """
    name: str = "HistoricalMean"

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        predictions = []
        for ticker in observations["ticker"].unique():
            tk = observations[observations["ticker"] == ticker].sort_values("date")
            returns = tk["return"].values
            for i, row in enumerate(tk.itertuples()):
                if i == 0:
                    pred = 0.0  # no history yet
                else:
                    pred = float(np.mean(returns[:i]))
                predictions.append({
                    "ticker": ticker,
                    "date": row.date,
                    "predicted_return": pred,
                    "regime": row.regime,
                })
        return pd.DataFrame(predictions)


@dataclass
class MomentumModel:
    """Predict using exponentially weighted moving average of recent returns.

    Captures short-term trend persistence (momentum effect).
    """
    name: str = "Momentum"
    span: int = 20  # EWMA span in days

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        decay = 2.0 / (self.span + 1)  # EWMA decay factor
        predictions = []
        for ticker in observations["ticker"].unique():
            tk = observations[observations["ticker"] == ticker].sort_values("date")
            returns = tk["return"].values
            ewma = 0.0
            for i, row in enumerate(tk.itertuples()):
                predictions.append({
                    "ticker": ticker,
                    "date": row.date,
                    "predicted_return": float(ewma),
                    "regime": row.regime,
                })
                ewma = decay * returns[i] + (1 - decay) * ewma
        return pd.DataFrame(predictions)


@dataclass
class VolatilityModel:
    """Predict return direction using a volatility regime signal.

    When realised vol > moving average vol: predict mean reversion.
    When realised vol < moving average vol: predict drift continuation.
    """
    name: str = "VolRegime"
    vol_window: int = 20    # realised vol window
    drift_window: int = 60  # drift estimation window

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        predictions = []
        for ticker in observations["ticker"].unique():
            tk = observations[observations["ticker"] == ticker].sort_values("date")
            returns = tk["return"].values
            n = len(returns)
            for i, row in enumerate(tk.itertuples()):
                if i < self.drift_window:
                    pred = 0.0  # not enough history
                else:
                    recent_vol = float(np.std(returns[i - self.vol_window:i]))
                    longer_vol = float(np.std(returns[i - self.drift_window:i]))
                    drift = float(np.mean(returns[i - self.drift_window:i]))

                    if recent_vol > longer_vol:
                        # High vol regime: predict mean reversion (toward zero)
                        pred = drift * 0.3
                    else:
                        # Low vol regime: predict drift continuation
                        pred = drift * 1.2

                predictions.append({
                    "ticker": ticker,
                    "date": row.date,
                    "predicted_return": pred,
                    "regime": row.regime,
                })
        return pd.DataFrame(predictions)
