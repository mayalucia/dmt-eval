"""Model adapter infrastructure.

An adapter is any object that satisfies a ``typing.Protocol`` declaring the
measurements a validation needs.  The ``@adapter`` decorator is syntactic
sugar that registers an adapter class in a global registry.
"""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class WeatherAdapter(Protocol):
    """Contract for weather model adapters.

    Any object with a ``.predict(observations) -> DataFrame`` method
    and a ``.name`` attribute satisfies this protocol.
    """
    name: str

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Return predictions aligned with the observation DataFrame.

        The returned DataFrame must have columns:
        city, day, predicted, season.
        """
        ...


def adapt(model) -> WeatherAdapter:
    """Validate that *model* satisfies the WeatherAdapter protocol.

    Our toy models already satisfy it by construction — their ``.predict()``
    and ``.name`` attributes match.  For real models you would write a wrapper.
    """
    if not isinstance(model, WeatherAdapter):
        raise TypeError(
            f"{type(model).__name__} does not satisfy WeatherAdapter: "
            f"needs .name and .predict(observations) -> DataFrame"
        )
    return model
