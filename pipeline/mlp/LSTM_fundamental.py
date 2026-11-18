"""One-layer LSTM using C=5 fundamental features without EXAA prices."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from pipeline.delivery_index import MTUS_PER_DAY
from pipeline.mlp.LSTM_exaa_enriched import (
    LSTMConfig,
    SEVEN_DAY_AVERAGE_PRICE_COLUMN,
    _rolling_lstm_forecast,
)
from pipeline.mlp.mlp import _column_groups, _skip_connection_mask


def _fundamental_skip_connection_mask(
    skip_columns: Sequence[str],
) -> np.ndarray:
    """Include EPEX lag skips and connect the weekly mean to every output."""
    average_skip = np.ones((1, MTUS_PER_DAY), dtype=bool)
    return np.vstack((_skip_connection_mask(skip_columns), average_skip))


def rolling_lstm_fundamental_forecast(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    forecast_days: Sequence[object],
    config: LSTMConfig = LSTMConfig(),
) -> pd.DataFrame:
    """Forecast with the EXAA LSTM architecture and only fundamental inputs."""
    if not isinstance(config, LSTMConfig):
        raise TypeError("config must be an LSTMConfig.")

    hidden_columns, skip_columns = _column_groups(X.columns.tolist())
    if SEVEN_DAY_AVERAGE_PRICE_COLUMN not in hidden_columns:
        raise ValueError(
            "LSTM features must include the seven-day average DA price."
        )

    skip_mask = _fundamental_skip_connection_mask(skip_columns)
    skip_columns.append(SEVEN_DAY_AVERAGE_PRICE_COLUMN)
    return _rolling_lstm_forecast(
        X,
        Y,
        forecast_days,
        config,
        hidden_columns=hidden_columns,
        skip_columns=skip_columns,
        skip_mask=skip_mask,
    )
