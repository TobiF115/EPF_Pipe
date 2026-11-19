"""One-layer LSTM using price lags, target-day EXAA, and load forecasts."""

from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

import pandas as pd

from pipeline.delivery_index import MTUS_PER_DAY
from pipeline.lear.lear_model import (
    build_load_features,
    build_price_features,
    build_y_matrix,
)
from pipeline.mlp.LSTM_exaa_enriched import (
    LSTMConfig,
    _rolling_lstm_forecast,
)
from pipeline.mlp.NN_exaa_enriched import (
    _skip_connection_mask_exaa,
)


def build_price_load_inputs(
    prices: pd.DataFrame,
    exaa: pd.DataFrame,
    load: pd.DataFrame,
    *,
    feature_start_date: date | datetime | pd.Timestamp | str,
    target_date: date | datetime | pd.Timestamp | str,
    timezone: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build EPEX-lag, EXAA, and load inputs without weather features."""
    if "price_da" not in prices.columns:
        raise ValueError("prices must contain a 'price_da' column.")
    if "price_exaa" not in exaa.columns:
        raise ValueError("exaa must contain a 'price_exaa' column.")
    if "load_fc" not in load.columns:
        raise ValueError("load must contain a 'load_fc' column.")
    start = pd.Timestamp(feature_start_date)
    end = pd.Timestamp(target_date)
    start = (
        start.tz_localize(timezone)
        if start.tzinfo is None
        else start.tz_convert(timezone)
    )
    end = (
        end.tz_localize(timezone)
        if end.tzinfo is None
        else end.tz_convert(timezone)
    )
    daily_index = pd.date_range(start, end, freq="D")
    features = build_price_features(
        prices,
        df_prices_exaa_15=exaa,
        exaa_vector=True,
        daily_index=daily_index,
    ).reindex(daily_index)
    load_features = build_load_features(load).reindex(daily_index)
    features = features.join(load_features, how="left")
    target = build_y_matrix(prices, features.index)
    return features, target


def rolling_lstm_exaa_enriched_price_load_forecast(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    forecast_days: Sequence[object],
    config: LSTMConfig = LSTMConfig(),
) -> pd.DataFrame:
    """Forecast using EPEX lags, target-day EXAA, and target-day load only."""
    expected_columns = {
        f"price_d{lag}_mtu_{mtu:02d}"
        for lag in (1, 2, 7)
        for mtu in range(MTUS_PER_DAY)
    }
    expected_columns.update(
        f"exaa_d0_mtu_{mtu:02d}" for mtu in range(MTUS_PER_DAY)
    )
    expected_columns.update(
        f"load_d0_mtu_{mtu:02d}" for mtu in range(MTUS_PER_DAY)
    )
    if set(X.columns) != expected_columns:
        missing = sorted(expected_columns.difference(X.columns))
        unexpected = sorted(set(X.columns).difference(expected_columns))
        raise ValueError(
            "Price/load LSTM inputs must contain exactly the 1-, 2-, and "
            "7-day EPEX lags, target-day EXAA prices, and target-day load; "
            f"missing={missing}, "
            f"unexpected={unexpected}."
        )

    exaa_columns = [
        f"exaa_d0_mtu_{mtu:02d}" for mtu in range(MTUS_PER_DAY)
    ]
    load_columns = [
        f"load_d0_mtu_{mtu:02d}" for mtu in range(MTUS_PER_DAY)
    ]
    price_columns = [
        f"price_d{lag}_mtu_{mtu:02d}"
        for lag in (1, 2, 7)
        for mtu in range(MTUS_PER_DAY)
    ]
    hidden_columns = [*exaa_columns, *load_columns]
    skip_columns = [*price_columns, *exaa_columns]
    return _rolling_lstm_forecast(
        X,
        Y,
        forecast_days,
        config,
        hidden_columns=hidden_columns,
        skip_columns=skip_columns,
        skip_mask=_skip_connection_mask_exaa(skip_columns),
        require_seven_day_average=False,
    )
