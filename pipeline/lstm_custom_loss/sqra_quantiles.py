"""Probabilistic forecasts for the custom-loss LSTMs with SQRA.

Smoothing Quantile Regression Averaging (Uniejewski, "Smoothing quantile
regression averaging: A new approach to probabilistic forecasting of
electricity prices", J. Commodity Markets 39, 2025; arXiv:2302.00411):

* Regressors: several point forecasts of the same price. In the paper these
  are five forecasts of one expert model that differ only in the
  variance-stabilising transformation; here they are the point forecasts of
  the three LSTM variants of this package (pretrained only, rolling window,
  adaptive), which share the network and differ only in the daily update.
* For every quantile q, one linear quantile regression of the realised price
  on these forecasts (with intercept) is fitted on a rolling calibration
  window, with the check loss smoothed by a Gaussian kernel::

      beta_q = argmin sum_i [ H * phi(e_i / H) + (q - Phi(-e_i / H)) * e_i ],
      e_i = P_i - X_i beta

  The bandwidth H follows the rule of thumb of the paper
  (H = 1.06 * min(std, IQR) * N^(-1/5) of the QR residuals); the estimator
  is ``remodels.qra.SQRA``, the same one as the LEAR SQRA in ``pipeline.sqra``.
* The quantiles are estimated separately and sorted afterwards (paper:
  "sorted independently for each day and hour") to remove crossings.

Calibration defaults follow the operational LEAR SQRA: 60 days, one model per
quantile pooled over all 96 MTUs (``mtu_specific=False``); the paper uses
182 days and hourly data. Only out-of-sample point forecasts may enter the
calibration window, i.e. the forecasts each LSTM produced for its target days
(not in-sample fits of the pretraining period).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.delivery_index import (
    delivery_dates,
    read_forecast_csv,
    validate_delivery_index,
)
from pipeline.sqra.sqra_model import (
    DEFAULT_QUANTILES,
    aggregate_pinball_score,
    empirical_coverage,
    generate_sqra_forecast,
)

MEMBER_MODELS: tuple[str, ...] = ("pretrained_only", "rolling_window", "adaptive")
DEFAULT_TRAIN_DAYS = 60
MIN_CALIBRATION_DAYS = 14

__all__ = [
    "DEFAULT_QUANTILES",
    "DEFAULT_TRAIN_DAYS",
    "MEMBER_MODELS",
    "build_panel",
    "evaluate_quantiles",
    "load_point_forecasts",
    "sqra_quantile_forecast",
]


def build_panel(
    point_forecasts: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """Align the members' point forecasts into one SQRA panel.

    ``point_forecasts`` maps a member name to a frame with the canonical
    (delivery_date, mtu) index and columns ``y_pred`` and ``y_true``. The panel
    keeps the delivery keys present in every member; ``y_true`` is taken from
    whichever member knows it (NaN for days whose prices are not yet known).
    """
    if len(point_forecasts) < 1:
        raise ValueError("SQRA needs at least one point forecast.")
    frames = {}
    for name, frame in point_forecasts.items():
        validate_delivery_index(frame.index)
        missing = {"y_pred", "y_true"} - set(frame.columns)
        if missing:
            raise ValueError(f"Point forecast {name!r} lacks columns {sorted(missing)}.")
        frames[name] = frame
    common = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index)
    common = common.sort_values()
    if common.empty:
        raise ValueError("The point forecasts share no delivery keys.")

    feature_cols = [f"y_pred_{name}" for name in frames]
    panel = pd.DataFrame(index=common)
    truth = pd.Series(np.nan, index=common)
    for name, column in zip(frames, feature_cols):
        aligned = frames[name].loc[common]
        panel[column] = aligned["y_pred"].to_numpy(dtype=float)
        known = aligned["y_true"].astype(float)
        conflict = truth.notna() & known.notna() & ~np.isclose(truth, known)
        if conflict.any():
            raise ValueError(f"y_true of {name!r} differs from the other members.")
        truth = truth.fillna(known)
    panel["y_true"] = truth
    return panel, feature_cols


def load_point_forecasts(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    """Read the members' forecast histories (canonical forecast CSVs)."""
    return {
        name: read_forecast_csv(Path(path), required_columns=("y_pred", "y_true"))[
            ["y_pred", "y_true"]
        ]
        for name, path in paths.items()
    }


def sqra_quantile_forecast(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    target_dates: Sequence[date] | date,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    train_days: int = DEFAULT_TRAIN_DAYS,
    mtu_specific: bool = False,
    min_calibration_days: int = MIN_CALIBRATION_DAYS,
) -> pd.DataFrame:
    """SQRA quantiles for each target day from the ``train_days`` days before it.

    Returns one row per (delivery_date, mtu) with columns ``q0.100`` ... and
    ``y_true``; quantiles are sorted row-wise. Raises if a target day has fewer
    than ``min_calibration_days`` complete calibration days.
    """
    if isinstance(target_dates, date):
        target_dates = [target_dates]
    if not 0 < min(quantiles) <= max(quantiles) < 1:
        raise ValueError("Quantiles must lie strictly between 0 and 1.")
    days = delivery_dates(panel.index)
    complete = (
        panel[[*feature_cols, "y_true"]].notna().all(axis=1).groupby(days).all()
    )
    for target in target_dates:
        target_day = pd.Timestamp(target)
        if not (days == target_day).any():
            raise ValueError(f"No point forecasts for the target day {target}.")
        window = complete[
            (complete.index >= target_day - pd.Timedelta(days=train_days))
            & (complete.index < target_day)
        ]
        if int(window.sum()) < min_calibration_days:
            raise ValueError(
                f"Only {int(window.sum())} complete calibration days before {target} "
                f"(< {min_calibration_days})."
            )
    forecast, _ = generate_sqra_forecast(
        df=panel,
        forecast_days=[pd.Timestamp(d) for d in target_dates],
        train_days=train_days,
        quantiles=list(quantiles),
        feature_cols=list(feature_cols),
        mtu_specific=mtu_specific,
    )
    validate_delivery_index(forecast.index, require_complete_days=True)
    return forecast


def evaluate_quantiles(
    forecast: pd.DataFrame, quantiles: Sequence[float] = DEFAULT_QUANTILES
) -> dict:
    """Average pinball score and empirical coverage of the central intervals."""
    scored = forecast.dropna(subset=["y_true"])
    per_quantile, average = aggregate_pinball_score(scored, list(quantiles))
    result = {
        "days": int(delivery_dates(scored.index).nunique()),
        "pinball_average": average,
        "pinball_per_quantile": {f"{q:.3f}": v for q, v in per_quantile.items()},
    }
    ordered = sorted(quantiles)
    for low, high in zip(ordered, reversed(ordered)):
        if low >= high:
            break
        result[f"coverage_{round((high - low) * 100)}"] = empirical_coverage(scored, low, high)
    return result
