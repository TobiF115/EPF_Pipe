"""Model 2: custom-loss LSTM with daily rolling-window fine-tuning.

After each delivery day ``d`` the model is fine-tuned (custom loss,
``update_epochs`` Adam steps at ``update_learning_rate``) on the complete days
``d - window_days + 1 .. d``. The update is skipped if ``d`` itself is
incomplete or fewer than ``min_window_days`` complete days are available.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from pipeline.lstm_custom_loss.common import (
    MODEL_ROOT,
    CustomLossLSTMConfig,
    ModelState,
    ModelStore,
    advance,
    complete_days,
    forecast_state,
    pretrain_state,
    train_from,
)

MODEL_NAME = "lstm_cl_rolling"
MODEL_DIR = MODEL_ROOT / "rolling_window"


def rolling_update_state(
    state: ModelState,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    covers_through: date,
    config: CustomLossLSTMConfig = CustomLossLSTMConfig(),
    *,
    metadata: dict | None = None,
) -> ModelState:
    first = covers_through - timedelta(days=config.window_days - 1)
    rows = complete_days(X, Y, first, covers_through)
    used = [pd.Timestamp(i).date() for i in rows]
    info = {
        "kind": "updated",
        "window_first_day": first.isoformat(),
        "window_last_day": covers_through.isoformat(),
        "window_days_used": [d.isoformat() for d in used],
        **(metadata or {}),
    }
    if covers_through not in used:
        info.update(kind="skipped", reason=f"{covers_through} itself has incomplete inputs or prices")
        return state.with_parameters(state.parameters, covers_through, info)
    if len(rows) < config.min_window_days:
        info.update(
            kind="skipped",
            reason=f"only {len(rows)} complete days in window (< {config.min_window_days})",
        )
        return state.with_parameters(state.parameters, covers_through, info)

    parameters, before, after = train_from(state, rows, X, Y, covers_through, config)
    info["window_loss_before"] = before
    info["window_loss_after"] = after
    return state.with_parameters(parameters, covers_through, info)


def advance_state(
    store: ModelStore,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_date: date,
    config: CustomLossLSTMConfig = CustomLossLSTMConfig(),
    *,
    day_metadata=None,
) -> tuple[ModelState, list[dict]]:
    return advance(store, X, Y, target_date, config, rolling_update_state, day_metadata=day_metadata)


def first_required_feature_day(
    store: ModelStore,
    target_date: date,
    config: CustomLossLSTMConfig = CustomLossLSTMConfig(),
) -> date:
    state = store.load_latest(not_after=target_date - timedelta(days=1))
    first_update = state.covers_through + timedelta(days=1)
    return min(
        first_update - timedelta(days=config.window_days - 1),
        target_date - timedelta(days=config.bias_window_days),
    )


__all__ = [
    "MODEL_DIR",
    "MODEL_NAME",
    "advance_state",
    "first_required_feature_day",
    "forecast_state",
    "pretrain_state",
    "rolling_update_state",
]
