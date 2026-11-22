"""Model 3: custom-loss LSTM with validated single-day updates.

Adaptive online learning after Salihoglu, Ahmed & Asadi (arXiv:2510.16898,
Section 4, Algorithm 1), with the same one-layer LSTM, loss and update
settings as model 2. After each delivery day ``d``:

1. Train a candidate theta' on day ``d`` alone, starting from theta*.
2. Evaluate the RMSE in EUR/MWh of theta* and theta' on the validation days
   ``d - validation_days .. d - 1`` (default: the 6 previous days).
3. Accept theta' only if ``RMSE_V(theta*) - RMSE_V(theta') > acceptance_margin``
   (delta = 0); otherwise keep theta*.

The paper compares the training loss on V; here the acceptance uses the RMSE,
the evaluation criterion of this project.

Skipped if ``d`` is incomplete or fewer than ``min_validation_days`` complete
validation days exist. Each day's checkpoint records accepted / rejected /
skipped and both validation RMSEs.
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
    rmse_on,
    train_from,
)

MODEL_NAME = "lstm_cl_adaptive"
MODEL_DIR = MODEL_ROOT / "adaptive"


def adaptive_update_state(
    state: ModelState,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    covers_through: date,
    config: CustomLossLSTMConfig = CustomLossLSTMConfig(),
    *,
    metadata: dict | None = None,
) -> ModelState:
    new_rows = complete_days(X, Y, covers_through, covers_through)
    first_val = covers_through - timedelta(days=config.validation_days)
    val_rows = complete_days(X, Y, first_val, covers_through - timedelta(days=1))
    info = {
        "kind": "skipped",
        "update_day": covers_through.isoformat(),
        "validation_first_day": first_val.isoformat(),
        "validation_days_used": [pd.Timestamp(i).date().isoformat() for i in val_rows],
        **(metadata or {}),
    }
    if not len(new_rows):
        info["reason"] = f"{covers_through} itself has incomplete inputs or prices"
        return state.with_parameters(state.parameters, covers_through, info)
    if len(val_rows) < config.min_validation_days:
        info["reason"] = (
            f"only {len(val_rows)} complete validation days (< {config.min_validation_days})"
        )
        return state.with_parameters(state.parameters, covers_through, info)

    candidate, before, after = train_from(state, new_rows, X, Y, covers_through, config)
    rmse_current = rmse_on(state.parameters, state, val_rows, X, Y, config)
    rmse_candidate = rmse_on(candidate, state, val_rows, X, Y, config)
    accepted = rmse_current - rmse_candidate > config.acceptance_margin
    info.update(
        kind="accepted" if accepted else "rejected",
        new_day_loss_before=before,
        new_day_loss_after=after,
        validation_rmse_current=rmse_current,
        validation_rmse_candidate=rmse_candidate,
    )
    return state.with_parameters(candidate if accepted else state.parameters, covers_through, info)


def advance_state(
    store: ModelStore,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_date: date,
    config: CustomLossLSTMConfig = CustomLossLSTMConfig(),
    *,
    day_metadata=None,
) -> tuple[ModelState, list[dict]]:
    return advance(store, X, Y, target_date, config, adaptive_update_state, day_metadata=day_metadata)


def first_required_feature_day(
    store: ModelStore,
    target_date: date,
    config: CustomLossLSTMConfig = CustomLossLSTMConfig(),
) -> date:
    state = store.load_latest(not_after=target_date - timedelta(days=1))
    first_update = state.covers_through + timedelta(days=1)
    return min(
        first_update - timedelta(days=config.validation_days),
        target_date - timedelta(days=config.bias_window_days),
    )


__all__ = [
    "MODEL_DIR",
    "MODEL_NAME",
    "adaptive_update_state",
    "advance_state",
    "first_required_feature_day",
    "forecast_state",
    "pretrain_state",
]
