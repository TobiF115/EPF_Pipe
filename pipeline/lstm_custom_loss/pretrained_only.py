"""Model 1: custom-loss LSTM, pretrained once on ERA5, never updated.

Every forecast uses the pretrained parameters; only the arcsinh bias
correction (residuals of the last ``bias_window_days``) follows recent data.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from pipeline.lstm_custom_loss.common import (
    MODEL_ROOT,
    CustomLossLSTMConfig,
    ModelState,
    ModelStore,
    forecast_state,
    pretrain_state,
)

MODEL_NAME = "lstm_cl_pretrained"
MODEL_DIR = MODEL_ROOT / "pretrained_only"


def advance_state(
    store: ModelStore,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_date: date,
    config: CustomLossLSTMConfig = CustomLossLSTMConfig(),
    **_: object,
) -> tuple[ModelState, list[dict]]:
    """No updates: always the pretrained state (same interface as the online models)."""
    state = store.load_pretrained()
    if state.covers_through >= target_date:
        raise ValueError(
            f"The pretrained model covers {state.covers_through}, not before {target_date}."
        )
    return state, []


def first_required_feature_day(
    store: ModelStore,
    target_date: date,
    config: CustomLossLSTMConfig = CustomLossLSTMConfig(),
) -> date:
    return target_date - timedelta(days=config.bias_window_days)


__all__ = [
    "MODEL_DIR",
    "MODEL_NAME",
    "advance_state",
    "first_required_feature_day",
    "forecast_state",
    "pretrain_state",
]
