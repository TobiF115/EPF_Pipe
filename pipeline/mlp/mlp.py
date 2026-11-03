"""Daily C=5 neural-network point forecasts with linear price skip connections."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

from pipeline.delivery_index import MTUS_PER_DAY, make_delivery_index
from pipeline.lear.lear_model import (
    inverse_vst_bias_corrected,
    robust_params,
    scale_fold_point,
)


_PRICE_COLUMN = re.compile(r"^price_d([127])_mtu_(\d{2})$")
_PRICE_LAGS = (1, 2, 7)
_LAST_HOUR_MTUS = range(MTUS_PER_DAY - 4, MTUS_PER_DAY)


@dataclass(frozen=True)
class MLPConfig:
    """Training settings for the one-hidden-layer daily MLP."""

    hidden_units: int = 32
    train_days: int = 56
    min_training_days: int = 14
    max_epochs: int = 400
    patience: int = 40
    learning_rate: float = 0.003
    l2: float = 1e-4
    validation_fraction: float = 0.2
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_units != 32:
            raise ValueError("The MLP architecture requires exactly 32 hidden units.")
        if self.train_days < 1:
            raise ValueError("train_days must be positive.")
        if self.min_training_days < 3:
            raise ValueError("min_training_days must be at least 3.")
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("max_epochs and patience must be positive.")
        if self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("learning_rate must be positive and l2 non-negative.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")


def _column_groups(columns: Sequence[str]) -> tuple[list[str], list[str]]:
    """Separate LEAR price lags for the linear branch from other input features."""
    by_lag: dict[int, dict[int, str]] = {lag: {} for lag in _PRICE_LAGS}
    hidden_columns: list[str] = []
    unsupported_price_columns: list[str] = []

    for column in columns:
        match = _PRICE_COLUMN.fullmatch(column)
        if match is None:
            if column.startswith(("price_", "exaa_")):
                unsupported_price_columns.append(column)
            else:
                hidden_columns.append(column)
            continue
        lag = int(match.group(1))
        mtu = int(match.group(2))
        by_lag[lag][mtu] = column

    if unsupported_price_columns:
        raise ValueError(
            "The C=5 fundamental MLP accepts only price_d1, price_d2, and "
            f"price_d7 features; unsupported columns: {unsupported_price_columns}."
        )
    if not hidden_columns:
        raise ValueError("At least one non-price weather/load feature is required.")

    expected_mtus = set(range(MTUS_PER_DAY))
    for lag, lag_columns in by_lag.items():
        if set(lag_columns) != expected_mtus:
            missing = sorted(expected_mtus - set(lag_columns))
            extra = sorted(set(lag_columns) - expected_mtus)
            raise ValueError(
                f"price_d{lag} must contain MTU columns 00 through 95; "
                f"missing={missing}, extra={extra}."
            )

    skip_columns = [
        by_lag[1][mtu] for mtu in _LAST_HOUR_MTUS
    ] + [
        by_lag[lag][mtu]
        for lag in _PRICE_LAGS
        for mtu in range(MTUS_PER_DAY)
    ]
    return hidden_columns, skip_columns


def _as_day(value: object) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid forecast day: {value!r}.")
    return timestamp.date()


def _daily_rows(frame: pd.DataFrame, label: str) -> dict[date, int]:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{label} must have a DatetimeIndex of daily rows.")
    if not frame.index.equals(frame.index.normalize()):
        raise ValueError(f"{label} index must contain normalized daily dates.")
    if frame.index.has_duplicates:
        raise ValueError(f"{label} contains duplicate daily indexes.")
    dates = [_as_day(value) for value in frame.index]
    if len(set(dates)) != len(dates):
        raise ValueError(f"{label} contains more than one row for a delivery day.")
    return {day: position for position, day in enumerate(dates)}


class _OneHiddenLayerMLP:
    """Dense 32-unit ReLU branch plus a direct linear price-lag branch."""

    def __init__(self, config: MLPConfig) -> None:
        self.config = config
        self.parameters: list[np.ndarray] = []

    @staticmethod
    def _forward(
        hidden_inputs: np.ndarray,
        skip_inputs: np.ndarray,
        parameters: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        weights_in, bias_hidden, weights_out, bias_out, weights_skip = parameters
        hidden = np.maximum(0.0, hidden_inputs @ weights_in + bias_hidden)
        output = hidden @ weights_out + bias_out + skip_inputs @ weights_skip
        return output, hidden

    def _train(
        self,
        hidden_inputs: np.ndarray,
        skip_inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int,
        seed: int,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], int]:
        rng = np.random.default_rng(seed)
        input_count = hidden_inputs.shape[1]
        skip_count = skip_inputs.shape[1]
        hidden_count = self.config.hidden_units
        output_count = targets.shape[1]
        parameters = [
            rng.normal(0.0, math.sqrt(2.0 / input_count), (input_count, hidden_count)),
            np.zeros(hidden_count),
            rng.normal(0.0, math.sqrt(2.0 / hidden_count), (hidden_count, output_count)),
            np.zeros(output_count),
            np.zeros((skip_count, output_count)),
        ]
        first_moments = [np.zeros_like(value) for value in parameters]
        second_moments = [np.zeros_like(value) for value in parameters]
        beta_1, beta_2 = 0.9, 0.999

        best_parameters: list[np.ndarray] | None = None
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            predictions, hidden = self._forward(
                hidden_inputs, skip_inputs, parameters
            )
            residual = 2.0 * (predictions - targets) / targets.shape[0]
            weights_in, _, weights_out, _, weights_skip = parameters
            hidden_gradient = residual @ weights_out.T
            hidden_gradient *= (hidden > 0.0)
            gradients = [
                hidden_inputs.T @ hidden_gradient
                + 2.0 * self.config.l2 * weights_in,
                hidden_gradient.sum(axis=0),
                hidden.T @ residual + 2.0 * self.config.l2 * weights_out,
                residual.sum(axis=0),
                skip_inputs.T @ residual
                + 2.0 * self.config.l2 * weights_skip,
            ]
            for index, gradient in enumerate(gradients):
                first_moments[index] = (
                    beta_1 * first_moments[index] + (1.0 - beta_1) * gradient
                )
                second_moments[index] = (
                    beta_2 * second_moments[index]
                    + (1.0 - beta_2) * gradient * gradient
                )
                corrected_first = first_moments[index] / (1.0 - beta_1**epoch)
                corrected_second = second_moments[index] / (1.0 - beta_2**epoch)
                parameters[index] -= self.config.learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + 1e-8
                )
            if validation is not None:
                validation_hidden, validation_skip, validation_targets = validation
                validation_predictions, _ = self._forward(
                    validation_hidden, validation_skip, parameters
                )
                loss = float(
                    np.mean((validation_predictions - validation_targets) ** 2)
                )
                if not np.isfinite(loss):
                    raise FloatingPointError(
                        "MLP validation loss became non-finite."
                    )
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    best_parameters = [value.copy() for value in parameters]
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break
        if validation is None:
            return parameters, epochs
        if best_parameters is None:
            raise RuntimeError("MLP training did not produce a valid validation loss.")
        return best_parameters, best_epoch

    def fit(
        self,
        hidden_inputs: np.ndarray,
        skip_inputs: np.ndarray,
        targets: np.ndarray,
    ) -> "_OneHiddenLayerMLP":
        validation_count = max(
            1, int(math.ceil(len(targets) * self.config.validation_fraction))
        )
        split_at = len(targets) - validation_count
        if split_at < 2:
            raise ValueError(
                "Not enough training days remain for a time-ordered validation split."
            )
        validation = (
            hidden_inputs[split_at:],
            skip_inputs[split_at:],
            targets[split_at:],
        )
        _, best_epoch = self._train(
            hidden_inputs[:split_at],
            skip_inputs[:split_at],
            targets[:split_at],
            self.config.max_epochs,
            self.config.random_seed,
            validation,
        )
        self.parameters, _ = self._train(
            hidden_inputs,
            skip_inputs,
            targets,
            best_epoch,
            self.config.random_seed,
        )
        return self

    def predict(
        self, hidden_inputs: np.ndarray, skip_inputs: np.ndarray
    ) -> np.ndarray:
        if not self.parameters:
            raise RuntimeError("MLP must be fitted before calling predict.")
        predictions, _ = self._forward(
            hidden_inputs, skip_inputs, self.parameters
        )
        if not np.isfinite(predictions).all():
            raise FloatingPointError("MLP produced non-finite predictions.")
        return predictions


def rolling_mlp_forecast(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    forecast_days: Sequence[object],
    config: MLPConfig = MLPConfig(),
) -> pd.DataFrame:
    """Forecast each day from preprocessed C=5 LEAR features.

    ``X`` must be the daily feature matrix made by the existing LEAR pipeline;
    ``Y`` must be its 96-MTU daily target matrix. The four final quarter-hours
    of the previous day are interpreted as the previous-day-last-hour skip.
    """
    if not isinstance(config, MLPConfig):
        raise TypeError("config must be an MLPConfig.")
    expected_targets = list(range(MTUS_PER_DAY))
    if list(Y.columns) != expected_targets:
        raise ValueError("Y must contain target columns 0 through 95 in order.")
    if X.empty or Y.empty:
        raise ValueError("X and Y must not be empty.")
    if X.columns.has_duplicates:
        raise ValueError("X contains duplicate feature columns.")

    hidden_columns, skip_columns = _column_groups(X.columns.tolist())
    x_rows = _daily_rows(X, "X")
    y_rows = _daily_rows(Y, "Y")
    days = [_as_day(value) for value in forecast_days]
    if not days:
        raise ValueError("At least one forecast day is required.")
    if len(set(days)) != len(days):
        raise ValueError("forecast_days must not contain duplicates.")

    x_values = X.to_numpy(dtype=float)
    y_values = Y.to_numpy(dtype=float)
    output_records: list[np.ndarray] = []
    output_days: list[date] = []

    for day in sorted(days):
        if day not in x_rows:
            raise ValueError(f"No feature row is available for forecast day {day}.")
        test_position = x_rows[day]
        test_features = x_values[[test_position]]
        if not np.isfinite(test_features).all():
            raise ValueError(f"Forecast features are incomplete for {day}.")

        first_training_day = day - timedelta(days=config.train_days)
        candidate_days = [
            sample_day
            for sample_day in sorted(x_rows)
            if first_training_day <= sample_day < day and sample_day in y_rows
        ]
        training_positions = np.asarray([x_rows[value] for value in candidate_days])
        target_positions = np.asarray([y_rows[value] for value in candidate_days])
        if len(candidate_days):
            candidate_x = x_values[training_positions]
            candidate_y = y_values[target_positions]
            complete = np.isfinite(candidate_x).all(axis=1) & np.isfinite(
                candidate_y
            ).all(axis=1)
            training_positions = training_positions[complete]
            target_positions = target_positions[complete]
        if len(training_positions) < config.min_training_days:
            raise ValueError(
                f"Only {len(training_positions)} complete training days remain "
                f"for {day}; at least {config.min_training_days} are required."
            )

        x_train = X.iloc[training_positions]
        x_test = X.iloc[[test_position]]
        y_train_raw = y_values[target_positions]

        target_centers = np.empty(MTUS_PER_DAY)
        target_scales = np.empty(MTUS_PER_DAY)
        transformed_targets = np.empty_like(y_train_raw)
        for mtu in range(MTUS_PER_DAY):
            center, scale = robust_params(y_train_raw[:, mtu])
            target_centers[mtu] = center
            target_scales[mtu] = scale
            transformed_targets[:, mtu] = np.arcsinh(
                (y_train_raw[:, mtu] - center) / scale
            )

        # Fit feature transforms only on this rolling training window.
        scaled_full_train, scaled_test, _, _, _ = scale_fold_point(
            X_tr=x_train,
            X_va=x_test,
            y_tr=pd.Series(y_train_raw[:, 0]),
            y_va=None,
        )
        hidden_full_scaled = scaled_full_train[hidden_columns].to_numpy(dtype=float)
        hidden_test_scaled = scaled_test[hidden_columns].to_numpy(dtype=float)
        skip_full_scaled = scaled_full_train[skip_columns].to_numpy(dtype=float)
        skip_test_scaled = scaled_test[skip_columns].to_numpy(dtype=float)
        model = _OneHiddenLayerMLP(config)
        model.fit(hidden_full_scaled, skip_full_scaled, transformed_targets)
        transformed_prediction = model.predict(
            hidden_test_scaled, skip_test_scaled
        )[0]
        fitted_residuals = transformed_targets - model.predict(
            hidden_full_scaled, skip_full_scaled
        )
        prediction = np.asarray(
            [
                inverse_vst_bias_corrected(
                    [transformed_prediction[mtu]],
                    fitted_residuals[:, mtu],
                    target_centers[mtu],
                    target_scales[mtu],
                )[0]
                for mtu in range(MTUS_PER_DAY)
            ]
        )
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"MLP forecast is non-finite for {day}.")

        actual = (
            y_values[y_rows[day]]
            if day in y_rows
            else np.full(MTUS_PER_DAY, np.nan)
        )
        output_records.append(np.column_stack((prediction, actual)))
        output_days.append(day)

    delivery_dates = pd.DatetimeIndex(output_days)
    forecast_index = make_delivery_index(delivery_dates)
    forecast_values = np.vstack(output_records)
    return pd.DataFrame(
        forecast_values,
        index=forecast_index,
        columns=["y_pred", "y_true"],
    )