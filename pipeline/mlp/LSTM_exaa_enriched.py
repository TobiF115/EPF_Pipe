"""One-layer LSTM using C=5 features and target-day EXAA prices."""

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
    build_daily_mtu_matrix,
    inverse_vst_bias_corrected,
    robust_params,
    scale_fold_point,
)
from pipeline.mlp.NN_exaa_enriched import (
    _column_groups_exaa,
    _skip_connection_mask_exaa,
)
from pipeline.mlp.mlp import _daily_rows


_MTU_COLUMN = re.compile(r"^(.*)_mtu_?(\d{2})$")
_HOUR_COLUMN = re.compile(r"^(.*)_h(\d{2})$")
SEVEN_DAY_AVERAGE_PRICE_COLUMN = "average_da_price_last_7d"


def add_seven_day_average_price_feature(
    X: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Add the mean EPEX price from the seven complete days before each row."""
    if "price_da" not in prices.columns:
        raise ValueError("Prices lack the required 'price_da' column.")
    if not isinstance(X.index, pd.DatetimeIndex) or X.index.tz is None:
        raise ValueError("X must have a timezone-aware DatetimeIndex.")
    if SEVEN_DAY_AVERAGE_PRICE_COLUMN in X.columns:
        raise ValueError(
            f"X already contains {SEVEN_DAY_AVERAGE_PRICE_COLUMN!r}."
        )

    daily_prices = build_daily_mtu_matrix(prices["price_da"], "price_da")
    values = daily_prices.to_numpy(dtype=float)
    complete_days = np.isfinite(values).all(axis=1)
    daily_average = pd.Series(
        np.where(complete_days, values.mean(axis=1), np.nan),
        index=daily_prices.index,
    )
    if daily_average.index.tz != X.index.tz:
        daily_average.index = daily_average.index.tz_convert(X.index.tz)
    daily_index = daily_average.index.union(X.index).sort_values()
    daily_average = daily_average.reindex(daily_index)
    prior_seven_day_average = (
        daily_average.rolling(window=7, min_periods=7).mean().shift(1)
    )
    result = X.copy()
    result[SEVEN_DAY_AVERAGE_PRICE_COLUMN] = prior_seven_day_average.reindex(
        X.index
    )
    return result


def _skip_connection_mask_with_average_price(
    skip_columns: Sequence[str],
) -> np.ndarray:
    """Append an unrestricted skip connection for the seven-day mean."""
    return np.vstack(
        (
            _skip_connection_mask_exaa(skip_columns),
            np.ones((1, MTUS_PER_DAY), dtype=bool),
        )
    )


@dataclass(frozen=True)
class LSTMConfig:
    """Training settings shared by the EXAA-enriched and fundamental LSTMs."""

    hidden_units: int = 32
    train_days: int | None = None
    min_training_days: int = 7
    max_epochs: int = 400
    patience: int = 40
    learning_rate: float = 0.003
    fine_tune_fraction: float = 0.3
    fine_tune_learning_rate: float = 0.0005
    l2: float = 1e-4
    validation_fraction: float = 0.2
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_units < 1:
            raise ValueError("hidden_units must be positive.")
        if self.train_days is not None and self.train_days < 1:
            raise ValueError("train_days must be positive when provided.")
        if self.min_training_days < 6:
            raise ValueError("min_training_days must be at least 6 for two phases.")
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("max_epochs and patience must be positive.")
        if (
            self.learning_rate <= 0
            or self.fine_tune_learning_rate <= 0
            or self.fine_tune_learning_rate >= self.learning_rate
            or self.l2 < 0
        ):
            raise ValueError(
                "Learning rates must be positive with fine_tune_learning_rate "
                "below learning_rate, and l2 must be non-negative."
            )
        if not 0 < self.fine_tune_fraction < 1:
            raise ValueError("fine_tune_fraction must be between 0 and 1.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")


LSTMExaaEnrichedConfig = LSTMConfig


def _sequence_feature_tensor(
    features: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    """Map wide daily features to [day, MTU, channel] LSTM inputs."""
    column_positions = {column: index for index, column in enumerate(columns)}
    channel_names: list[str] = []
    assignments: list[tuple[int, int, np.ndarray]] = []

    for column in columns:
        mtu_match = _MTU_COLUMN.fullmatch(column)
        hour_match = _HOUR_COLUMN.fullmatch(column)
        if mtu_match:
            channel = mtu_match.group(1)
            mtu = int(mtu_match.group(2))
            if mtu >= MTUS_PER_DAY:
                raise ValueError(f"Invalid MTU in feature column {column!r}.")
            mtus = np.asarray([mtu], dtype=int)
        elif hour_match:
            channel = hour_match.group(1)
            hour = int(hour_match.group(2))
            if hour >= 24:
                raise ValueError(f"Invalid hour in feature column {column!r}.")
            mtus = np.arange(hour * 4, (hour + 1) * 4, dtype=int)
        else:
            channel = column
            mtus = np.arange(MTUS_PER_DAY, dtype=int)

        if channel not in channel_names:
            channel_names.append(channel)
        assignments.append(
            (column_positions[column], channel_names.index(channel), mtus)
        )

    coverage = np.zeros((len(channel_names), MTUS_PER_DAY), dtype=bool)
    for _, channel, mtus in assignments:
        if coverage[channel, mtus].any():
            raise ValueError(
                f"Feature channel {channel!r} maps multiple columns to one MTU."
            )
        coverage[channel, mtus] = True
    incomplete = [
        channel_names[index]
        for index in range(len(channel_names))
        if not coverage[index].all()
    ]
    if incomplete:
        raise ValueError(
            "LSTM feature channels must cover all 96 MTUs; incomplete: "
            + ", ".join(incomplete)
        )

    values = features.loc[:, columns].to_numpy(dtype=float)
    sequence = np.empty(
        (len(features), MTUS_PER_DAY, len(channel_names)), dtype=float
    )
    for column_index, channel_index, mtus in assignments:
        sequence[:, mtus, channel_index] = values[:, column_index, None]
    return sequence


class _OneLayerLSTMNetwork:
    """A sequence-to-sequence LSTM with a masked linear skip branch."""

    def __init__(
        self,
        config: LSTMConfig,
        skip_mask: np.ndarray,
    ) -> None:
        self.config = config
        self.skip_mask = np.asarray(skip_mask, dtype=bool)
        self.parameters: list[np.ndarray] = []

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))

    def _forward(
        self,
        sequence_inputs: np.ndarray,
        skip_inputs: np.ndarray,
        parameters: list[np.ndarray],
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, ...]]]:
        weights_recurrent, bias_recurrent, weights_out, bias_out, weights_skip = (
            parameters
        )
        sample_count, sequence_length, _ = sequence_inputs.shape
        hidden_count = self.config.hidden_units
        hidden = np.zeros((sample_count, hidden_count))
        cell = np.zeros_like(hidden)
        hidden_states: list[np.ndarray] = []
        cache: list[tuple[np.ndarray, ...]] = []
        outputs = np.empty((sample_count, sequence_length))
        skip_weights = weights_skip * self.skip_mask
        skip_values = skip_inputs @ skip_weights

        for step in range(sequence_length):
            previous_hidden = hidden
            previous_cell = cell
            recurrent_input = np.concatenate(
                (sequence_inputs[:, step, :], previous_hidden), axis=1
            )
            input_gate, forget_gate, output_gate, candidate = np.split(
                recurrent_input @ weights_recurrent + bias_recurrent, 4, axis=1
            )
            input_gate = self._sigmoid(input_gate)
            forget_gate = self._sigmoid(forget_gate)
            output_gate = self._sigmoid(output_gate)
            candidate = np.tanh(candidate)
            cell = forget_gate * previous_cell + input_gate * candidate
            cell_tanh = np.tanh(cell)
            hidden = output_gate * cell_tanh
            outputs[:, step] = (
                hidden @ weights_out[:, 0] + bias_out[0] + skip_values[:, step]
            )
            hidden_states.append(hidden)
            cache.append(
                (
                    recurrent_input,
                    previous_cell,
                    input_gate,
                    forget_gate,
                    output_gate,
                    candidate,
                    cell_tanh,
                )
            )
        return outputs, cache

    def _train(
        self,
        sequence_inputs: np.ndarray,
        skip_inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int,
        learning_rate: float,
        seed: int,
        *,
        initial_parameters: list[np.ndarray] | None = None,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], int]:
        rng = np.random.default_rng(seed)
        input_count = sequence_inputs.shape[2]
        hidden_count = self.config.hidden_units
        skip_count = skip_inputs.shape[1]
        output_count = targets.shape[1]
        if initial_parameters is None:
            parameters = [
                rng.normal(
                    0.0,
                    math.sqrt(1.0 / (input_count + hidden_count)),
                    (input_count + hidden_count, 4 * hidden_count),
                ),
                np.zeros(4 * hidden_count),
                rng.normal(
                    0.0,
                    math.sqrt(2.0 / (hidden_count + 1)),
                    (hidden_count, 1),
                ),
                np.zeros(1),
                np.zeros((skip_count, output_count)),
            ]
        else:
            if len(initial_parameters) != 5:
                raise ValueError("Initial parameters do not match the LSTM model.")
            parameters = [value.copy() for value in initial_parameters]

        first_moments = [np.zeros_like(value) for value in parameters]
        second_moments = [np.zeros_like(value) for value in parameters]
        beta_1, beta_2 = 0.9, 0.999
        best_parameters: list[np.ndarray] | None = None
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0

        for epoch in range(1, epochs + 1):
            predictions, cache = self._forward(
                sequence_inputs, skip_inputs, parameters
            )
            residual = 2.0 * (predictions - targets) / targets.size
            weights_recurrent, _, weights_out, _, weights_skip = parameters
            grad_recurrent = np.zeros_like(weights_recurrent)
            grad_bias_recurrent = np.zeros_like(parameters[1])
            grad_out = np.zeros_like(weights_out)
            grad_bias_out = np.asarray([residual.sum()])
            grad_skip = (
                skip_inputs.T @ residual + 2.0 * self.config.l2 * weights_skip
            ) * self.skip_mask
            hidden_gradient_next = np.zeros(
                (sequence_inputs.shape[0], hidden_count)
            )
            cell_gradient_next = np.zeros_like(hidden_gradient_next)

            for step in range(sequence_inputs.shape[1] - 1, -1, -1):
                (
                    recurrent_input,
                    previous_cell,
                    input_gate,
                    forget_gate,
                    output_gate,
                    candidate,
                    cell_tanh,
                ) = cache[step]
                step_residual = residual[:, step : step + 1]
                hidden = output_gate * cell_tanh
                grad_out += hidden.T @ step_residual
                hidden_gradient = (
                    hidden_gradient_next
                    + step_residual @ weights_out.T
                )
                output_gate_gradient = hidden_gradient * cell_tanh
                cell_gradient = (
                    cell_gradient_next
                    + hidden_gradient
                    * output_gate
                    * (1.0 - cell_tanh**2)
                )
                forget_gate_gradient = cell_gradient * previous_cell
                input_gate_gradient = cell_gradient * candidate
                candidate_gradient = cell_gradient * input_gate
                cell_gradient_next = cell_gradient * forget_gate

                gate_gradients = np.concatenate(
                    (
                        input_gate_gradient * input_gate * (1.0 - input_gate),
                        forget_gate_gradient * forget_gate * (1.0 - forget_gate),
                        output_gate_gradient * output_gate * (1.0 - output_gate),
                        candidate_gradient * (1.0 - candidate**2),
                    ),
                    axis=1,
                )
                grad_recurrent += recurrent_input.T @ gate_gradients
                grad_bias_recurrent += gate_gradients.sum(axis=0)
                hidden_gradient_next = (
                    gate_gradients @ weights_recurrent[input_count:, :].T
                )

            grad_recurrent += 2.0 * self.config.l2 * weights_recurrent
            grad_out += 2.0 * self.config.l2 * weights_out
            gradients = [
                grad_recurrent,
                grad_bias_recurrent,
                grad_out,
                grad_bias_out,
                grad_skip,
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
                parameters[index] -= learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + 1e-8
                )

            if validation is not None:
                validation_sequence, validation_skip, validation_targets = validation
                validation_predictions, _ = self._forward(
                    validation_sequence, validation_skip, parameters
                )
                loss = float(
                    np.mean((validation_predictions - validation_targets) ** 2)
                )
                if not np.isfinite(loss):
                    raise FloatingPointError("LSTM validation loss became non-finite.")
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
            raise RuntimeError("LSTM training did not produce a valid validation loss.")
        return best_parameters, best_epoch

    def fit(
        self,
        sequence_inputs: np.ndarray,
        skip_inputs: np.ndarray,
        targets: np.ndarray,
    ) -> "_OneLayerLSTMNetwork":
        if sequence_inputs.ndim != 3 or targets.shape != sequence_inputs.shape[:2]:
            raise ValueError("LSTM inputs and targets must share day and MTU axes.")
        if self.skip_mask.shape != (skip_inputs.shape[1], targets.shape[1]):
            raise ValueError("Skip-connection mask does not match model inputs.")
        if not self.skip_mask.any(axis=0).all():
            raise ValueError("Every output must have at least one price skip input.")

        phase_2_count = max(
            3, int(math.ceil(len(targets) * self.config.fine_tune_fraction))
        )
        phase_1_count = len(targets) - phase_2_count
        if phase_1_count < 3:
            raise ValueError("Not enough training days for the two model phases.")

        phase_1_sequence = sequence_inputs[:phase_1_count]
        phase_1_skip = skip_inputs[:phase_1_count]
        phase_1_targets = targets[:phase_1_count]
        phase_1_split = len(phase_1_targets) - max(
            1,
            int(math.ceil(len(phase_1_targets) * self.config.validation_fraction)),
        )
        if phase_1_split < 2:
            raise ValueError("Phase 1 is too short for validation.")
        _, phase_1_epochs = self._train(
            phase_1_sequence[:phase_1_split],
            phase_1_skip[:phase_1_split],
            phase_1_targets[:phase_1_split],
            self.config.max_epochs,
            self.config.learning_rate,
            self.config.random_seed,
            validation=(
                phase_1_sequence[phase_1_split:],
                phase_1_skip[phase_1_split:],
                phase_1_targets[phase_1_split:],
            ),
        )
        phase_1_parameters, _ = self._train(
            phase_1_sequence,
            phase_1_skip,
            phase_1_targets,
            phase_1_epochs,
            self.config.learning_rate,
            self.config.random_seed,
        )

        phase_2_sequence = sequence_inputs[phase_1_count:]
        phase_2_skip = skip_inputs[phase_1_count:]
        phase_2_targets = targets[phase_1_count:]
        phase_2_split = len(phase_2_targets) - max(
            1,
            int(math.ceil(len(phase_2_targets) * self.config.validation_fraction)),
        )
        if phase_2_split < 2:
            raise ValueError("Phase 2 is too short for validation.")
        _, phase_2_epochs = self._train(
            phase_2_sequence[:phase_2_split],
            phase_2_skip[:phase_2_split],
            phase_2_targets[:phase_2_split],
            self.config.max_epochs,
            self.config.fine_tune_learning_rate,
            self.config.random_seed,
            initial_parameters=phase_1_parameters,
            validation=(
                phase_2_sequence[phase_2_split:],
                phase_2_skip[phase_2_split:],
                phase_2_targets[phase_2_split:],
            ),
        )
        self.parameters, _ = self._train(
            phase_2_sequence,
            phase_2_skip,
            phase_2_targets,
            phase_2_epochs,
            self.config.fine_tune_learning_rate,
            self.config.random_seed,
            initial_parameters=phase_1_parameters,
        )
        return self

    def predict(
        self,
        sequence_inputs: np.ndarray,
        skip_inputs: np.ndarray,
    ) -> np.ndarray:
        if not self.parameters:
            raise RuntimeError("LSTM must be fitted before calling predict.")
        predictions, _ = self._forward(sequence_inputs, skip_inputs, self.parameters)
        if not np.isfinite(predictions).all():
            raise FloatingPointError("LSTM produced non-finite predictions.")
        return predictions


def _rolling_lstm_forecast(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    forecast_days: Sequence[object],
    config: LSTMConfig,
    *,
    hidden_columns: Sequence[str],
    skip_columns: Sequence[str],
    skip_mask: np.ndarray,
    require_seven_day_average: bool = True,
) -> pd.DataFrame:
    """Train and forecast from caller-selected sequence and skip features."""
    if not isinstance(config, LSTMConfig):
        raise TypeError("config must be an LSTMConfig.")
    if list(Y.columns) != list(range(MTUS_PER_DAY)):
        raise ValueError("Y must contain target columns 0 through 95 in order.")
    if X.empty or Y.empty:
        raise ValueError("X and Y must not be empty.")
    if X.columns.has_duplicates:
        raise ValueError("X contains duplicate feature columns.")

    if (
        require_seven_day_average
        and SEVEN_DAY_AVERAGE_PRICE_COLUMN not in hidden_columns
    ):
        raise ValueError("LSTM features must include the seven-day average DA price.")
    hidden_columns = list(hidden_columns)
    skip_columns = list(skip_columns)
    skip_mask = np.asarray(skip_mask, dtype=bool)
    if skip_mask.shape != (len(skip_columns), MTUS_PER_DAY):
        raise ValueError("Skip-connection mask does not match model inputs.")
    x_rows = _daily_rows(X, "X")
    y_rows = _daily_rows(Y, "Y")
    days = [pd.Timestamp(value).date() for value in forecast_days]
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
        if not np.isfinite(x_values[[test_position]]).all():
            raise ValueError(f"Forecast features are incomplete for {day}.")

        first_training_day = (
            day - timedelta(days=config.train_days)
            if config.train_days is not None
            else None
        )
        candidate_days = [
            sample_day
            for sample_day in sorted(x_rows)
            if (first_training_day is None or first_training_day <= sample_day)
            and sample_day < day
            and sample_day in y_rows
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

        scaled_train, scaled_test, _, _, _ = scale_fold_point(
            X_tr=x_train,
            X_va=x_test,
            y_tr=pd.Series(y_train_raw[:, 0]),
            y_va=None,
        )
        train_sequence = _sequence_feature_tensor(scaled_train, hidden_columns)
        test_sequence = _sequence_feature_tensor(scaled_test, hidden_columns)
        skip_train = scaled_train[skip_columns].to_numpy(dtype=float)
        skip_test = scaled_test[skip_columns].to_numpy(dtype=float)
        model = _OneLayerLSTMNetwork(config, skip_mask)
        model.fit(train_sequence, skip_train, transformed_targets)
        transformed_prediction = model.predict(test_sequence, skip_test)[0]
        fitted_residuals = transformed_targets - model.predict(
            train_sequence, skip_train
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
            raise FloatingPointError(f"LSTM forecast is non-finite for {day}.")

        actual = (
            y_values[y_rows[day]]
            if day in y_rows
            else np.full(MTUS_PER_DAY, np.nan)
        )
        output_records.append(np.column_stack((prediction, actual)))
        output_days.append(day)

    forecast_index = make_delivery_index(pd.DatetimeIndex(output_days))
    return pd.DataFrame(
        np.vstack(output_records),
        index=forecast_index,
        columns=["y_pred", "y_true"],
    )


_OneLayerLSTMExaaNetwork = _OneLayerLSTMNetwork


def rolling_lstm_exaa_enriched_forecast(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    forecast_days: Sequence[object],
    config: LSTMExaaEnrichedConfig = LSTMExaaEnrichedConfig(),
) -> pd.DataFrame:
    """Forecast each day as a 96-step MTU sequence with EXAA-enriched inputs."""
    if not isinstance(config, LSTMExaaEnrichedConfig):
        raise TypeError("config must be an LSTMExaaEnrichedConfig.")
    hidden_columns, skip_columns = _column_groups_exaa(X.columns.tolist())
    if SEVEN_DAY_AVERAGE_PRICE_COLUMN not in hidden_columns:
        raise ValueError("LSTM features must include the seven-day average DA price.")
    skip_mask = _skip_connection_mask_with_average_price(skip_columns)
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
