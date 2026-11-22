"""Shared core of the three custom-loss LSTMs.

Everything here is identical for all three models: inputs and frozen scaling,
the one-layer LSTM, the loss, the ERA5 pretraining, the forecast and the
per-day checkpoint store. The models differ only in the daily update rule
passed to ``advance``.

Network
-------
The 96 MTUs of the target day are the time steps of a one-layer LSTM. Its
input channels (MTU-resolved columns at their MTU, hourly columns at the
hour's four MTUs, daily columns at all MTUs) are the target-day EXAA prices,
the seven-day mean EPEX price, C5 weather, load and calendar dummies.

The output of MTU t is a linear map of the hidden state plus a masked linear
skip branch on the prices, which carries the (largely linear) persistence:
EPEX d-1 at MTUs t-8..t+8, EPEX d-2 and d-7 at MTU t, EXAA at MTUs t-2..t+2
(cyclic) and the seven-day mean, i.e. 25 links per output. The EPEX lags
enter only the skip branch: also feeding them to the LSTM raised the backtest
RMSE by ~1.7 EUR/MWh (overfitting); EXAA and the mean enter both branches.

Loss (Salihoglu, Ahmed & Asadi, arXiv:2510.16898, Eqs. 6-11)::

    Loss = MAE + alpha * JSD + beta * SM        (alpha = 0.001, beta = 0.03)

* ``MAE``: mean absolute error over all days and MTUs. In the backtests it
  gave a lower out-of-sample RMSE than MSE- or RMSE-based losses, presumably
  because it is robust to price spikes in the training days.
* ``JSD``: Jensen-Shannon divergence between the softmax of the predicted and
  of the actual daily profile (96 MTUs): a shape regulariser.
* ``SM``: sum of squared first differences of the predicted profile.

Further differences from the paper: the loss is evaluated on the model scale
(per-MTU median/MAD + arcsinh), JSD and SM are averaged over the days of a
batch instead of summed, and the L2 weight penalty (1e-4) is kept. The weights
alpha and beta are the paper's values, tuned there for the MAE term. As in the
paper, early stopping in pretraining monitors the validation MAE.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from pipeline.delivery_index import MTUS_PER_DAY, make_delivery_index
from pipeline.lear.lear_model import inverse_vst_bias_corrected, robust_params
from pipeline.mlp.LSTM_era5_online import (
    FrozenFeatureScaler,
    _as_date,
    _atomic_write,
    _complete_rows,
    era5_pretrain_config,
)
from pipeline.mlp.LSTM_exaa_enriched import (
    SEVEN_DAY_AVERAGE_PRICE_COLUMN,
    LSTMConfig,
    _sequence_feature_tensor,
)
from pipeline.mlp.mlp import _PRICE_COLUMN as _EPEX_LAG_COLUMN
from pipeline.mlp.mlp import _column_groups, _skip_connection_mask
from pipeline.mlp.NN_exaa_enriched import _column_groups_exaa, _skip_connection_mask_exaa

MODEL_ROOT = Path("results/operational/mlp/lstm_custom_loss")
STATE_FORMAT_VERSION = 5  # 5: EPEX lags skip-only; EXAA and 7-day mean in LSTM and skip
PARAMETER_COUNT = 5
PRETRAINED_NAME = "pretrained"
CHECKPOINT_DIR = "checkpoints"
_EPS = 1e-12



# ====================================================================
# Configuration
# ====================================================================
@dataclass(frozen=True)
class CustomLoss:
    """Regulariser weights next to the MAE (paper: 0.001 and 0.03)."""

    jsd_weight: float = 0.001
    smoothness_weight: float = 0.03

    def __post_init__(self) -> None:
        if self.jsd_weight < 0 or self.smoothness_weight < 0:
            raise ValueError("Loss weights must be non-negative.")


@dataclass(frozen=True)
class CustomLossLSTMConfig:
    """Settings of all three models; each model reads only its own update block."""

    pretrain: LSTMConfig = field(default_factory=era5_pretrain_config)
    loss: CustomLoss = field(default_factory=CustomLoss)
    bias_window_days: int = 28
    max_catch_up_days: int = 120
    # Daily update (both online models): Adam steps and learning rate.
    update_epochs: int = 10
    update_learning_rate: float = 1e-5
    # Rolling-window model: train on the last 7 complete days.
    window_days: int = 7
    min_window_days: int = 5
    # Adaptive model: train on the new day, validate on the 6 days before it;
    # accept if the validation RMSE (EUR/MWh) drops by more than the margin.
    validation_days: int = 6
    min_validation_days: int = 3
    acceptance_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.bias_window_days < 1 or self.max_catch_up_days < 1:
            raise ValueError("bias_window_days and max_catch_up_days must be positive.")
        if self.update_epochs < 1 or self.update_learning_rate <= 0:
            raise ValueError("update_epochs and update_learning_rate must be positive.")
        if not 1 <= self.min_window_days <= self.window_days:
            raise ValueError("Require 1 <= min_window_days <= window_days.")
        if not 1 <= self.min_validation_days <= self.validation_days:
            raise ValueError("Require 1 <= min_validation_days <= validation_days.")
        if self.acceptance_margin < 0:
            raise ValueError("acceptance_margin must be non-negative.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "CustomLossLSTMConfig":
        """Defaults, overridable via ``LSTM_CL_JSD_WEIGHT``,
        ``LSTM_CL_SMOOTHNESS_WEIGHT``, ``LSTM_CL_UPDATE_LEARNING_RATE`` and
        ``LSTM_CL_UPDATE_EPOCHS``."""
        env = os.environ if environ is None else environ
        loss = CustomLoss(
            jsd_weight=float(env.get("LSTM_CL_JSD_WEIGHT") or CustomLoss.jsd_weight),
            smoothness_weight=float(
                env.get("LSTM_CL_SMOOTHNESS_WEIGHT") or CustomLoss.smoothness_weight
            ),
        )
        overrides: dict = {"loss": loss}
        if env.get("LSTM_CL_UPDATE_LEARNING_RATE"):
            overrides["update_learning_rate"] = float(env["LSTM_CL_UPDATE_LEARNING_RATE"])
        if env.get("LSTM_CL_UPDATE_EPOCHS"):
            overrides["update_epochs"] = int(env["LSTM_CL_UPDATE_EPOCHS"])
        return cls(**overrides)


# ====================================================================
# Loss: MAE + alpha * JSD + beta * SM
# ====================================================================
def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - values.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def _jsd_terms(predictions: np.ndarray, targets: np.ndarray):
    p = _softmax(predictions)
    q = _softmax(targets)
    m = 0.5 * (p + q)
    log_pm = np.log((p + _EPS) / (m + _EPS))
    log_qm = np.log((q + _EPS) / (m + _EPS))
    jsd = 0.5 * (np.sum(p * log_pm, axis=1) + np.sum(q * log_qm, axis=1))
    return p, log_pm, jsd


def custom_loss_components(
    predictions: np.ndarray, targets: np.ndarray, loss: CustomLoss
) -> dict[str, float]:
    """The three loss terms (JSD and SM averaged over days) and their total."""
    predictions = np.atleast_2d(predictions)
    targets = np.atleast_2d(targets)
    mae = float(np.mean(np.abs(predictions - targets)))
    _, _, jsd = _jsd_terms(predictions, targets)
    smoothness = np.sum(np.diff(predictions, axis=1) ** 2, axis=1)
    total = (
        mae
        + loss.jsd_weight * float(jsd.mean())
        + loss.smoothness_weight * float(smoothness.mean())
    )
    return {"loss": total, "mae": mae, "jsd": float(jsd.mean()), "smoothness": float(smoothness.mean())}


def custom_loss(predictions: np.ndarray, targets: np.ndarray, loss: CustomLoss) -> float:
    return custom_loss_components(predictions, targets, loss)["loss"]


def custom_loss_gradient(
    predictions: np.ndarray, targets: np.ndarray, loss: CustomLoss
) -> np.ndarray:
    """d Loss / d predictions, shape (days, MTUs)."""
    day_count = predictions.shape[0]
    gradient = np.sign(predictions - targets) / predictions.size  # MAE subgradient
    if loss.jsd_weight:
        # dJSD/dp_i = 0.5 * log(p_i / m_i); chain rule through the softmax.
        p, log_pm, _ = _jsd_terms(predictions, targets)
        g = 0.5 * log_pm
        gradient += loss.jsd_weight / day_count * p * (g - np.sum(p * g, axis=1, keepdims=True))
    if loss.smoothness_weight:
        steps = np.diff(predictions, axis=1)
        sm_grad = np.zeros_like(predictions)
        sm_grad[:, 1:] += 2.0 * steps
        sm_grad[:, :-1] -= 2.0 * steps
        gradient += loss.smoothness_weight / day_count * sm_grad
    return gradient




# ====================================================================
# Network: one-layer LSTM, linear output plus masked price skip branch
# ====================================================================
class LSTMNetwork:
    """Sequence-to-sequence LSTM over the 96 MTUs, trained with the custom loss.

    Parameters: ``[W (inputs+hidden, 4*hidden), b (4*hidden), w_out (hidden, 1),
    b_out (1), V (skip inputs, 96)]``; gate order input, forget, output,
    candidate. Only the entries of ``V`` allowed by ``skip_mask`` are used.
    """

    def __init__(self, config: LSTMConfig, loss: CustomLoss, skip_mask: np.ndarray) -> None:
        self.config = config
        self.loss = loss
        self.skip_mask = np.asarray(skip_mask, dtype=bool)
        self.parameters: list[np.ndarray] = []

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))

    def _forward(self, sequence: np.ndarray, skip: np.ndarray, parameters: list[np.ndarray]):
        weights, bias, weights_out, bias_out, weights_skip = parameters
        skip_values = skip @ (weights_skip * self.skip_mask)
        sample_count, sequence_length, _ = sequence.shape
        hidden = np.zeros((sample_count, self.config.hidden_units))
        cell = np.zeros_like(hidden)
        outputs = np.empty((sample_count, sequence_length))
        cache = []
        for step in range(sequence_length):
            previous_cell = cell
            recurrent_input = np.concatenate((sequence[:, step, :], hidden), axis=1)
            input_gate, forget_gate, output_gate, candidate = np.split(
                recurrent_input @ weights + bias, 4, axis=1
            )
            input_gate = self._sigmoid(input_gate)
            forget_gate = self._sigmoid(forget_gate)
            output_gate = self._sigmoid(output_gate)
            candidate = np.tanh(candidate)
            cell = forget_gate * previous_cell + input_gate * candidate
            cell_tanh = np.tanh(cell)
            hidden = output_gate * cell_tanh
            outputs[:, step] = hidden @ weights_out[:, 0] + bias_out[0] + skip_values[:, step]
            cache.append(
                (recurrent_input, previous_cell, input_gate, forget_gate, output_gate, candidate, cell_tanh)
            )
        return outputs, cache

    def _gradients(
        self,
        sequence: np.ndarray,
        skip: np.ndarray,
        targets: np.ndarray,
        parameters: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Backpropagation through time for the custom loss plus L2."""
        input_count = sequence.shape[2]
        hidden_count = self.config.hidden_units
        predictions, cache = self._forward(sequence, skip, parameters)
        residual = custom_loss_gradient(predictions, targets, self.loss)
        weights, _, weights_out, _, weights_skip = parameters
        grad_skip = (skip.T @ residual + 2.0 * self.config.l2 * weights_skip) * self.skip_mask
        grad_weights = np.zeros_like(weights)
        grad_bias = np.zeros_like(parameters[1])
        grad_out = np.zeros_like(weights_out)
        grad_bias_out = np.asarray([residual.sum()])
        hidden_gradient_next = np.zeros((sequence.shape[0], hidden_count))
        cell_gradient_next = np.zeros_like(hidden_gradient_next)

        for step in range(sequence.shape[1] - 1, -1, -1):
            (recurrent_input, previous_cell, input_gate, forget_gate,
             output_gate, candidate, cell_tanh) = cache[step]
            step_residual = residual[:, step : step + 1]
            grad_out += (output_gate * cell_tanh).T @ step_residual
            hidden_gradient = hidden_gradient_next + step_residual @ weights_out.T
            output_gate_gradient = hidden_gradient * cell_tanh
            cell_gradient = cell_gradient_next + hidden_gradient * output_gate * (1.0 - cell_tanh**2)
            gate_gradients = np.concatenate(
                (
                    cell_gradient * candidate * input_gate * (1.0 - input_gate),
                    cell_gradient * previous_cell * forget_gate * (1.0 - forget_gate),
                    output_gate_gradient * output_gate * (1.0 - output_gate),
                    cell_gradient * input_gate * (1.0 - candidate**2),
                ),
                axis=1,
            )
            cell_gradient_next = cell_gradient * forget_gate
            grad_weights += recurrent_input.T @ gate_gradients
            grad_bias += gate_gradients.sum(axis=0)
            hidden_gradient_next = gate_gradients @ weights[input_count:, :].T

        grad_weights += 2.0 * self.config.l2 * weights
        grad_out += 2.0 * self.config.l2 * weights_out
        return [grad_weights, grad_bias, grad_out, grad_bias_out, grad_skip]

    def _initial_parameters(self, rng: np.random.Generator, input_count: int) -> list[np.ndarray]:
        """Glorot-type LSTM weights, forget-gate bias, zero skip weights."""
        hidden_count = self.config.hidden_units
        bias = np.zeros(4 * hidden_count)
        bias[hidden_count : 2 * hidden_count] = self.config.forget_gate_bias
        return [
            rng.normal(
                0.0,
                math.sqrt(1.0 / (input_count + hidden_count)),
                (input_count + hidden_count, 4 * hidden_count),
            ),
            bias,
            rng.normal(0.0, math.sqrt(2.0 / (hidden_count + 1)), (hidden_count, 1)),
            np.zeros(1),
            np.zeros(self.skip_mask.shape),
        ]

    def _clip(self, gradients: list[np.ndarray]) -> list[np.ndarray]:
        """Rescale all gradients together when their global L2 norm exceeds the cap."""
        clip_norm = self.config.gradient_clip_norm
        if clip_norm is None:
            return gradients
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients))
        return gradients if norm <= clip_norm else [g * (clip_norm / norm) for g in gradients]

    def _batches(self, sample_count: int, rng: np.random.Generator) -> list:
        batch_size = self.config.batch_size
        if batch_size is None or batch_size >= sample_count:
            return [slice(None)]
        order = rng.permutation(sample_count)
        return [order[start : start + batch_size] for start in range(0, sample_count, batch_size)]

    def _train(
        self,
        sequence: np.ndarray,
        skip: np.ndarray,
        targets: np.ndarray,
        epochs: int,
        learning_rate: float,
        seed: int,
        *,
        initial_parameters: list[np.ndarray] | None = None,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], int]:
        """Adam with mini-batches, stepwise decay and clipping; early stopping
        on the validation MAE when ``validation`` is given."""
        rng = np.random.default_rng(seed)
        if initial_parameters is None:
            parameters = self._initial_parameters(rng, sequence.shape[2])
        else:
            if len(initial_parameters) != PARAMETER_COUNT:
                raise ValueError("Initial parameters do not match the LSTM model.")
            parameters = [value.copy() for value in initial_parameters]

        first_moments = [np.zeros_like(value) for value in parameters]
        second_moments = [np.zeros_like(value) for value in parameters]
        beta_1, beta_2 = 0.9, 0.999
        adam_step = 0
        best_parameters: list[np.ndarray] | None = None
        best_loss, best_epoch, stale_epochs = float("inf"), 0, 0

        for epoch in range(1, epochs + 1):
            epoch_rate = self.config.learning_rate_at(learning_rate, epoch)
            for batch in self._batches(len(targets), rng):
                gradients = self._clip(
                    self._gradients(sequence[batch], skip[batch], targets[batch], parameters)
                )
                adam_step += 1
                for index, gradient in enumerate(gradients):
                    first_moments[index] = beta_1 * first_moments[index] + (1.0 - beta_1) * gradient
                    second_moments[index] = (
                        beta_2 * second_moments[index] + (1.0 - beta_2) * gradient * gradient
                    )
                    corrected_first = first_moments[index] / (1.0 - beta_1**adam_step)
                    corrected_second = second_moments[index] / (1.0 - beta_2**adam_step)
                    parameters[index] -= epoch_rate * corrected_first / (
                        np.sqrt(corrected_second) + 1e-8
                    )

            if validation is not None:
                predictions, _ = self._forward(validation[0], validation[1], parameters)
                loss = float(np.mean(np.abs(predictions - validation[2])))
                if not np.isfinite(loss):
                    raise FloatingPointError("LSTM validation loss became non-finite.")
                if loss < best_loss:
                    best_loss, best_epoch, stale_epochs = loss, epoch, 0
                    best_parameters = [value.copy() for value in parameters]
                else:
                    stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if validation is None:
            return parameters, epochs
        if best_parameters is None:
            raise RuntimeError("LSTM training did not produce a valid validation loss.")
        return best_parameters, best_epoch

    def _phase(self, sequence, skip, targets, learning_rate, initial_parameters=None):
        """Select the epoch count on a chronological hold-out, then refit on all days."""
        split = len(targets) - max(1, int(math.ceil(len(targets) * self.config.validation_fraction)))
        if split < 2:
            raise ValueError("Training phase is too short for validation.")
        _, epochs = self._train(
            sequence[:split], skip[:split], targets[:split], self.config.max_epochs,
            learning_rate, self.config.random_seed, initial_parameters=initial_parameters,
            validation=(sequence[split:], skip[split:], targets[split:]),
        )
        parameters, _ = self._train(
            sequence, skip, targets, epochs, learning_rate, self.config.random_seed,
            initial_parameters=initial_parameters,
        )
        return parameters, epochs

    def fit(self, sequence: np.ndarray, skip: np.ndarray, targets: np.ndarray) -> "LSTMNetwork":
        """Two phases: older days at ``learning_rate``, then the most recent
        ``fine_tune_fraction`` of the days at ``fine_tune_learning_rate``."""
        if sequence.ndim != 3 or targets.shape != sequence.shape[:2]:
            raise ValueError("LSTM inputs and targets must share day and MTU axes.")
        if self.skip_mask.shape != (skip.shape[1], targets.shape[1]):
            raise ValueError("Skip-connection mask does not match the skip inputs.")
        phase_2_count = max(3, int(math.ceil(len(targets) * self.config.fine_tune_fraction)))
        phase_1_count = len(targets) - phase_2_count
        if phase_1_count < 3:
            raise ValueError("Not enough training days for the two model phases.")
        first, second = slice(None, phase_1_count), slice(phase_1_count, None)
        phase_1, epochs_1 = self._phase(
            sequence[first], skip[first], targets[first], self.config.learning_rate
        )
        self.parameters, epochs_2 = self._phase(
            sequence[second], skip[second], targets[second],
            self.config.fine_tune_learning_rate, initial_parameters=phase_1,
        )
        self.selected_epochs = (epochs_1, epochs_2)
        return self

    def predict(
        self,
        inputs: tuple[np.ndarray, np.ndarray],
        parameters: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        parameters = self.parameters if parameters is None else parameters
        if not parameters:
            raise RuntimeError("LSTM must be fitted before calling predict.")
        predictions, _ = self._forward(inputs[0], inputs[1], parameters)
        if not np.isfinite(predictions).all():
            raise FloatingPointError("LSTM produced non-finite predictions.")
        return predictions


# ====================================================================
# Model state and checkpoint store
# ====================================================================
@dataclass
class ModelState:
    """Parameters plus the frozen scaling; ``covers_through`` is the last
    delivery day whose prices the model has been trained on."""

    covers_through: date
    parameters: list[np.ndarray]
    scaler: FrozenFeatureScaler
    columns: list[str]
    skip_columns: list[str]
    target_centers: np.ndarray
    target_scales: np.ndarray
    hidden_units: int
    metadata: dict = field(default_factory=dict)

    def network(self, config: CustomLossLSTMConfig) -> LSTMNetwork:
        network = LSTMNetwork(
            replace(config.pretrain, hidden_units=self.hidden_units),
            config.loss,
            skip_mask(self.skip_columns),
        )
        network.parameters = [value.copy() for value in self.parameters]
        return network

    def with_parameters(self, parameters, covers_through: date, metadata: dict) -> "ModelState":
        return replace(
            self,
            covers_through=covers_through,
            parameters=[value.copy() for value in parameters],
            metadata=metadata,
        )

    def inputs(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(LSTM sequence [day, MTU, channel], skip inputs [day, column])."""
        scaled = self.scaler.transform(X)
        return (
            _sequence_feature_tensor(scaled, self.columns),
            scaled[self.skip_columns].to_numpy(dtype=float),
        )

    def targets(self, Y: pd.DataFrame) -> np.ndarray:
        return np.arcsinh((Y.to_numpy(dtype=float) - self.target_centers) / self.target_scales)

    def to_prices(self, transformed: np.ndarray) -> np.ndarray:
        return self.target_centers + self.target_scales * np.sinh(transformed)


UpdateRule = Callable[..., ModelState]


def lstm_columns_for(columns) -> list[str]:
    """LSTM channels: every feature except the EPEX price lags (skip branch only)."""
    return [c for c in columns if not _EPEX_LAG_COLUMN.fullmatch(c)]


def skip_columns_for(columns) -> list[str]:
    """Skip-branch columns: EPEX d-1/d-2/d-7, EXAA (if present), seven-day mean."""
    price_columns = [c for c in columns if c != SEVEN_DAY_AVERAGE_PRICE_COLUMN]
    if any(c.startswith("exaa_") for c in price_columns):
        lags = _column_groups_exaa(price_columns)[1]
    else:
        lags = _column_groups(price_columns)[1]
    return [*lags, SEVEN_DAY_AVERAGE_PRICE_COLUMN]


def skip_mask(skip_columns) -> np.ndarray:
    """Local windows for the price lags, an unrestricted link for the mean."""
    lags = [c for c in skip_columns if c != SEVEN_DAY_AVERAGE_PRICE_COLUMN]
    if any(c.startswith("exaa_") for c in lags):
        mask = _skip_connection_mask_exaa(lags)
    else:
        mask = _skip_connection_mask(lags)
    return np.vstack((mask, np.ones((1, MTUS_PER_DAY), dtype=bool)))


class ModelStore:
    """Pretrained model plus one checkpoint per learned delivery day."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _paths(self, name: str) -> tuple[Path, Path]:
        base = self.root / name if name == PRETRAINED_NAME else self.root / CHECKPOINT_DIR / name
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def _save(self, name: str, state: ModelState, *, frozen: bool) -> None:
        npz_path, json_path = self._paths(name)
        arrays = {f"p{i}": value for i, value in enumerate(state.parameters)}
        payload = {
            "format_version": STATE_FORMAT_VERSION,
            "covers_through": state.covers_through.isoformat(),
            "metadata": state.metadata,
        }
        if frozen:
            arrays.update(target_centers=state.target_centers, target_scales=state.target_scales)
            payload.update(
                hidden_units=state.hidden_units,
                columns=state.columns,
                skip_columns=state.skip_columns,
                scaler=state.scaler.to_json(),
            )

        def write_npz(path: Path) -> None:
            with path.open("wb") as handle:
                np.savez(handle, **arrays)

        _atomic_write(npz_path, write_npz)
        _atomic_write(
            json_path,
            lambda path: path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"),
        )

    @staticmethod
    def _parameters(npz_path: Path) -> list[np.ndarray]:
        with np.load(npz_path) as arrays:
            return [arrays[f"p{i}"].copy() for i in range(PARAMETER_COUNT)]

    def save_pretrained(self, state: ModelState) -> None:
        self._save(PRETRAINED_NAME, state, frozen=True)
        # A new pretrained model invalidates checkpoints derived from an old one.
        for path in (self.root / CHECKPOINT_DIR).glob("covers_*"):
            path.unlink()

    def save_checkpoint(self, state: ModelState) -> None:
        self._save(f"covers_{state.covers_through.isoformat()}", state, frozen=False)

    def has_pretrained(self) -> bool:
        return all(path.is_file() for path in self._paths(PRETRAINED_NAME))

    def load_pretrained(self) -> ModelState:
        npz_path, json_path = self._paths(PRETRAINED_NAME)
        if not self.has_pretrained():
            raise FileNotFoundError(f"No pretrained custom-loss LSTM in {self.root}.")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != STATE_FORMAT_VERSION:
            raise ValueError(f"Unsupported model format in {json_path}; pretrain again.")
        with np.load(npz_path) as arrays:
            centers = arrays["target_centers"].copy()
            scales = arrays["target_scales"].copy()
        return ModelState(
            covers_through=date.fromisoformat(payload["covers_through"]),
            parameters=self._parameters(npz_path),
            scaler=FrozenFeatureScaler.from_json(payload["scaler"]),
            columns=list(payload["columns"]),
            skip_columns=list(payload["skip_columns"]),
            target_centers=centers,
            target_scales=scales,
            hidden_units=int(payload["hidden_units"]),
            metadata=payload["metadata"],
        )

    def checkpoint_dates(self) -> list[date]:
        directory = self.root / CHECKPOINT_DIR
        return sorted(
            date.fromisoformat(path.stem.removeprefix("covers_"))
            for path in directory.glob("covers_*.npz")
            if path.with_suffix(".json").is_file()
        ) if directory.is_dir() else []

    def load_latest(self, *, not_after: date) -> ModelState:
        """Latest state covering at most ``not_after``."""
        pretrained = self.load_pretrained()
        if pretrained.covers_through > not_after:
            raise ValueError(
                f"The pretrained model already covers {pretrained.covers_through}, "
                f"after {not_after}; a forecast for that target would use future data."
            )
        candidates = [d for d in self.checkpoint_dates() if pretrained.covers_through < d <= not_after]
        if not candidates:
            return pretrained
        npz_path, json_path = self._paths(f"covers_{candidates[-1].isoformat()}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return pretrained.with_parameters(self._parameters(npz_path), candidates[-1], payload["metadata"])


# ====================================================================
# Training helpers for the daily updates
# ====================================================================
def complete_days(X: pd.DataFrame, Y: pd.DataFrame, first: date, last: date) -> pd.Index:
    """Rows of the complete delivery days ``first .. last`` (inclusive)."""
    index = [i for i in X.index if first <= _as_date(i) <= last]
    return _complete_rows(X.loc[index], Y) if index else pd.Index([])


def train_from(
    state: ModelState,
    rows: pd.Index,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    covers_through: date,
    config: CustomLossLSTMConfig,
) -> tuple[list[np.ndarray], float, float]:
    """One daily update on ``rows`` starting from ``state``; returns the new
    parameters and the custom loss on ``rows`` before and after."""
    inputs = state.inputs(X.loc[rows])
    targets = state.targets(Y.loc[rows])
    network = state.network(config)
    parameters, _ = network._train(
        *inputs,
        targets,
        config.update_epochs,
        config.update_learning_rate,
        config.pretrain.random_seed + int(covers_through.toordinal()),
        initial_parameters=state.parameters,
    )
    before = custom_loss(network.predict(inputs, state.parameters), targets, config.loss)
    after = custom_loss(network.predict(inputs, parameters), targets, config.loss)
    return parameters, before, after


def rmse_on(
    parameters: list[np.ndarray],
    state: ModelState,
    rows: pd.Index,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    config: CustomLossLSTMConfig,
) -> float:
    """RMSE in EUR/MWh on ``rows`` (plain inverse transform, no bias correction)."""
    transformed = state.network(config).predict(state.inputs(X.loc[rows]), parameters)
    errors = state.to_prices(transformed) - Y.loc[rows].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(errors**2)))


# ====================================================================
# Pretraining, daily advance, forecasting
# ====================================================================
def pretrain_state(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    config: CustomLossLSTMConfig = CustomLossLSTMConfig(),
    *,
    metadata: dict | None = None,
) -> ModelState:
    """Fit scaling and the LSTM (custom loss) once on all complete pretraining days."""
    if list(Y.columns) != list(range(MTUS_PER_DAY)):
        raise ValueError("Y must contain target columns 0 through 95 in order.")
    if SEVEN_DAY_AVERAGE_PRICE_COLUMN not in X.columns:
        raise ValueError("LSTM features must include the seven-day average DA price.")
    rows = _complete_rows(X, Y)
    if len(rows) < config.pretrain.min_training_days:
        raise ValueError(f"Only {len(rows)} complete pretraining days available.")
    X_train, Y_train = X.loc[rows], Y.loc[rows]

    raw = Y_train.to_numpy(dtype=float)
    centers, scales = np.empty(MTUS_PER_DAY), np.empty(MTUS_PER_DAY)
    for mtu in range(MTUS_PER_DAY):
        centers[mtu], scales[mtu] = robust_params(raw[:, mtu])

    state = ModelState(
        covers_through=_as_date(rows[-1]),
        parameters=[],
        scaler=FrozenFeatureScaler.fit(X_train),
        columns=lstm_columns_for(X.columns),
        skip_columns=skip_columns_for(X.columns),
        target_centers=centers,
        target_scales=scales,
        hidden_units=config.pretrain.hidden_units,
    )
    inputs = state.inputs(X_train)
    targets = state.targets(Y_train)
    network = state.network(config).fit(*inputs, targets)
    state.parameters = [value.copy() for value in network.parameters]
    state.metadata = {
        "kind": "pretrained",
        "model": "lstm_custom_loss",
        "pretrain_first_day": _as_date(rows[0]).isoformat(),
        "pretrain_last_day": _as_date(rows[-1]).isoformat(),
        "pretrain_days": int(len(rows)),
        "input_channels": int(inputs[0].shape[2]),
        "skip_links_per_output": int(network.skip_mask.sum(axis=0).max()),
        "selected_epochs": list(network.selected_epochs),
        "train_loss": custom_loss_components(network.predict(inputs), targets, config.loss),
        "config": config.to_dict(),
        **(metadata or {}),
    }
    return state


def advance(
    store: ModelStore,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_date: date,
    config: CustomLossLSTMConfig,
    update: UpdateRule,
    *,
    day_metadata: Mapping[date, dict] | None = None,
) -> tuple[ModelState, list[dict]]:
    """Apply ``update`` day by day until the state covers ``target_date - 1``.

    One checkpoint per day, so missed days are caught up in order and a rerun
    for the same target reproduces the same model.
    """
    goal = target_date - timedelta(days=1)
    state = store.load_latest(not_after=goal)
    steps = (goal - state.covers_through).days
    if steps > config.max_catch_up_days:
        raise RuntimeError(
            f"{steps} days to catch up (state covers {state.covers_through}); more than "
            f"max_catch_up_days={config.max_catch_up_days}. Pretrain again instead."
        )
    log: list[dict] = []
    for _ in range(steps):
        day = state.covers_through + timedelta(days=1)
        state = update(state, X, Y, day, config, metadata=dict((day_metadata or {}).get(day, {})))
        store.save_checkpoint(state)
        log.append({"covers_through": day.isoformat(), **state.metadata})
    return state, log


def forecast_state(
    state: ModelState,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_date: date,
    config: CustomLossLSTMConfig = CustomLossLSTMConfig(),
) -> tuple[pd.DataFrame, dict]:
    """Forecast one delivery day with bias correction from recent residuals."""
    if state.covers_through >= target_date:
        raise ValueError(
            f"State covers {state.covers_through}, which is not before target {target_date}."
        )
    target_rows = [i for i in X.index if _as_date(i) == target_date]
    if not target_rows:
        raise ValueError(f"No feature row for target day {target_date}.")
    X_target = X.loc[target_rows]
    if not np.isfinite(X_target[state.scaler.columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Forecast features are incomplete for {target_date}.")

    network = state.network(config)
    transformed = network.predict(state.inputs(X_target))[0]
    bias_rows = complete_days(
        X, Y, target_date - timedelta(days=config.bias_window_days), target_date - timedelta(days=1)
    )
    if len(bias_rows):
        residuals = state.targets(Y.loc[bias_rows]) - network.predict(state.inputs(X.loc[bias_rows]))
    else:
        residuals = np.empty((0, MTUS_PER_DAY))
    prediction = np.asarray([
        inverse_vst_bias_corrected(
            [transformed[mtu]], residuals[:, mtu], state.target_centers[mtu], state.target_scales[mtu]
        )[0]
        for mtu in range(MTUS_PER_DAY)
    ])
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"Custom-loss LSTM forecast is non-finite for {target_date}.")

    actual = (
        Y.loc[target_rows[0]].to_numpy(dtype=float)
        if target_rows[0] in Y.index
        else np.full(MTUS_PER_DAY, np.nan)
    )
    forecast = pd.DataFrame(
        np.column_stack((prediction, actual)),
        index=make_delivery_index(pd.DatetimeIndex([pd.Timestamp(target_date)])),
        columns=["y_pred", "y_true"],
    )
    return forecast, {"bias_correction_days": int(len(bias_rows))}


__all__ = [
    "MODEL_ROOT",
    "CustomLoss",
    "CustomLossLSTMConfig",
    "LSTMNetwork",
    "ModelState",
    "ModelStore",
    "advance",
    "complete_days",
    "custom_loss",
    "custom_loss_components",
    "custom_loss_gradient",
    "forecast_state",
    "pretrain_state",
    "rmse_on",
    "train_from",
]
