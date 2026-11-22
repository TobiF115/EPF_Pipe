"""ERA5-pretrained LSTM with daily shifting-window fine-tuning.

Workflow
--------
1. ``pretrain_state``: train the one-layer LSTM once on ERA5 weather from 2025
   on. With EXAA columns in the features it is exactly the architecture of
   ``LSTM_exaa_enriched``; without them that of ``LSTM_fundamental``.
   Feature scaling and the per-MTU target transform are fitted here and then
   frozen, so every later update sees inputs on the same scale and the input
   dimension never changes.
2. ``advance_state``: every day, starting from the latest checkpoint, the
   model is fine-tuned on the most recent ``finetune_window_days`` complete
   delivery days (ICON-D2 weather, ERA5 where ICON-D2 is missing). One
   checkpoint is stored per learned day, so missed days are caught up in
   order and a rerun for the same target reproduces the same model.
3. ``forecast_state``: predict the target day; the arcsinh bias correction
   uses the current model's residuals on the most recent days.

A state is identified by ``covers_through``: the last delivery day whose
prices the model has been trained on. A forecast for target day ``T`` uses
the state covering ``T - 1``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.delivery_index import MTUS_PER_DAY, make_delivery_index
from pipeline.lear.lear_model import inverse_vst_bias_corrected, robust_params
from pipeline.mlp.LSTM_exaa_enriched import (
    LSTMConfig,
    SEVEN_DAY_AVERAGE_PRICE_COLUMN,
    _OneLayerLSTMNetwork,
    _sequence_feature_tensor,
    _skip_connection_mask_with_average_price,
)
from pipeline.mlp.LSTM_fundamental import _fundamental_skip_connection_mask
from pipeline.mlp.NN_exaa_enriched import _column_groups_exaa
from pipeline.mlp.mlp import _column_groups

STATE_FORMAT_VERSION = 1
PRETRAINED_NAME = "pretrained"
CHECKPOINT_DIR = "checkpoints"


# ====================================================================
# Configuration
# ====================================================================
def era5_pretrain_config() -> LSTMConfig:
    """Pretraining settings of the ERA5 online LSTM.

    Unlike the original full-batch, constant-rate LSTMs: shuffled mini-batches
    of 32 days, the learning rate halved every 10 epochs (validation optima sit
    at epochs 15-21, so the decay acts before early stopping; patience 40 leaves
    room to profit from a decay), forget-gate bias 1 so the cell starts out
    retaining its state, and global gradient-norm clipping at 5 (about twice the
    typical 32-day batch norm of ~2.8, so it only catches spikes).
    """
    return LSTMConfig(
        batch_size=32,
        lr_decay_step_epochs=10,
        lr_decay_factor=0.5,
        forget_gate_bias=1.0,
        gradient_clip_norm=5.0,
    )


@dataclass(frozen=True)
class OnlineLSTMConfig:
    """Pretraining and daily fine-tuning settings."""

    pretrain: LSTMConfig = field(default_factory=era5_pretrain_config)
    finetune_window_days: int = 7
    min_finetune_days: int = 5
    # Daily updates use a very small learning rate (300x below pretraining) so
    # the model drifts slowly and does not overfit a single week. With Adam,
    # each weight moves by roughly at most epochs * learning_rate per day.
    finetune_epochs: int = 10
    finetune_learning_rate: float = 1e-5
    bias_window_days: int = 28
    max_catch_up_days: int = 120

    def __post_init__(self) -> None:
        if self.finetune_window_days < 1 or self.min_finetune_days < 1:
            raise ValueError("Fine-tuning window sizes must be positive.")
        if self.min_finetune_days > self.finetune_window_days:
            raise ValueError("min_finetune_days cannot exceed finetune_window_days.")
        if self.finetune_epochs < 1 or self.finetune_learning_rate <= 0:
            raise ValueError("finetune_epochs and finetune_learning_rate must be positive.")
        if self.bias_window_days < 1 or self.max_catch_up_days < 1:
            raise ValueError("bias_window_days and max_catch_up_days must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "OnlineLSTMConfig":
        """Defaults, overridable via ``LSTM_ERA5_FINETUNE_LEARNING_RATE``,
        ``LSTM_ERA5_FINETUNE_EPOCHS`` and ``LSTM_ERA5_FINETUNE_WINDOW_DAYS``."""
        env = os.environ if environ is None else environ
        overrides: dict = {}
        if env.get("LSTM_ERA5_FINETUNE_LEARNING_RATE"):
            overrides["finetune_learning_rate"] = float(env["LSTM_ERA5_FINETUNE_LEARNING_RATE"])
        if env.get("LSTM_ERA5_FINETUNE_EPOCHS"):
            overrides["finetune_epochs"] = int(env["LSTM_ERA5_FINETUNE_EPOCHS"])
        if env.get("LSTM_ERA5_FINETUNE_WINDOW_DAYS"):
            window = int(env["LSTM_ERA5_FINETUNE_WINDOW_DAYS"])
            overrides["finetune_window_days"] = window
            overrides["min_finetune_days"] = min(cls.min_finetune_days, window)
        return cls(**overrides)


# ====================================================================
# Frozen feature scaling (mirrors lear_model.scale_fold_point)
# ====================================================================
@dataclass
class FrozenFeatureScaler:
    """Per-column transforms fitted once on the pretraining rows.

    Reproduces ``scale_fold_point`` exactly on the data it was fitted on:
    arcsinh-robust scaling for price lags, robust scaling for wind and load,
    zeroing of degenerate solar columns, min-max scaling for the remaining
    continuous columns, and identity for calendar dummies.
    """

    transforms: dict[str, tuple]

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        *,
        ssrd_filter_min_range: float = 20.0,
        ssrd_filter_min_pos_share: float = 0.50,
    ) -> "FrozenFeatureScaler":
        cols = X.columns.tolist()
        price_cols = [c for c in cols if c.startswith(("price_d", "exaa_d"))]
        time_cols = [
            c for c in cols
            if c.startswith("weekday_") or c in ["is_15min_market", "is_holiday"]
        ]
        ssrd_cols = [c for c in cols if "ssrd" in c.lower()]
        sw_dir_cols = [c for c in cols if "sw_dir" in c.lower()]
        sw_dif_cols = [c for c in cols if "sw_dif" in c.lower()]
        wind_cols = [c for c in cols if "wind_speed" in c.lower()]
        load_cols = [c for c in cols if "load" in c.lower()]
        categorized = set(
            price_cols + time_cols + ssrd_cols + sw_dir_cols + sw_dif_cols + wind_cols + load_cols
        )
        other_cols = [c for c in cols if c not in categorized]

        work = X.astype(float).copy()
        degenerate = set()
        for col in ssrd_cols + sw_dir_cols + sw_dif_cols:
            values = work[col]
            if (values.max() - values.min() < ssrd_filter_min_range) or (
                (values > 0).mean() < ssrd_filter_min_pos_share
            ):
                degenerate.add(col)
        for col in degenerate:
            work[col] = 0.0

        transforms: dict[str, tuple] = {col: ("identity",) for col in cols}
        for col in price_cols:
            transforms[col] = ("asinh_robust", *robust_params(work[col].values))
        for col in wind_cols:
            transforms[col] = ("robust", *robust_params(work[col].values))
        for col in load_cols:
            transforms[col] = ("robust", *robust_params(work[col].values))
        for col in ssrd_cols + sw_dir_cols + sw_dif_cols + other_cols:
            if col in degenerate:
                transforms[col] = ("zero",)
                continue
            low, high = float(work[col].min()), float(work[col].max())
            span = high - low
            transforms[col] = ("minmax", low, span if span > 0 else 1.0)
        return cls(transforms)

    @property
    def columns(self) -> list[str]:
        return list(self.transforms)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.transforms if c not in X.columns]
        if missing:
            raise ValueError(f"Features lack columns seen in pretraining: {missing[:5]} ...")
        out: dict[str, np.ndarray] = {}
        for col, spec in self.transforms.items():
            values = X[col].to_numpy(dtype=float)
            kind = spec[0]
            if kind == "asinh_robust":
                out[col] = np.arcsinh((values - spec[1]) / spec[2])
            elif kind == "robust":
                out[col] = (values - spec[1]) / spec[2]
            elif kind == "minmax":
                out[col] = (values - spec[1]) / spec[2]
            elif kind == "zero":
                out[col] = np.where(np.isfinite(values), 0.0, np.nan)
            else:
                out[col] = values
        return pd.DataFrame(out, index=X.index)

    def to_json(self) -> dict:
        return {col: list(spec) for col, spec in self.transforms.items()}

    @classmethod
    def from_json(cls, payload: Mapping[str, Sequence]) -> "FrozenFeatureScaler":
        return cls({col: tuple(spec) for col, spec in payload.items()})


# ====================================================================
# Model state
# ====================================================================
@dataclass
class OnlineLSTMState:
    covers_through: date
    parameters: list[np.ndarray]
    scaler: FrozenFeatureScaler
    hidden_columns: list[str]
    skip_columns: list[str]
    target_centers: np.ndarray
    target_scales: np.ndarray
    hidden_units: int
    metadata: dict = field(default_factory=dict)

    def network(self, training: LSTMConfig | None = None) -> _OneLayerLSTMNetwork:
        """Network with the stored weights; ``training`` supplies optimiser options."""
        skip_mask = _skip_mask(self.skip_columns)
        base = training if training is not None else LSTMConfig()
        network = _OneLayerLSTMNetwork(replace(base, hidden_units=self.hidden_units), skip_mask)
        network.parameters = [value.copy() for value in self.parameters]
        return network

    def with_parameters(self, parameters, covers_through: date, metadata: dict) -> "OnlineLSTMState":
        return OnlineLSTMState(
            covers_through=covers_through,
            parameters=[value.copy() for value in parameters],
            scaler=self.scaler,
            hidden_columns=self.hidden_columns,
            skip_columns=self.skip_columns,
            target_centers=self.target_centers,
            target_scales=self.target_scales,
            hidden_units=self.hidden_units,
            metadata=metadata,
        )


def _uses_exaa(columns: Sequence[str]) -> bool:
    return any(column.startswith("exaa_") for column in columns)


def _split_columns(columns: Sequence[str]) -> tuple[list[str], list[str]]:
    """Sequence and skip inputs exactly as in LSTM_exaa_enriched / LSTM_fundamental.

    With EXAA columns, the 96 target-day EXAA prices feed both the LSTM and
    skip links to the same MTU and its two cyclic neighbours on each side.
    """
    columns = list(columns)
    if _uses_exaa(columns):
        hidden_columns, skip_columns = _column_groups_exaa(columns)
    else:
        hidden_columns, skip_columns = _column_groups(columns)
    if SEVEN_DAY_AVERAGE_PRICE_COLUMN not in hidden_columns:
        raise ValueError("LSTM features must include the seven-day average DA price.")
    return hidden_columns, [*skip_columns, SEVEN_DAY_AVERAGE_PRICE_COLUMN]


def _skip_mask(skip_columns: Sequence[str]) -> np.ndarray:
    base = list(skip_columns)[:-1]  # last entry is the seven-day average
    if _uses_exaa(base):
        return _skip_connection_mask_with_average_price(base)
    return _fundamental_skip_connection_mask(base)


def _complete_rows(X: pd.DataFrame, Y: pd.DataFrame | None = None) -> pd.Index:
    mask = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    if Y is not None:
        y = Y.reindex(X.index).to_numpy(dtype=float)
        mask &= np.isfinite(y).all(axis=1)
    return X.index[mask]


def _model_inputs(state: OnlineLSTMState, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaled = state.scaler.transform(X)
    return (
        _sequence_feature_tensor(scaled, state.hidden_columns),
        scaled[state.skip_columns].to_numpy(dtype=float),
    )


def _transform_targets(state: OnlineLSTMState, Y: pd.DataFrame) -> np.ndarray:
    return np.arcsinh((Y.to_numpy(dtype=float) - state.target_centers) / state.target_scales)


def _as_date(value) -> date:
    return pd.Timestamp(value).date()


# ====================================================================
# Pretraining, fine-tuning, forecasting
# ====================================================================
def pretrain_state(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    config: OnlineLSTMConfig = OnlineLSTMConfig(),
    *,
    metadata: dict | None = None,
) -> OnlineLSTMState:
    """Fit scaling and the LSTM once on all complete pretraining days."""
    if list(Y.columns) != list(range(MTUS_PER_DAY)):
        raise ValueError("Y must contain target columns 0 through 95 in order.")
    rows = _complete_rows(X, Y)
    if len(rows) < config.pretrain.min_training_days:
        raise ValueError(f"Only {len(rows)} complete pretraining days available.")
    X_train, Y_train = X.loc[rows], Y.loc[rows]

    hidden_columns, skip_columns = _split_columns(X.columns)
    scaler = FrozenFeatureScaler.fit(X_train)
    raw = Y_train.to_numpy(dtype=float)
    centers = np.empty(MTUS_PER_DAY)
    scales = np.empty(MTUS_PER_DAY)
    for mtu in range(MTUS_PER_DAY):
        centers[mtu], scales[mtu] = robust_params(raw[:, mtu])

    state = OnlineLSTMState(
        covers_through=_as_date(rows[-1]),
        parameters=[],
        scaler=scaler,
        hidden_columns=hidden_columns,
        skip_columns=skip_columns,
        target_centers=centers,
        target_scales=scales,
        hidden_units=config.pretrain.hidden_units,
    )
    sequence, skip = _model_inputs(state, X_train)
    network = _OneLayerLSTMNetwork(config.pretrain, _skip_mask(skip_columns))
    network.fit(sequence, skip, _transform_targets(state, Y_train))
    state.parameters = [value.copy() for value in network.parameters]
    state.metadata = {
        "kind": "pretrained",
        "pretrain_first_day": _as_date(rows[0]).isoformat(),
        "pretrain_last_day": _as_date(rows[-1]).isoformat(),
        "pretrain_days": int(len(rows)),
        "exaa_features": _uses_exaa(X.columns),
        "config": config.to_dict(),
        **(metadata or {}),
    }
    return state


def finetune_state(
    state: OnlineLSTMState,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    covers_through: date,
    config: OnlineLSTMConfig = OnlineLSTMConfig(),
    *,
    metadata: dict | None = None,
) -> OnlineLSTMState:
    """Fine-tune on the window ending at ``covers_through`` (inclusive)."""
    first = covers_through - timedelta(days=config.finetune_window_days - 1)
    in_window = [i for i in X.index if first <= _as_date(i) <= covers_through]
    rows = _complete_rows(X.loc[in_window], Y) if in_window else pd.Index([])
    info = {
        "kind": "finetuned",
        "window_first_day": first.isoformat(),
        "window_last_day": covers_through.isoformat(),
        "window_complete_days": int(len(rows)),
        "window_days_used": [_as_date(i).isoformat() for i in rows],
        **(metadata or {}),
    }
    if covers_through not in {_as_date(i) for i in rows}:
        # The newly covered day itself is incomplete (e.g. missing EXAA prices):
        # training again would only repeat days that earlier updates already used.
        info["kind"] = "skipped"
        info["reason"] = f"{covers_through} itself has incomplete inputs or prices"
        return state.with_parameters(state.parameters, covers_through, info)
    if len(rows) < config.min_finetune_days:
        info["kind"] = "skipped"
        info["reason"] = (
            f"only {len(rows)} complete days in window (< {config.min_finetune_days})"
        )
        return state.with_parameters(state.parameters, covers_through, info)

    sequence, skip = _model_inputs(state, X.loc[rows])
    targets = _transform_targets(state, Y.loc[rows])
    # Same clipping/batching as pretraining; a 7-day window is one batch and
    # 10 epochs never reach the first learning-rate step.
    network = state.network(config.pretrain)
    parameters, _ = network._train(
        sequence,
        skip,
        targets,
        config.finetune_epochs,
        config.finetune_learning_rate,
        config.pretrain.random_seed + int(covers_through.toordinal()),
        initial_parameters=state.parameters,
    )
    before = network._forward(sequence, skip, state.parameters)[0]
    after = network._forward(sequence, skip, parameters)[0]
    info["window_mse_before"] = float(np.mean((before - targets) ** 2))
    info["window_mse_after"] = float(np.mean((after - targets) ** 2))
    return state.with_parameters(parameters, covers_through, info)


def forecast_state(
    state: OnlineLSTMState,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_date: date,
    config: OnlineLSTMConfig = OnlineLSTMConfig(),
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

    network = state.network()
    sequence, skip = _model_inputs(state, X_target)
    transformed = network.predict(sequence, skip)[0]

    first_bias = target_date - timedelta(days=config.bias_window_days)
    bias_index = [i for i in X.index if first_bias <= _as_date(i) < target_date]
    bias_rows = _complete_rows(X.loc[bias_index], Y) if bias_index else pd.Index([])
    if len(bias_rows):
        b_seq, b_skip = _model_inputs(state, X.loc[bias_rows])
        residuals = _transform_targets(state, Y.loc[bias_rows]) - network.predict(b_seq, b_skip)
    else:
        residuals = np.empty((0, MTUS_PER_DAY))
    prediction = np.asarray([
        inverse_vst_bias_corrected(
            [transformed[mtu]], residuals[:, mtu], state.target_centers[mtu], state.target_scales[mtu]
        )[0]
        for mtu in range(MTUS_PER_DAY)
    ])
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"Online LSTM forecast is non-finite for {target_date}.")

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


# ====================================================================
# Persistence
# ====================================================================
def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class OnlineLSTMStore:
    """Pretrained model plus one checkpoint per learned delivery day."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _paths(self, name: str) -> tuple[Path, Path]:
        base = self.root / name if name == PRETRAINED_NAME else self.root / CHECKPOINT_DIR / name
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def _save(self, name: str, state: OnlineLSTMState, *, frozen: bool) -> None:
        npz_path, json_path = self._paths(name)
        arrays = {f"p{i}": value for i, value in enumerate(state.parameters)}
        if frozen:
            arrays["target_centers"] = state.target_centers
            arrays["target_scales"] = state.target_scales

        def write_npz(path: Path) -> None:
            with path.open("wb") as handle:
                np.savez(handle, **arrays)

        payload = {
            "format_version": STATE_FORMAT_VERSION,
            "covers_through": state.covers_through.isoformat(),
            "metadata": state.metadata,
        }
        if frozen:
            payload.update(
                hidden_units=state.hidden_units,
                hidden_columns=state.hidden_columns,
                skip_columns=state.skip_columns,
                scaler=state.scaler.to_json(),
            )
        _atomic_write(npz_path, write_npz)
        _atomic_write(
            json_path,
            lambda path: path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"),
        )

    def save_pretrained(self, state: OnlineLSTMState) -> None:
        self._save(PRETRAINED_NAME, state, frozen=True)
        # A new pretrained model invalidates checkpoints derived from an old one.
        checkpoint_dir = self.root / CHECKPOINT_DIR
        if checkpoint_dir.is_dir():
            for path in checkpoint_dir.glob("covers_*"):
                path.unlink()

    def save_checkpoint(self, state: OnlineLSTMState) -> None:
        self._save(f"covers_{state.covers_through.isoformat()}", state, frozen=False)

    def has_pretrained(self) -> bool:
        return all(path.is_file() for path in self._paths(PRETRAINED_NAME))

    def load_pretrained(self) -> OnlineLSTMState:
        npz_path, json_path = self._paths(PRETRAINED_NAME)
        if not self.has_pretrained():
            raise FileNotFoundError(
                f"No pretrained ERA5 LSTM in {self.root}. Run "
                "`python run_pipeline.py --lstm-era5-pretrain` first."
            )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != STATE_FORMAT_VERSION:
            raise ValueError(f"Unsupported model format in {json_path}.")
        with np.load(npz_path) as arrays:
            parameters = [arrays[f"p{i}"].copy() for i in range(5)]
            centers = arrays["target_centers"].copy()
            scales = arrays["target_scales"].copy()
        return OnlineLSTMState(
            covers_through=date.fromisoformat(payload["covers_through"]),
            parameters=parameters,
            scaler=FrozenFeatureScaler.from_json(payload["scaler"]),
            hidden_columns=list(payload["hidden_columns"]),
            skip_columns=list(payload["skip_columns"]),
            target_centers=centers,
            target_scales=scales,
            hidden_units=int(payload["hidden_units"]),
            metadata=payload["metadata"],
        )

    def checkpoint_dates(self) -> list[date]:
        directory = self.root / CHECKPOINT_DIR
        if not directory.is_dir():
            return []
        dates = []
        for path in directory.glob("covers_*.npz"):
            if path.with_suffix(".json").is_file():
                dates.append(date.fromisoformat(path.stem.removeprefix("covers_")))
        return sorted(dates)

    def load_latest(self, *, not_after: date) -> OnlineLSTMState:
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
        with np.load(npz_path) as arrays:
            parameters = [arrays[f"p{i}"].copy() for i in range(5)]
        return pretrained.with_parameters(parameters, candidates[-1], payload["metadata"])


def advance_state(
    store: OnlineLSTMStore,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_date: date,
    config: OnlineLSTMConfig = OnlineLSTMConfig(),
    *,
    day_metadata: Mapping[date, dict] | None = None,
) -> tuple[OnlineLSTMState, list[dict]]:
    """Fine-tune day by day until the state covers ``target_date - 1``."""
    goal = target_date - timedelta(days=1)
    state = store.load_latest(not_after=goal)
    steps = (goal - state.covers_through).days
    if steps > config.max_catch_up_days:
        raise RuntimeError(
            f"{steps} days to catch up (state covers {state.covers_through}); more than "
            f"max_catch_up_days={config.max_catch_up_days}. Pretrain again instead."
        )
    log: list[dict] = []
    for offset in range(1, steps + 1):
        day = state.covers_through + timedelta(days=1)
        state = finetune_state(
            state, X, Y, day, config,
            metadata=dict((day_metadata or {}).get(day, {})),
        )
        store.save_checkpoint(state)
        log.append({"covers_through": day.isoformat(), **state.metadata})
    return state, log


def first_required_feature_day(
    store: OnlineLSTMStore,
    target_date: date,
    config: OnlineLSTMConfig = OnlineLSTMConfig(),
) -> date:
    """Earliest delivery day whose features ``advance_state``/forecast need."""
    state = store.load_latest(not_after=target_date - timedelta(days=1))
    first_update = state.covers_through + timedelta(days=1)
    return min(
        first_update - timedelta(days=config.finetune_window_days - 1),
        target_date - timedelta(days=config.bias_window_days),
    )
