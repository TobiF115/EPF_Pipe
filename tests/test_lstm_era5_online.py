"""Tests for the ERA5-pretrained LSTM with daily shifting-window fine-tuning."""

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.lear.lear_model import scale_fold_point
from pipeline.mlp.LSTM_era5_online import (
    FrozenFeatureScaler,
    OnlineLSTMConfig,
    OnlineLSTMStore,
    advance_state,
    era5_pretrain_config,
    finetune_state,
    first_required_feature_day,
    forecast_state,
    pretrain_state,
)
from pipeline.mlp.LSTM_exaa_enriched import LSTMConfig, _OneLayerLSTMNetwork

TZ = "Europe/Berlin"
CONFIG = OnlineLSTMConfig(
    pretrain=LSTMConfig(hidden_units=4, max_epochs=3, patience=2, min_training_days=10),
    finetune_window_days=7,
    min_finetune_days=5,
    finetune_epochs=5,
    bias_window_days=7,
    max_catch_up_days=30,
)


def _matrices(day_count=45, seed=0, exaa=False):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=day_count, freq="D", tz=TZ)
    columns = {}
    for lag in (1, 2, 7):
        for mtu in range(96):
            columns[f"price_d{lag}_mtu_{mtu:02d}"] = rng.normal(50 + lag, 10, day_count)
    if exaa:
        for mtu in range(96):
            columns[f"exaa_d0_mtu_{mtu:02d}"] = rng.normal(60, 12, day_count)
    for hour in range(24):
        columns[f"wind_speed_cluster_0_h{hour:02d}"] = rng.normal(5, 2, day_count)
    daylight = np.clip(np.sin((np.arange(96) / 4 - 6) / 12 * np.pi), 0, None)
    for name, peak in (("sw_dir_cluster_0", 400.0), ("sw_dif_cluster_0", 150.0)):
        for mtu in range(96):
            columns[f"{name}_mtu{mtu:02d}"] = peak * daylight[mtu] * rng.uniform(0.2, 1.0, day_count)
    for mtu in range(96):
        columns[f"load_d0_mtu{mtu:02d}"] = rng.normal(55000, 5000, day_count)
    columns["is_holiday"] = rng.integers(0, 2, day_count).astype(float)
    columns["average_da_price_last_7d"] = rng.normal(80, 5, day_count)
    X = pd.DataFrame(columns, index=index)
    anchor = "exaa_d0" if exaa else "price_d1"
    Y = pd.DataFrame(
        X[[f"{anchor}_mtu_{m:02d}" for m in range(96)]].to_numpy() * 0.8
        + rng.normal(10, 5, (day_count, 96)),
        index=index,
        columns=range(96),
    )
    return X, Y


class ConfigTests(unittest.TestCase):
    def test_default_finetune_learning_rate_is_far_below_pretraining(self):
        config = OnlineLSTMConfig()
        self.assertEqual(config.finetune_learning_rate, 1e-5)
        self.assertLessEqual(config.finetune_learning_rate * 100, config.pretrain.learning_rate)
        self.assertLess(config.finetune_learning_rate, config.pretrain.fine_tune_learning_rate)

    def test_env_overrides(self):
        config = OnlineLSTMConfig.from_env({
            "LSTM_ERA5_FINETUNE_LEARNING_RATE": "5e-6",
            "LSTM_ERA5_FINETUNE_EPOCHS": "4",
            "LSTM_ERA5_FINETUNE_WINDOW_DAYS": "3",
        })
        self.assertEqual(config.finetune_learning_rate, 5e-6)
        self.assertEqual(config.finetune_epochs, 4)
        self.assertEqual((config.finetune_window_days, config.min_finetune_days), (3, 3))
        self.assertEqual(OnlineLSTMConfig.from_env({}), OnlineLSTMConfig())

    def test_small_learning_rate_bounds_daily_weight_change(self):
        X, Y = _matrices(40)
        config = OnlineLSTMConfig(pretrain=CONFIG.pretrain)  # default 1e-5, 10 epochs
        state = pretrain_state(X.iloc[:30], Y.iloc[:30], config)
        updated = finetune_state(state, X, Y, X.index[34].date(), config)
        change = max(np.max(np.abs(a - b)) for a, b in zip(updated.parameters, state.parameters))
        self.assertGreater(change, 0.0)
        # Adam moves each weight by at most ~learning_rate per epoch.
        self.assertLessEqual(change, config.finetune_epochs * config.finetune_learning_rate * 1.01)


class EquivalenceWithExistingLSTMTests(unittest.TestCase):
    """Pretraining once must reproduce the existing rolling LSTMs exactly."""

    PRETRAIN = LSTMConfig(hidden_units=4, max_epochs=8, patience=3, min_training_days=10)

    def _assert_equivalent(self, exaa: bool, rolling_forecast):
        X, Y = _matrices(31, seed=3, exaa=exaa)
        target = X.index[30]
        expected = rolling_forecast(X, Y, [target], self.PRETRAIN)
        # A bias window covering all training days uses the same residuals.
        config = OnlineLSTMConfig(pretrain=self.PRETRAIN, bias_window_days=10_000)
        state = pretrain_state(X.iloc[:30], Y.iloc[:30], config)
        actual, _ = forecast_state(state, X, Y, target.date(), config)
        np.testing.assert_allclose(actual["y_pred"], expected["y_pred"], rtol=1e-9, atol=1e-9)
        return state

    def test_exaa_enriched_matches_rolling_lstm_exaa_enriched(self):
        from pipeline.mlp.LSTM_exaa_enriched import rolling_lstm_exaa_enriched_forecast

        state = self._assert_equivalent(True, rolling_lstm_exaa_enriched_forecast)
        self.assertTrue(state.metadata["exaa_features"])
        exaa_inputs = [c for c in state.skip_columns if c.startswith("exaa_")]
        self.assertEqual(len(exaa_inputs), 96)
        self.assertEqual(sum(c.startswith("exaa_") for c in state.hidden_columns), 96)

    def test_fundamental_matches_rolling_lstm_fundamental(self):
        from pipeline.mlp.LSTM_fundamental import rolling_lstm_fundamental_forecast

        state = self._assert_equivalent(False, rolling_lstm_fundamental_forecast)
        self.assertFalse(state.metadata["exaa_features"])

    def test_exaa_state_survives_save_and_finetune(self):
        X, Y = _matrices(40, seed=4, exaa=True)
        with tempfile.TemporaryDirectory() as tmp:
            store = OnlineLSTMStore(Path(tmp))
            store.save_pretrained(pretrain_state(X.iloc[:30], Y.iloc[:30], CONFIG))
            state, log = advance_state(store, X, Y, X.index[35].date(), CONFIG)
            self.assertEqual(len(log), 5)
            forecast, _ = forecast_state(state, X, Y, X.index[35].date(), CONFIG)
            self.assertTrue(np.isfinite(forecast["y_pred"]).all())


class FrozenScalerTests(unittest.TestCase):
    def test_matches_scale_fold_point_on_fit_data(self):
        X, Y = _matrices(30)
        expected, _, _, _, info = scale_fold_point(X, X.iloc[:2], pd.Series(Y[0].to_numpy()), None)
        scaler = FrozenFeatureScaler.fit(X)
        actual = scaler.transform(X)
        np.testing.assert_allclose(actual[expected.columns].to_numpy(), expected.to_numpy(), atol=1e-10)
        # Night-time solar columns are degenerate and become zero in both.
        self.assertIn("sw_dir_cluster_0_mtu00", info["degenerate_ssrd_cols"])
        self.assertEqual(scaler.transforms["sw_dir_cluster_0_mtu00"][0], "zero")

    def test_parameters_stay_frozen_for_new_data(self):
        X, _ = _matrices(30)
        scaler = FrozenFeatureScaler.fit(X.iloc[:20])
        restored = FrozenFeatureScaler.from_json(json.loads(json.dumps(scaler.to_json())))
        later = X.iloc[20:]
        np.testing.assert_allclose(
            restored.transform(later).to_numpy(), scaler.transform(later).to_numpy()
        )
        # A refit on the later window would differ: the transform is not refitted.
        self.assertFalse(np.allclose(
            FrozenFeatureScaler.fit(later).transform(later).to_numpy(),
            scaler.transform(later).to_numpy(),
        ))


class OnlineLSTMTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = OnlineLSTMStore(Path(self._tmp.name) / "model")
        self.X, self.Y = _matrices(45)
        self.days = [ts.date() for ts in self.X.index]
        self.pretrained = pretrain_state(self.X.iloc[:30], self.Y.iloc[:30], CONFIG)
        self.store.save_pretrained(self.pretrained)

    def tearDown(self):
        self._tmp.cleanup()

    def test_pretrained_round_trip_predicts_identically(self):
        loaded = self.store.load_pretrained()
        self.assertEqual(loaded.covers_through, self.days[29])
        a, _ = forecast_state(self.pretrained, self.X, self.Y, self.days[30], CONFIG)
        b, _ = forecast_state(loaded, self.X, self.Y, self.days[30], CONFIG)
        np.testing.assert_allclose(a["y_pred"], b["y_pred"])

    def test_catch_up_creates_one_checkpoint_per_day_with_shifting_window(self):
        target = self.days[40]
        state, log = advance_state(self.store, self.X, self.Y, target, CONFIG)
        self.assertEqual(state.covers_through, self.days[39])
        self.assertEqual(self.store.checkpoint_dates(), self.days[30:40])
        self.assertEqual([entry["covers_through"] for entry in log], [d.isoformat() for d in self.days[30:40]])
        first = log[0]
        self.assertEqual(first["window_first_day"], self.days[24].isoformat())
        self.assertEqual(first["window_last_day"], self.days[30].isoformat())
        self.assertEqual(first["window_complete_days"], 7)
        self.assertTrue(all(entry["kind"] == "finetuned" for entry in log))
        self.assertLessEqual(first["window_mse_after"], first["window_mse_before"])

    def test_rerun_for_same_target_is_idempotent(self):
        target = self.days[40]
        state, _ = advance_state(self.store, self.X, self.Y, target, CONFIG)
        forecast, _ = forecast_state(state, self.X, self.Y, target, CONFIG)
        state_again, log_again = advance_state(self.store, self.X, self.Y, target, CONFIG)
        self.assertEqual(log_again, [])
        forecast_again, _ = forecast_state(state_again, self.X, self.Y, target, CONFIG)
        np.testing.assert_allclose(forecast["y_pred"], forecast_again["y_pred"])

    def test_daily_runs_equal_one_catch_up(self):
        other = OnlineLSTMStore(Path(self._tmp.name) / "other")
        other.save_pretrained(self.pretrained)
        for day in self.days[31:41]:
            daily, _ = advance_state(other, self.X, self.Y, day, CONFIG)
        caught_up, _ = advance_state(self.store, self.X, self.Y, self.days[40], CONFIG)
        for a, b in zip(daily.parameters, caught_up.parameters):
            np.testing.assert_allclose(a, b)

    def test_update_changes_parameters_but_not_scaling(self):
        state, _ = advance_state(self.store, self.X, self.Y, self.days[32], CONFIG)
        self.assertFalse(np.allclose(state.parameters[0], self.pretrained.parameters[0]))
        self.assertEqual(state.scaler.transforms, self.pretrained.scaler.transforms)
        np.testing.assert_array_equal(state.target_centers, self.pretrained.target_centers)

    def test_window_with_too_few_complete_days_is_skipped(self):
        Y = self.Y.copy()
        Y.iloc[26:31] = np.nan  # leaves 2 complete days in the first window
        state = finetune_state(self.pretrained, self.X, Y, self.days[30], CONFIG)
        self.assertEqual(state.metadata["kind"], "skipped")
        for a, b in zip(state.parameters, self.pretrained.parameters):
            np.testing.assert_array_equal(a, b)

    def test_refuses_target_inside_pretraining_period(self):
        with self.assertRaises(ValueError):
            advance_state(self.store, self.X, self.Y, self.days[25], CONFIG)

    def test_refuses_too_long_catch_up(self):
        config = OnlineLSTMConfig(pretrain=CONFIG.pretrain, max_catch_up_days=3)
        with self.assertRaises(RuntimeError):
            advance_state(self.store, self.X, self.Y, self.days[40], config)

    def test_first_required_feature_day_covers_window_and_bias(self):
        first = first_required_feature_day(self.store, self.days[40], CONFIG)
        self.assertEqual(first, self.days[30] - timedelta(days=6))
        advance_state(self.store, self.X, self.Y, self.days[40], CONFIG)
        self.assertEqual(
            first_required_feature_day(self.store, self.days[41], CONFIG), self.days[34]
        )

    def test_forecast_shape_and_bias_window(self):
        state, _ = advance_state(self.store, self.X, self.Y, self.days[40], CONFIG)
        forecast, info = forecast_state(state, self.X, self.Y, self.days[40], CONFIG)
        self.assertEqual(len(forecast), 96)
        self.assertEqual(list(forecast.columns), ["y_pred", "y_true"])
        self.assertTrue(np.isfinite(forecast["y_pred"]).all())
        self.assertEqual(info["bias_correction_days"], 7)
        with self.assertRaises(ValueError):
            forecast_state(state, self.X, self.Y, self.days[39], CONFIG)

    def test_new_pretraining_discards_old_checkpoints(self):
        advance_state(self.store, self.X, self.Y, self.days[35], CONFIG)
        self.assertTrue(self.store.checkpoint_dates())
        self.store.save_pretrained(self.pretrained)
        self.assertEqual(self.store.checkpoint_dates(), [])


class IncompleteDayTests(unittest.TestCase):
    def test_no_update_when_the_newly_covered_day_is_incomplete(self):
        X, Y = _matrices(40)
        state = pretrain_state(X.iloc[:30], Y.iloc[:30], CONFIG)
        day = X.index[34].date()
        X_missing = X.copy()
        X_missing.loc[X.index[34], [c for c in X.columns if c.startswith("price_d1")][:1]] = np.nan
        updated = finetune_state(state, X_missing, Y, day, CONFIG)
        self.assertEqual(updated.metadata["kind"], "skipped")
        self.assertIn("itself", updated.metadata["reason"])
        self.assertEqual(updated.covers_through, day)
        for before, after in zip(state.parameters, updated.parameters):
            np.testing.assert_array_equal(before, after)
        # With the day complete, the same window is fine-tuned.
        self.assertEqual(finetune_state(state, X, Y, day, CONFIG).metadata["kind"], "finetuned")


class OptimiserOptionTests(unittest.TestCase):
    """Mini-batches, stepwise decay, forget-gate bias and gradient clipping."""

    @staticmethod
    def _network(**options):
        mask = np.zeros((3, 96), dtype=bool)
        mask[0] = True
        return _OneLayerLSTMNetwork(LSTMConfig(hidden_units=4, **options), mask)

    def test_era5_defaults_enable_all_options(self):
        config = OnlineLSTMConfig().pretrain
        self.assertEqual(config, era5_pretrain_config())
        self.assertEqual(config.batch_size, 32)
        self.assertEqual((config.lr_decay_step_epochs, config.lr_decay_factor), (10, 0.5))
        self.assertEqual(config.forget_gate_bias, 1.0)
        self.assertEqual(config.gradient_clip_norm, 5.0)

    def test_stepwise_learning_rate(self):
        config = LSTMConfig(lr_decay_step_epochs=30, lr_decay_factor=0.5)
        rates = [config.learning_rate_at(3e-3, epoch) for epoch in (1, 30, 31, 60, 61)]
        self.assertEqual(rates, [3e-3, 3e-3, 1.5e-3, 1.5e-3, 7.5e-4])
        self.assertEqual(LSTMConfig().learning_rate_at(3e-3, 400), 3e-3)

    def test_forget_gate_bias_initialisation(self):
        network = self._network(forget_gate_bias=1.0)
        bias = network._initial_parameters(np.random.default_rng(0), 2, 3, 96)[1]
        np.testing.assert_array_equal(bias[4:8], 1.0)
        np.testing.assert_array_equal(np.r_[bias[:4], bias[8:]], 0.0)

    def test_minibatches_cover_every_day_once(self):
        network = self._network(batch_size=4)
        batches = network._batches(10, np.random.default_rng(0))
        self.assertEqual([len(b) for b in batches], [4, 4, 2])
        self.assertEqual(sorted(np.concatenate(batches).tolist()), list(range(10)))
        self.assertEqual(self._network()._batches(10, np.random.default_rng(0)), [slice(None)])

    def test_gradient_clipping_caps_the_global_norm(self):
        clip = _OneLayerLSTMNetwork._clip_gradients
        gradients = [np.full((2, 2), 3.0), np.full(3, 4.0)]  # norm = sqrt(36 + 48)
        norm = np.sqrt(84.0)
        clipped = clip(gradients, 1.0)
        self.assertAlmostEqual(np.sqrt(sum((g * g).sum() for g in clipped)), 1.0)
        np.testing.assert_allclose(clipped[0], gradients[0] / norm)  # direction kept
        self.assertIs(clip(gradients, 100.0), gradients)  # below the cap: unchanged
        self.assertIs(clip(gradients, None), gradients)

    def test_all_options_train_to_finite_weights(self):
        rng = np.random.default_rng(3)
        sequence = rng.normal(size=(9, 96, 2))
        skip = rng.normal(size=(9, 3))
        targets = rng.normal(size=(9, 96))
        network = self._network(
            batch_size=4, lr_decay_step_epochs=2, lr_decay_factor=0.5,
            forget_gate_bias=1.0, gradient_clip_norm=0.5,
        )
        trained, epochs = network._train(sequence, skip, targets, 5, 1e-3, 0)
        self.assertEqual(epochs, 5)
        self.assertTrue(all(np.isfinite(p).all() for p in trained))


if __name__ == "__main__":
    unittest.main()
