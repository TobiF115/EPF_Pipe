"""Tests for the three custom-loss LSTMs in pipeline/lstm_custom_loss."""

import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np

from pipeline.lstm_custom_loss import adaptive, pretrained_only, rolling_window
from pipeline.lstm_custom_loss.common import (
    CustomLoss,
    CustomLossLSTMConfig,
    LSTMNetwork,
    ModelStore,
    custom_loss,
    custom_loss_components,
    custom_loss_gradient,
    pretrain_state,
)
from pipeline.mlp.LSTM_exaa_enriched import LSTMConfig
from tests.test_lstm_era5_online import _matrices

CONFIG = CustomLossLSTMConfig(
    pretrain=LSTMConfig(hidden_units=4, max_epochs=3, patience=2, min_training_days=10),
    update_epochs=5,
    update_learning_rate=1e-3,
    bias_window_days=7,
    max_catch_up_days=30,
)


def _numeric(f, values, h=1e-6):
    grad = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        up, down = values.copy(), values.copy()
        up[index] += h
        down[index] -= h
        grad[index] = (f(up) - f(down)) / (2 * h)
    return grad


class LossTests(unittest.TestCase):
    def test_output_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        loss = CustomLoss(jsd_weight=0.5, smoothness_weight=0.3)
        P, Y = rng.normal(size=(3, 96)), rng.normal(size=(3, 96))
        np.testing.assert_allclose(
            custom_loss_gradient(P, Y, loss),
            _numeric(lambda p: custom_loss(p, Y, loss), P),
            atol=1e-7,
        )

    def test_network_gradients_match_finite_differences(self):
        rng = np.random.default_rng(1)
        loss = CustomLoss(jsd_weight=0.5, smoothness_weight=0.3)
        mask = rng.random((4, 96)) < 0.3
        net = LSTMNetwork(LSTMConfig(hidden_units=3, l2=1e-3, forget_gate_bias=1.0), loss, mask)
        seq, skip, T = rng.normal(size=(2, 96, 2)), rng.normal(size=(2, 4)), rng.normal(size=(2, 96))
        params = net._initial_parameters(rng, 2)
        params[4] = rng.normal(size=params[4].shape)  # non-zero skip weights
        analytic = net._gradients(seq, skip, T, params)

        def total(p):
            l2 = net.config.l2 * sum(float(np.sum(w * w)) for w in (p[0], p[2], p[4] * mask))
            return custom_loss(net._forward(seq, skip, p)[0], T, loss) + l2

        self.assertEqual(len(params), 5)
        for k in range(5):
            def f(values, k=k):
                p = [w.copy() for w in params]
                p[k] = values
                return total(p)

            numeric = _numeric(f, params[k])
            if k == 4:  # masked-out skip weights have no effect and zero gradient
                numeric = numeric * mask
            np.testing.assert_allclose(analytic[k], numeric, atol=1e-6)

    def test_mae_term_matches_plain_mae(self):
        rng = np.random.default_rng(4)
        P, Y = rng.normal(size=(3, 96)), rng.normal(size=(3, 96))
        loss = CustomLoss(jsd_weight=0.0, smoothness_weight=0.0)
        self.assertAlmostEqual(custom_loss(P, Y, loss), np.mean(np.abs(P - Y)), places=12)
        np.testing.assert_allclose(custom_loss_gradient(P, Y, loss), np.sign(P - Y) / P.size)

    def test_identical_profiles_have_zero_jsd(self):
        Y = np.random.default_rng(2).normal(size=(2, 96))
        components = custom_loss_components(Y, Y, CustomLoss(jsd_weight=1.0))
        self.assertAlmostEqual(components["jsd"], 0.0, places=10)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.Y = _matrices(45, exaa=True)
        cls.days = [ts.date() for ts in cls.X.index]
        cls.pretrained = pretrain_state(cls.X.iloc[:30], cls.Y.iloc[:30], CONFIG)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _store(self, name):
        store = ModelStore(Path(self._tmp.name) / name)
        store.save_pretrained(self.pretrained)
        return store

    def test_store_round_trip_and_inputs(self):
        loaded = self._store("s").load_pretrained()
        self.assertEqual(len(loaded.parameters), 5)
        # EPEX lags: skip branch only; EXAA and the 7-day mean: both branches.
        sequence, skip = loaded.inputs(self.X.iloc[:1])
        for column in ("price_d1_mtu_00", "price_d2_mtu_10", "price_d7_mtu_95"):
            self.assertNotIn(column, loaded.columns)
            self.assertIn(column, loaded.skip_columns)
        for column in ("exaa_d0_mtu_00", "average_da_price_last_7d"):
            self.assertIn(column, loaded.columns)
            self.assertIn(column, loaded.skip_columns)
        self.assertNotIn("price_d1", [c[:8] for c in loaded.columns])
        self.assertEqual(sequence.shape[:2], (1, 96))
        self.assertEqual(skip.shape, (1, 4 * 96 + 1))
        mask = loaded.network(CONFIG).skip_mask
        # d1 +-8, d2, d7, EXAA +-2, mean
        self.assertTrue((mask.sum(axis=0) == 17 + 1 + 1 + 5 + 1).all())
        a, _ = pretrained_only.forecast_state(self.pretrained, self.X, self.Y, self.days[30], CONFIG)
        b, _ = pretrained_only.forecast_state(loaded, self.X, self.Y, self.days[30], CONFIG)
        np.testing.assert_allclose(a["y_pred"], b["y_pred"])

    def test_pretrained_only_never_changes(self):
        state, log = pretrained_only.advance_state(self._store("p"), self.X, self.Y, self.days[40], CONFIG)
        self.assertEqual(log, [])
        for a, b in zip(state.parameters, self.pretrained.parameters):
            np.testing.assert_array_equal(a, b)
        forecast, _ = pretrained_only.forecast_state(state, self.X, self.Y, self.days[40], CONFIG)
        self.assertTrue(np.isfinite(forecast["y_pred"]).all())

    def test_rolling_window_trains_on_last_seven_days(self):
        store = self._store("r")
        state, log = rolling_window.advance_state(store, self.X, self.Y, self.days[40], CONFIG)
        self.assertEqual(store.checkpoint_dates(), self.days[30:40])
        self.assertTrue(all(entry["kind"] == "updated" for entry in log))
        self.assertEqual(log[0]["window_days_used"], [d.isoformat() for d in self.days[24:31]])
        self.assertLess(log[0]["window_loss_after"], log[0]["window_loss_before"])
        self.assertFalse(np.allclose(state.parameters[0], self.pretrained.parameters[0]))

    def test_adaptive_accepts_only_validated_updates(self):
        store = self._store("a")
        state, log = adaptive.advance_state(store, self.X, self.Y, self.days[40], CONFIG)
        self.assertEqual(store.checkpoint_dates(), self.days[30:40])
        for i, entry in zip(range(30, 40), log):
            self.assertIn(entry["kind"], ("accepted", "rejected"))
            self.assertEqual(
                entry["validation_days_used"], [d.isoformat() for d in self.days[i - 6 : i]]
            )
            improved = entry["validation_rmse_current"] - entry["validation_rmse_candidate"] > 0
            self.assertEqual(entry["kind"] == "accepted", improved)
            # A rejected day keeps the parameters of the day before.
            if entry["kind"] == "rejected":
                before = store.load_latest(not_after=self.days[i - 1]).parameters
                after = store.load_latest(not_after=self.days[i]).parameters
                for a, b in zip(before, after):
                    np.testing.assert_array_equal(a, b)

    def test_adaptive_margin_rejects_everything(self):
        config = replace(CONFIG, acceptance_margin=1e9)
        state, log = adaptive.advance_state(self._store("m"), self.X, self.Y, self.days[35], config)
        self.assertTrue(all(entry["kind"] == "rejected" for entry in log))
        for a, b in zip(state.parameters, self.pretrained.parameters):
            np.testing.assert_array_equal(a, b)

    def test_incomplete_new_day_is_skipped(self):
        Y = self.Y.copy()
        Y.iloc[30] = np.nan
        for module, update in (
            (rolling_window, rolling_window.rolling_update_state),
            (adaptive, adaptive.adaptive_update_state),
        ):
            state = update(self.pretrained, self.X, Y, self.days[30], CONFIG)
            self.assertEqual(state.metadata["kind"], "skipped", module.__name__)
            for a, b in zip(state.parameters, self.pretrained.parameters):
                np.testing.assert_array_equal(a, b)

    def test_first_required_feature_day(self):
        target = self.days[40]
        self.assertEqual(
            rolling_window.first_required_feature_day(self._store("f1"), target, CONFIG),
            self.days[30] - timedelta(days=6),
        )
        self.assertEqual(
            adaptive.first_required_feature_day(self._store("f2"), target, CONFIG),
            self.days[30] - timedelta(days=6),
        )
        self.assertEqual(
            pretrained_only.first_required_feature_day(self._store("f3"), target, CONFIG),
            target - timedelta(days=7),
        )


if __name__ == "__main__":
    unittest.main()


class SqraQuantileTests(unittest.TestCase):
    @staticmethod
    def _members(day_count=20, seed=0):
        import pandas as pd
        from pipeline.delivery_index import make_delivery_index

        rng = np.random.default_rng(seed)
        days = pd.date_range("2026-10-01", periods=day_count, freq="D")
        index = make_delivery_index(days)
        truth = rng.normal(80, 20, len(index))
        truth[-96:] = np.nan  # target day: prices not yet known
        members = {}
        for name, noise in (("pretrained_only", 8), ("rolling_window", 6), ("adaptive", 7)):
            members[name] = pd.DataFrame(
                {"y_pred": np.nan_to_num(truth, nan=80) + rng.normal(0, noise, len(index)), "y_true": truth},
                index=index,
            )
        return members, days

    def test_quantiles_are_sorted_and_complete(self):
        from pipeline.lstm_custom_loss.sqra_quantiles import build_panel, sqra_quantile_forecast

        members, days = self._members()
        panel, cols = build_panel(members)
        self.assertEqual(cols, ["y_pred_pretrained_only", "y_pred_rolling_window", "y_pred_adaptive"])
        forecast = sqra_quantile_forecast(panel, cols, days[-1].date(), train_days=19)
        quantile_cols = [c for c in forecast.columns if c.startswith("q")]
        self.assertEqual(len(forecast), 96)
        self.assertEqual(len(quantile_cols), 5)
        values = forecast[quantile_cols].to_numpy()
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((np.diff(values, axis=1) >= 0).all())

    def test_refuses_too_short_calibration(self):
        from pipeline.lstm_custom_loss.sqra_quantiles import build_panel, sqra_quantile_forecast

        members, days = self._members(day_count=8)
        panel, cols = build_panel(members)
        with self.assertRaises(ValueError):
            sqra_quantile_forecast(panel, cols, days[-1].date(), train_days=60)

    def test_conflicting_truth_is_rejected(self):
        from pipeline.lstm_custom_loss.sqra_quantiles import build_panel

        members, _ = self._members()
        members["adaptive"] = members["adaptive"].assign(y_true=members["adaptive"]["y_true"] + 1)
        with self.assertRaises(ValueError):
            build_panel(members)
