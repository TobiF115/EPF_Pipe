import unittest

import numpy as np
import pandas as pd

from pipeline.delivery_index import make_delivery_index
from pipeline.mlp.mlp import (
    MLPConfig,
    _column_groups,
    _skip_connection_mask,
    rolling_mlp_forecast,
)


class MLPForecastTests(unittest.TestCase):
    def test_default_config_uses_two_tanh_layers_and_two_training_phases(self):
        config = MLPConfig()

        self.assertEqual(config.hidden_units, 32)
        self.assertEqual(config.hidden_layers, 2)
        self.assertEqual(config.activation, "tanh")
        self.assertEqual(config.train_days, 116)
        self.assertEqual(config.fine_tune_fraction, 0.3)
        self.assertLess(
            config.fine_tune_learning_rate,
            config.learning_rate,
        )

    @staticmethod
    def _feature_matrices(day_count=16):
        rng = np.random.default_rng(7)
        index = pd.date_range(
            "2025-01-01", periods=day_count, freq="D", tz="Europe/Berlin"
        )
        columns = {}
        for lag in (1, 2, 7):
            for mtu in range(96):
                columns[f"price_d{lag}_mtu_{mtu:02d}"] = rng.normal(
                    50 + lag, 10, size=day_count
                )
        for feature in ("wind_speed_c5_00", "ssrd_c5_00", "is_holiday"):
            columns[feature] = rng.normal(size=day_count)
        for mtu in range(96):
            columns[f"load_d0_mtu_{mtu:02d}"] = rng.normal(
                100, 15, size=day_count
            )
        X = pd.DataFrame(columns, index=index)
        Y = pd.DataFrame(
            rng.normal(50, 20, size=(day_count, 96)),
            index=index,
            columns=range(96),
        )
        return X, Y

    def test_skip_branch_uses_mtu_aligned_price_lags_and_previous_day_window(self):
        X, _ = self._feature_matrices(day_count=1)
        hidden_columns, skip_columns = _column_groups(X.columns.tolist())
        skip_mask = _skip_connection_mask(skip_columns)

        self.assertIn("wind_speed_c5_00", hidden_columns)
        self.assertIn("load_d0_mtu_00", hidden_columns)
        self.assertEqual(len(skip_columns), 3 * 96)
        self.assertEqual(
            skip_columns[:4],
            [f"price_d1_mtu_{mtu:02d}" for mtu in range(4)],
        )
        self.assertEqual(
            skip_columns[96:100],
            [f"price_d2_mtu_{mtu:02d}" for mtu in range(4)],
        )
        self.assertTrue(np.all(skip_mask.sum(axis=0) == 19))
        d1_mtu_00 = skip_columns.index("price_d1_mtu_00")
        d2_mtu_00 = skip_columns.index("price_d2_mtu_00")
        d7_mtu_00 = skip_columns.index("price_d7_mtu_00")
        allowed_d1_targets = list(range(9)) + list(range(88, 96))
        self.assertTrue(skip_mask[d1_mtu_00, allowed_d1_targets].all())
        self.assertFalse(skip_mask[d1_mtu_00, 9:88].any())
        self.assertTrue(skip_mask[d2_mtu_00, 0])
        self.assertEqual(int(skip_mask[d2_mtu_00].sum()), 1)
        self.assertTrue(skip_mask[d7_mtu_00, 0])
        self.assertEqual(int(skip_mask[d7_mtu_00].sum()), 1)

    def test_rolling_forecast_returns_canonical_96_mtu_frame(self):
        X, Y = self._feature_matrices()
        forecast_day = X.index[-1]
        result = rolling_mlp_forecast(
            X,
            Y,
            [forecast_day],
            MLPConfig(
                train_days=20,
                min_training_days=10,
                max_epochs=5,
                patience=2,
            ),
        )

        self.assertEqual(list(result.columns), ["y_pred", "y_true"])
        self.assertTrue(result.index.equals(make_delivery_index([forecast_day])))
        self.assertEqual(len(result), 96)
        self.assertTrue(np.isfinite(result["y_pred"]).all())
        np.testing.assert_allclose(
            result["y_true"].to_numpy(), Y.iloc[-1].to_numpy()
        )

    def test_incomplete_target_features_fail_explicitly(self):
        X, Y = self._feature_matrices()
        forecast_day = X.index[-1]
        X.loc[forecast_day, "price_d2_mtu_00"] = np.nan

        with self.assertRaisesRegex(ValueError, "Forecast features are incomplete"):
            rolling_mlp_forecast(
                X,
                Y,
                [forecast_day],
                MLPConfig(
                    train_days=20,
                    min_training_days=10,
                    max_epochs=2,
                    patience=1,
                ),
            )


if __name__ == "__main__":
    unittest.main()
