import unittest

import numpy as np
import pandas as pd

from pipeline.delivery_index import make_delivery_index
from pipeline.mlp.LSTM_exaa_enriched import LSTMConfig
from pipeline.mlp.LSTM_fundamental import (
    _fundamental_skip_connection_mask,
    rolling_lstm_fundamental_forecast,
)


class FundamentalLSTMTests(unittest.TestCase):
    @staticmethod
    def _feature_matrices(day_count=16):
        rng = np.random.default_rng(42)
        index = pd.date_range(
            "2025-01-01", periods=day_count, freq="D", tz="Europe/Berlin"
        )
        columns = {}
        for lag in (1, 2, 7):
            for mtu in range(96):
                columns[f"price_d{lag}_mtu_{mtu:02d}"] = rng.normal(
                    50 + lag, 10, size=day_count
                )
        for hour in range(24):
            columns[f"wind_speed_cluster_5_h{hour:02d}"] = rng.normal(
                size=day_count
            )
        for variable in ("sw_dir_cluster_5", "sw_dif_cluster_5", "load_d0"):
            for mtu in range(96):
                columns[f"{variable}_mtu{mtu:02d}"] = rng.normal(
                    size=day_count
                )
        columns["is_holiday"] = rng.integers(0, 2, size=day_count)
        columns["average_da_price_last_7d"] = rng.normal(
            50, 5, size=day_count
        )
        X = pd.DataFrame(columns, index=index)
        Y = pd.DataFrame(
            rng.normal(50, 20, size=(day_count, 96)),
            index=index,
            columns=range(96),
        )
        return X, Y

    def test_weekly_average_skip_reaches_every_output(self):
        skip_columns = [
            f"price_d{lag}_mtu_{mtu:02d}"
            for lag in (1, 2, 7)
            for mtu in range(96)
        ]

        mask = _fundamental_skip_connection_mask(skip_columns)

        self.assertEqual(mask.shape, (289, 96))
        self.assertTrue(mask[-1].all())

    def test_forecast_runs_without_exaa_features(self):
        X, Y = self._feature_matrices()
        forecast_day = X.index[-1]

        result = rolling_lstm_fundamental_forecast(
            X,
            Y,
            [forecast_day],
            LSTMConfig(
                train_days=20,
                min_training_days=10,
                hidden_units=4,
                max_epochs=3,
                patience=2,
            ),
        )

        self.assertFalse(any(column.startswith("exaa_") for column in X.columns))
        self.assertEqual(list(result.columns), ["y_pred", "y_true"])
        self.assertTrue(result.index.equals(make_delivery_index([forecast_day])))
        self.assertEqual(len(result), 96)
        self.assertTrue(np.isfinite(result["y_pred"]).all())
        np.testing.assert_allclose(result["y_true"].to_numpy(), Y.iloc[-1].to_numpy())

    def test_exaa_features_are_rejected(self):
        X, Y = self._feature_matrices()
        X["exaa_d0_mtu_00"] = 1.0

        with self.assertRaisesRegex(ValueError, "unsupported columns"):
            rolling_lstm_fundamental_forecast(
                X,
                Y,
                [X.index[-1]],
                LSTMConfig(
                    train_days=20,
                    min_training_days=10,
                    max_epochs=2,
                    patience=1,
                ),
            )


if __name__ == "__main__":
    unittest.main()
