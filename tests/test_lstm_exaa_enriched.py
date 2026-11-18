import unittest

import numpy as np
import pandas as pd

from pipeline.delivery_index import make_delivery_index
from pipeline.mlp.LSTM_exaa_enriched import (
    LSTMExaaEnrichedConfig,
    SEVEN_DAY_AVERAGE_PRICE_COLUMN,
    _skip_connection_mask_with_average_price,
    _sequence_feature_tensor,
    add_seven_day_average_price_feature,
    rolling_lstm_exaa_enriched_forecast,
)


class ExaaEnrichedLSTMTests(unittest.TestCase):
    @staticmethod
    def _feature_matrices(day_count=16):
        rng = np.random.default_rng(23)
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
        for variable in ("sw_dir_cluster_5", "sw_dif_cluster_5"):
            for mtu in range(96):
                columns[f"{variable}_mtu{mtu:02d}"] = rng.normal(
                    size=day_count
                )
        for mtu in range(96):
            columns[f"load_d0_mtu_{mtu:02d}"] = rng.normal(
                100, 15, size=day_count
            )
            columns[f"exaa_d0_mtu_{mtu:02d}"] = rng.normal(
                50, 10, size=day_count
            )
        columns["is_holiday"] = rng.integers(0, 2, size=day_count)
        X = pd.DataFrame(columns, index=index)
        Y = pd.DataFrame(
            rng.normal(50, 20, size=(day_count, 96)),
            index=index,
            columns=range(96),
        )
        return X, Y

    def test_features_are_ordered_by_mtu_and_weather_resolution(self):
        X, _ = self._feature_matrices(day_count=1)
        hidden_columns = [column for column in X if not column.startswith("price_")]
        sequence = _sequence_feature_tensor(X, hidden_columns)

        self.assertEqual(sequence.shape, (1, 96, 6))
        channel = {
            name: position
            for position, name in enumerate(
                [
                    "wind_speed_cluster_5",
                    "sw_dir_cluster_5",
                    "sw_dif_cluster_5",
                    "load_d0",
                    "exaa_d0",
                    "is_holiday",
                ]
            )
        }
        # Wind is hourly and held constant for its four quarter-hour MTUs.
        wind = channel["wind_speed_cluster_5"]
        np.testing.assert_array_equal(sequence[0, :4, wind], sequence[0, 0, wind])
        np.testing.assert_array_equal(sequence[0, 4:8, wind], sequence[0, 4, wind])
        self.assertNotEqual(sequence[0, 0, wind], sequence[0, 4, wind])
        self.assertEqual(
            sequence[0, 50, channel["load_d0"]],
            X.iloc[0]["load_d0_mtu_50"],
        )
        self.assertEqual(
            sequence[0, 50, channel["exaa_d0"]],
            X.iloc[0]["exaa_d0_mtu_50"],
        )
        self.assertTrue(
            (sequence[:, :, channel["is_holiday"]] == X.iloc[0]["is_holiday"]).all()
        )

    def test_seven_day_price_mean_uses_only_complete_prior_days(self):
        days = pd.date_range(
            "2025-01-01", periods=13, freq="D", tz="Europe/Berlin"
        )
        timestamps = pd.date_range(
            days[0],
            days[-2] + pd.DateOffset(days=1),
            freq="15min",
            inclusive="left",
        )
        prices = pd.DataFrame(
            {
                "price_da": np.repeat(np.arange(10.0, 130.0, 10.0), 96)
            },
            index=timestamps,
        )
        X = pd.DataFrame({"feature": 1.0}, index=days)

        result = add_seven_day_average_price_feature(X, prices)

        self.assertTrue(
            result[SEVEN_DAY_AVERAGE_PRICE_COLUMN].iloc[:7].isna().all()
        )
        self.assertEqual(
            result.loc[days[7], SEVEN_DAY_AVERAGE_PRICE_COLUMN], 40.0
        )
        self.assertEqual(
            result.loc[days[8], SEVEN_DAY_AVERAGE_PRICE_COLUMN], 50.0
        )
        self.assertEqual(
            result.loc[days[-1], SEVEN_DAY_AVERAGE_PRICE_COLUMN], 90.0
        )

    def test_seven_day_mean_skip_connection_reaches_every_output(self):
        skip_columns = [f"exaa_d0_mtu_{mtu:02d}" for mtu in range(96)]

        mask = _skip_connection_mask_with_average_price(skip_columns)

        self.assertEqual(mask.shape, (97, 96))
        self.assertTrue(mask[-1].all())

    def test_forecast_returns_canonical_finite_output(self):
        X, Y = self._feature_matrices()
        forecast_day = X.index[-1]
        price_days = pd.date_range(
            X.index[0] - pd.DateOffset(days=7),
            X.index[-1] + pd.DateOffset(days=1),
            freq="D",
        )
        price_index = pd.date_range(
            price_days[0],
            price_days[-1],
            freq="15min",
            inclusive="left",
        )
        prices = pd.DataFrame(
            {"price_da": np.tile(np.arange(96, dtype=float), len(price_days) - 1)},
            index=price_index,
        )
        X = add_seven_day_average_price_feature(X, prices)
        result = rolling_lstm_exaa_enriched_forecast(
            X,
            Y,
            [forecast_day],
            LSTMExaaEnrichedConfig(
                train_days=20,
                min_training_days=10,
                hidden_units=4,
                max_epochs=3,
                patience=2,
            ),
        )

        self.assertEqual(list(result.columns), ["y_pred", "y_true"])
        self.assertTrue(result.index.equals(make_delivery_index([forecast_day])))
        self.assertEqual(len(result), 96)
        self.assertTrue(np.isfinite(result["y_pred"]).all())
        np.testing.assert_allclose(result["y_true"].to_numpy(), Y.iloc[-1].to_numpy())

    def test_incomplete_exaa_vector_fails_explicitly(self):
        X, Y = self._feature_matrices()
        X = X.drop(columns="exaa_d0_mtu_00")

        with self.assertRaisesRegex(ValueError, "EXAA must contain MTU columns"):
            rolling_lstm_exaa_enriched_forecast(
                X,
                Y,
                [X.index[-1]],
                LSTMExaaEnrichedConfig(
                    train_days=20,
                    min_training_days=10,
                    max_epochs=2,
                    patience=1,
                ),
            )


if __name__ == "__main__":
    unittest.main()
