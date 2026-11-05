import unittest

import numpy as np
import pandas as pd

from pipeline.delivery_index import make_delivery_index
from pipeline.mlp.NN_exaa_enriched import (
    NNExaaEnrichedConfig,
    _column_groups_exaa,
    rolling_nn_exaa_enriched_forecast,
)


class ExaaEnrichedNetworkTests(unittest.TestCase):
    @staticmethod
    def _feature_matrices(day_count=16):
        rng = np.random.default_rng(19)
        index = pd.date_range(
            "2025-01-01", periods=day_count, freq="D", tz="Europe/Berlin"
        )
        columns = {}
        for lag in (1, 2, 7):
            for mtu in range(96):
                columns[f"price_d{lag}_mtu_{mtu:02d}"] = rng.normal(
                    50 + lag, 10, size=day_count
                )
        columns["wind_speed_c5_h00"] = rng.normal(size=day_count)
        columns["load_d0_mtu_00"] = rng.normal(100, 15, size=day_count)
        for mtu in range(96):
            columns[f"exaa_d0_mtu_{mtu:02d}"] = rng.normal(
                50, 10, size=day_count
            )
        X = pd.DataFrame(columns, index=index)
        Y = pd.DataFrame(
            rng.normal(50, 20, size=(day_count, 96)),
            index=index,
            columns=range(96),
        )
        return X, Y

    def test_exaa_prices_are_hidden_inputs_not_lag_skip_inputs(self):
        X, _ = self._feature_matrices(day_count=1)
        hidden_columns, skip_columns = _column_groups_exaa(X.columns.tolist())

        self.assertIn("wind_speed_c5_h00", hidden_columns)
        self.assertIn("load_d0_mtu_00", hidden_columns)
        self.assertIn("exaa_d0_mtu_00", hidden_columns)
        self.assertIn("exaa_d0_mtu_95", hidden_columns)
        self.assertEqual(len([c for c in hidden_columns if c.startswith("exaa_")]), 96)
        self.assertFalse(any(c.startswith("exaa_") for c in skip_columns))

    def test_forecast_returns_canonical_finite_output(self):
        X, Y = self._feature_matrices()
        forecast_day = X.index[-1]
        result = rolling_nn_exaa_enriched_forecast(
            X,
            Y,
            [forecast_day],
            NNExaaEnrichedConfig(
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
        np.testing.assert_allclose(result["y_true"].to_numpy(), Y.iloc[-1].to_numpy())

    def test_missing_exaa_vector_fails_explicitly(self):
        X, Y = self._feature_matrices()
        X = X.drop(columns="exaa_d0_mtu_00")

        with self.assertRaisesRegex(ValueError, "EXAA must contain MTU columns"):
            rolling_nn_exaa_enriched_forecast(
                X,
                Y,
                [X.index[-1]],
                NNExaaEnrichedConfig(
                    train_days=20,
                    min_training_days=10,
                    max_epochs=2,
                    patience=1,
                ),
            )


if __name__ == "__main__":
    unittest.main()
