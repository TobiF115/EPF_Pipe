import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.delivery_index import make_delivery_index
from pipeline.mlp.NN_exaa_enriched import (
    NNExaaEnrichedConfig,
    _OneHiddenLayerExaaNetwork,
    _column_groups_exaa,
    _skip_connection_mask_exaa,
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

    def test_exaa_prices_feed_hidden_layer_and_local_skip_window(self):
        X, _ = self._feature_matrices(day_count=1)
        hidden_columns, skip_columns = _column_groups_exaa(X.columns.tolist())
        skip_mask = _skip_connection_mask_exaa(skip_columns)

        self.assertIn("wind_speed_c5_h00", hidden_columns)
        self.assertIn("load_d0_mtu_00", hidden_columns)
        self.assertIn("exaa_d0_mtu_00", hidden_columns)
        self.assertIn("exaa_d0_mtu_95", hidden_columns)
        self.assertEqual(len([c for c in hidden_columns if c.startswith("exaa_")]), 96)
        self.assertEqual(len([c for c in skip_columns if c.startswith("exaa_")]), 96)

        exaa_00_index = skip_columns.index("exaa_d0_mtu_00")
        expected_targets = {0, 1, 2, 94, 95}
        self.assertEqual(
            set(np.flatnonzero(skip_mask[exaa_00_index])),
            expected_targets,
        )
        self.assertEqual(int(skip_mask[exaa_00_index].sum()), 5)

        exaa_50_index = skip_columns.index("exaa_d0_mtu_50")
        self.assertEqual(
            set(np.flatnonzero(skip_mask[exaa_50_index])),
            {48, 49, 50, 51, 52},
        )

        yesterday_00_index = skip_columns.index("price_d1_mtu_00")
        self.assertEqual(
            set(np.flatnonzero(skip_mask[yesterday_00_index])),
            set(range(9)) | set(range(88, 96)),
        )
        self.assertEqual(int(skip_mask.sum()), 96 * 5 + 96 * 19)

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

    def test_expanding_window_adds_each_complete_day(self):
        X, Y = self._feature_matrices(day_count=12)
        fit_sizes = []
        original_fit = _OneHiddenLayerExaaNetwork.fit

        def record_fit(network, hidden_inputs, skip_inputs, targets):
            fit_sizes.append(len(targets))
            return original_fit(network, hidden_inputs, skip_inputs, targets)

        with patch.object(_OneHiddenLayerExaaNetwork, "fit", new=record_fit):
            result = rolling_nn_exaa_enriched_forecast(
                X,
                Y,
                X.index[7:10],
                NNExaaEnrichedConfig(max_epochs=1, patience=1),
            )

        self.assertEqual(fit_sizes, [7, 8, 9])
        self.assertEqual(len(result), 3 * 96)

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
