from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.mlp.LSTM_exaa_enriched_price_load import (
    build_price_load_inputs,
    rolling_lstm_exaa_enriched_price_load_forecast,
)


class LSTMExaaEnrichedPriceLoadTests(unittest.TestCase):
    def test_inputs_contain_epex_lags_exaa_and_load_without_weather(self):
        timezone = "Europe/Berlin"
        epex_index = pd.date_range(
            "2025-09-24",
            "2025-10-04",
            freq="15min",
            tz=timezone,
            inclusive="left",
        )
        exaa_index = pd.date_range(
            "2025-10-01",
            "2025-10-05",
            freq="15min",
            tz=timezone,
            inclusive="left",
        )
        prices = pd.DataFrame(
            {"price_da": np.arange(len(epex_index), dtype=float)},
            index=epex_index,
        )
        exaa = pd.DataFrame(
            {"price_exaa": np.arange(len(exaa_index), dtype=float)},
            index=exaa_index,
        )
        load = pd.DataFrame(
            {"load_fc": np.arange(len(exaa_index), dtype=float)},
            index=exaa_index,
        )

        features, target = build_price_load_inputs(
            prices,
            exaa,
            load,
            feature_start_date=date(2025, 10, 1),
            target_date=date(2025, 10, 4),
            timezone=timezone,
        )

        self.assertEqual(len(features), 4)
        self.assertEqual(len(target), 4)
        self.assertEqual(features.shape[1], 288 + 96 + 96)
        self.assertTrue(
            all(
                column.startswith(("price_d", "exaa_d0", "load_d0"))
                for column in features
            )
        )
        self.assertTrue(
            np.isfinite(
                features.loc[pd.Timestamp("2025-10-01", tz=timezone)]
            ).all()
        )
        self.assertEqual(
            int(np.isfinite(target.loc[pd.Timestamp("2025-10-04", tz=timezone)]).sum()),
            0,
        )

    def test_model_rejects_weather_inputs(self):
        timezone = "Europe/Berlin"
        index = pd.DatetimeIndex([pd.Timestamp("2025-10-01", tz=timezone)])
        columns = [
            *(
                f"price_d{lag}_mtu_{mtu:02d}"
                for lag in (1, 2, 7)
                for mtu in range(96)
            ),
            *(f"exaa_d0_mtu_{mtu:02d}" for mtu in range(96)),
            *(f"load_d0_mtu_{mtu:02d}" for mtu in range(96)),
            "weather_u10",
        ]
        features = pd.DataFrame(
            np.ones((1, len(columns))),
            index=index,
            columns=columns,
        )
        target = pd.DataFrame(np.ones((1, 96)), index=index, columns=range(96))

        with patch(
            "pipeline.mlp.LSTM_exaa_enriched_price_load._rolling_lstm_forecast"
        ) as rolling:
            with self.assertRaisesRegex(ValueError, "price/load LSTM inputs"):
                rolling_lstm_exaa_enriched_price_load_forecast(
                    features,
                    target,
                    [index[0]],
                )
        rolling.assert_not_called()

    def test_model_routes_price_exaa_and_load_without_weekly_average(self):
        timezone = "Europe/Berlin"
        index = pd.DatetimeIndex([pd.Timestamp("2025-10-01", tz=timezone)])
        columns = [
            *(
                f"price_d{lag}_mtu_{mtu:02d}"
                for lag in (1, 2, 7)
                for mtu in range(96)
            ),
            *(f"exaa_d0_mtu_{mtu:02d}" for mtu in range(96)),
            *(f"load_d0_mtu_{mtu:02d}" for mtu in range(96)),
        ]
        features = pd.DataFrame(
            np.ones((1, len(columns))),
            index=index,
            columns=columns,
        )
        target = pd.DataFrame(np.ones((1, 96)), index=index, columns=range(96))

        with patch(
            "pipeline.mlp.LSTM_exaa_enriched_price_load._rolling_lstm_forecast",
            return_value="forecast",
        ) as rolling:
            result = rolling_lstm_exaa_enriched_price_load_forecast(
                features,
                target,
                [index[0]],
            )

        self.assertEqual(result, "forecast")
        self.assertFalse(rolling.call_args.kwargs["require_seven_day_average"])
        self.assertTrue(
            all(
                column.startswith(("exaa_d0", "load_d0"))
                for column in rolling.call_args.kwargs["hidden_columns"]
            )
        )


if __name__ == "__main__":
    unittest.main()
