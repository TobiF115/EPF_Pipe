"""Tests for ERA5 download, aggregation helpers and ICON-format export."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pipeline.dwd_history import (
    consolidated_delivery_available,
    history_is_complete,
    load_consolidated_dwd,
    read_history_tables,
    update_history,
)
from pipeline.lear.lear_model import build_dwd_features, load_era5
from preprocessing import aggregate_era5, download_era5, prepare_era5

TZ = "Europe/Berlin"
N_CLUSTERS = 3


class DownloadRequestTests(unittest.TestCase):
    def test_month_requests_cover_range_with_partial_last_month(self):
        requests = download_era5.month_requests(date(2025, 1, 1), date(2026, 11, 13))
        self.assertEqual(len(requests), 23)
        feb = [r for r in requests if (r.year, r.month) == (2025, 2)][0]
        self.assertEqual(feb.days, tuple(range(1, 29)))
        self.assertEqual(requests[-1].days, tuple(range(1, 14)))
        self.assertEqual(requests[-1].filename, "era5_2026_11.grib")

    def test_payload_matches_manual_cds_request(self):
        (request,) = download_era5.month_requests(date(2025, 3, 5), date(2025, 3, 6))
        payload = request.payload()
        self.assertEqual(payload["data_format"], "grib")
        self.assertEqual(payload["download_format"], "unarchived")
        self.assertEqual(payload["area"], [55, 5.75, 47.25, 15])
        self.assertEqual(payload["day"], ["05", "06"])
        self.assertEqual(len(payload["time"]), 24)
        self.assertEqual(len(payload["variable"]), 4)
        # The four requested variables are exactly the required ones.
        self.assertEqual(
            len(aggregate_era5.MAIN_VARIABLES) + len(aggregate_era5.SOLAR_VARIABLES), 4
        )

    def test_every_cluster_gets_era5_points_on_real_grid(self):
        north, west, south, east = download_era5.AREA
        lats = np.arange(north, south - 1e-9, -0.25)
        lons = np.arange(west, east + 1e-9, 0.25)
        repo = Path(__file__).resolve().parent.parent
        for clusters in (1, 5, 25):
            cluster_df = pd.read_parquet(
                repo / "data" / "clustering" / f"icon_d2_clustering_c{clusters}.parquet"
            )
            grid = aggregate_era5.build_era5_cluster_grid(cluster_df, lats, lons)
            counts = np.bincount(grid[grid >= 0], minlength=clusters)
            self.assertEqual(len(counts), clusters)
            self.assertTrue((counts >= 20).all(), f"C={clusters}: {counts}")
            # Box corners (e.g. Austria near 47.25N/15E) must be masked.
            self.assertTrue((grid == -1).any())

    def test_monthly_marker_requires_exact_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "era5_2025_01.grib"
            request, = download_era5.month_requests(date(2025, 1, 1), date(2025, 1, 10))
            path.write_bytes(b"GRIB....")
            self.assertFalse(download_era5.monthly_file_complete(path, request))
            path.with_suffix(".grib.days").write_text(",".join(map(str, request.days)))
            self.assertTrue(download_era5.monthly_file_complete(path, request))
            longer, = download_era5.month_requests(date(2025, 1, 1), date(2025, 1, 20))
            self.assertFalse(download_era5.monthly_file_complete(path, longer))


class MultiFileAggregationTests(unittest.TestCase):
    def test_combine_parts_sorts_dedupes_and_drops_missing_hours(self):
        t1 = pd.date_range("2025-01-01 00:00", periods=4, freq="1h")
        t2 = pd.date_range("2025-01-01 02:00", periods=4, freq="1h")  # overlaps t1
        v1 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [np.nan, np.nan]])
        v2 = np.array([[30.0, 30.0], [40.0, 40.0], [50.0, 50.0], [np.nan, np.nan]])
        times, values = aggregate_era5.combine_parts([(t2, v2), (t1, v1)])
        self.assertEqual(list(times), list(pd.date_range("2025-01-01 00:00", periods=5, freq="1h")))
        # 02:00 is in both files; the all-NaN 03:00 row of file 1 must not win.
        np.testing.assert_array_equal(values[:, 0], [1.0, 2.0, 30.0, 40.0, 50.0])

    def test_aggregate_ignores_masked_points_and_nan_fields(self):
        values = np.array([
            [[1.0, 3.0], [100.0, 5.0]],
            [[np.nan, np.nan], [np.nan, np.nan]],
        ])
        grid = np.array([[0, 0], [-1, 1]])
        out = aggregate_era5.aggregate_to_clusters(values, grid, 2)
        np.testing.assert_array_equal(out[0], [2.0, 5.0])
        self.assertTrue(np.isnan(out[1]).all())

    def test_input_files_finds_manual_download_and_monthly_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "monthly").mkdir()
            manual = root / "8f3c2a1b.grib"  # CDS names downloads with a hash
            monthly = root / "monthly" / "era5_2026_11.grib"
            for path in (manual, monthly):
                path.write_bytes(b"GRIB")
            (root / "monthly" / "era5_2026_12.grib.part").write_bytes(b"GRIB")
            saved = aggregate_era5.ERA5_RAW_ARCHIVE, aggregate_era5.ERA5_MAIN_FILE
            aggregate_era5.ERA5_RAW_ARCHIVE = root
            aggregate_era5.ERA5_MAIN_FILE = root / "missing.grib"
            try:
                files = aggregate_era5.input_files()
            finally:
                aggregate_era5.ERA5_RAW_ARCHIVE, aggregate_era5.ERA5_MAIN_FILE = saved
            self.assertEqual(sorted(files), sorted([manual, monthly]))


class HourlyPriceFallbackTests(unittest.TestCase):
    def test_hourly_prices_fill_days_before_15_minute_go_live(self):
        from pipeline.lear.lear_model import _prefer_quarter_hourly

        hourly = pd.Series(1.0, index=pd.date_range("2025-09-29 22:00", "2025-09-30 21:00", freq="1h", tz="UTC"))
        quarter = pd.Series(2.0, index=pd.date_range("2025-09-30 22:00", "2025-10-01 21:45", freq="15min", tz="UTC"))
        combined = _prefer_quarter_hourly({"15T": quarter, "60T": hourly}, sequence=1)
        self.assertEqual(len(combined), 24 + 96)
        self.assertTrue((combined.loc[:"2025-09-30 21:00"] == 1.0).all())
        self.assertTrue((combined.loc["2025-09-30 22:00":] == 2.0).all())
        self.assertEqual(len(_prefer_quarter_hourly({"60T": hourly}, sequence=1)), 24)
        with self.assertRaises(ValueError):
            _prefer_quarter_hourly({}, sequence=1)

    def test_newer_entsoe_resolution_keys(self):
        # entsoe-py >= 0.7 keys parse_prices() by "15min"/"30min"/"60min".
        from pipeline.lear.lear_model import _prefer_quarter_hourly

        hourly = pd.Series(1.0, index=pd.date_range("2025-09-29 22:00", "2025-09-30 21:00", freq="1h", tz="UTC"))
        quarter = pd.Series(2.0, index=pd.date_range("2025-09-30 22:00", "2025-10-01 21:45", freq="15min", tz="UTC"))
        parsed = {"15min": quarter, "30min": pd.Series(dtype=float), "60min": hourly}
        self.assertEqual(len(_prefer_quarter_hourly(parsed, sequence=1)), 24 + 96)
        self.assertEqual(len(_prefer_quarter_hourly({"15min": quarter}, sequence=1)), 96)


class SolarTimeAxisTests(unittest.TestCase):
    def test_time_step_hypercube_flattens_to_unique_interval_starts(self):
        init = pd.to_datetime(["2025-01-01T06:00", "2025-01-01T18:00"])
        steps = pd.to_timedelta(range(1, 13), unit="h")
        valid = np.array([[t + s for s in steps] for t in init], dtype="datetime64[ns]")
        values = np.arange(2 * 12 * 2 * 2, dtype=float).reshape(2, 12, 2, 2) * 3600.0
        ds = xr.Dataset(
            {"ssrd": (("time", "step", "latitude", "longitude"), values)},
            coords={
                "time": init, "step": steps,
                "latitude": [50.0, 49.75], "longitude": [8.0, 8.25],
                "valid_time": (("time", "step"), valid),
            },
        )
        flat, times = aggregate_era5.extract_solar_values_and_times(ds, ds["ssrd"])
        self.assertEqual(flat.shape, (24, 2, 2))
        self.assertTrue(times.is_unique)
        self.assertEqual(times[0], pd.Timestamp("2025-01-01T06:00"))  # 07:00 valid -> 06:00 start
        self.assertEqual(times[-1], pd.Timestamp("2025-01-02T05:00"))

        cluster_df = pd.DataFrame({"lon": [8.0, 8.25], "lat": [50.0, 50.0], "cluster_id": [0, 1]})
        grid = aggregate_era5.build_era5_cluster_grid(
            cluster_df, ds.latitude.values, ds.longitude.values, max_distance_deg=None
        )
        np.testing.assert_array_equal(grid, [[0, 1], [0, 1]])
        # With the mask, the row 0.25 deg away from every cluster point is dropped.
        masked = aggregate_era5.build_era5_cluster_grid(
            cluster_df, ds.latitude.values, ds.longitude.values, max_distance_deg=0.1
        )
        np.testing.assert_array_equal(masked, [[0, 1], [-1, -1]])
        means = aggregate_era5.aggregate_to_clusters(flat, grid, 2)
        np.testing.assert_allclose(means[0], [flat[0, :, 0].mean(), flat[0, :, 1].mean()])


def _write_synthetic_era5(year_dir: Path, start: str, end: str) -> pd.DatetimeIndex:
    """Write CSVs exactly like aggregate_era5.write_icon_style_csv does."""
    times = pd.date_range(start, end, freq="1h")  # naive UTC, interval starts
    rng = np.random.default_rng(0)
    hour = times.hour.to_numpy()[:, None]
    daylight = np.clip(np.sin((hour - 5) / 14 * np.pi), 0, None)
    data = {
        "u10": rng.normal(3, 2, (len(times), N_CLUSTERS)),
        "v10": rng.normal(-1, 2, (len(times), N_CLUSTERS)),
        "ssrd": 400 * daylight * rng.uniform(0.5, 1.0, (len(times), N_CLUSTERS)),
    }
    # Direct part slightly above global in a few cells to exercise clipping.
    data["fdir"] = data["ssrd"] * rng.uniform(0.3, 1.05, (len(times), N_CLUSTERS))
    units = {"u10": "m s**-1", "v10": "m s**-1", "ssrd": "W m-2", "fdir": "W m-2"}
    for var, values in data.items():
        name = f"{var}_{aggregate_era5.safe_unit_for_filename(units[var])}_{times[0]:%Y%m%d%H}.csv"
        aggregate_era5.write_icon_style_csv(
            year_dir / name, times, values, var, var, units[var], N_CLUSTERS, "test"
        )
    return times


class HistoryExportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        year_dir = self.root / "era5" / "c3" / "2026"
        # Includes the DST switch on 2026-03-29 (23-hour local day).
        _write_synthetic_era5(year_dir, "2026-03-27 00:00", "2026-03-31 23:00")
        self.df = load_era5([year_dir], target_tz=TZ)
        self.tables = prepare_era5.era5_to_history_tables(self.df, timezone=TZ)

    def tearDown(self):
        self._tmp.cleanup()

    def test_only_complete_local_days_are_exported(self):
        days = sorted(self.tables["u10"]["delivery_date"].unique())
        # 03-27 starts at 23:00 UTC on 03-26 locally -> incomplete, dropped.
        self.assertEqual(days, ["2026-03-28", "2026-03-29", "2026-03-30", "2026-03-31"])
        dst = self.tables["u10"][self.tables["u10"]["delivery_date"] == "2026-03-29"]
        self.assertEqual(len(dst), 23)
        dst_solar = self.tables["ASWDIR_S"][self.tables["ASWDIR_S"]["delivery_date"] == "2026-03-29"]
        self.assertEqual(len(dst_solar), 92)
        self.assertEqual(prepare_era5.validate_all_days(self.tables, timezone=TZ), 4)

    def test_metadata_and_radiation_mapping(self):
        u10 = self.tables["u10"]
        self.assertTrue((u10["run_hour"] == "06").all())
        first = u10.iloc[0]
        self.assertEqual(first["issue_date"], "2026-03-27")
        ts = pd.Timestamp("2026-03-30 10:00", tz="UTC")
        ssrd = self.df.loc[ts, "ssrd_cluster_1"]
        fdir = self.df.loc[ts, "fdir_cluster_1"]
        quarter = self.tables["ASWDIFD_S"].set_index("timestamp")
        direct = self.tables["ASWDIR_S"].set_index("timestamp")
        for minute in (0, 15, 30, 45):
            stamp = ts + pd.Timedelta(minutes=minute)
            self.assertAlmostEqual(direct.loc[stamp, "cluster_1"], fdir)
            self.assertAlmostEqual(quarter.loc[stamp, "cluster_1"], max(ssrd - fdir, 0.0))
        self.assertGreaterEqual(self.tables["ASWDIFD_S"]["cluster_0"].min(), 0.0)

    def test_written_history_is_usable_by_model_loaders(self):
        cluster_dir = self.root / "era5_icon" / "c3"
        prepare_era5.write_history(cluster_dir, self.tables, timezone=TZ, source="ERA5")
        self.assertTrue(history_is_complete(cluster_dir))
        self.assertEqual(json.loads((cluster_dir / "dwd_history.json").read_text())["source"], "ERA5")
        self.assertTrue(consolidated_delivery_available(
            cluster_dir, delivery_date=date(2026, 3, 29), run_hour="06", timezone=TZ
        ))
        hourly, quarter = load_consolidated_dwd(
            cluster_dir, start_issue_date=date(2026, 3, 27), end_issue_date=None,
            required_run="06", skipped_issue_dates=[], target_tz=TZ,
        )
        features = build_dwd_features(hourly, quarter, target_tz=TZ)
        self.assertEqual(len(features), 4)
        self.assertEqual(
            sum(c.startswith("wind_speed_cluster_0_h") for c in features.columns), 24
        )
        self.assertEqual(
            sum(c.startswith("sw_dir_cluster_0_mtu") for c in features.columns), 96
        )

    def test_feature_columns_match_real_icon_history(self):
        repo = Path(__file__).resolve().parent.parent
        icon_dir = repo / "data" / "icon" / "c5"
        if not history_is_complete(icon_dir):
            self.skipTest("No ICON-D2 history available.")
        year_dir = self.root / "era5" / "c5" / "2026"
        global N_CLUSTERS
        saved, N_CLUSTERS = N_CLUSTERS, 5
        try:
            _write_synthetic_era5(year_dir, "2026-06-01 00:00", "2026-06-03 23:00")
        finally:
            N_CLUSTERS = saved
        tables = prepare_era5.era5_to_history_tables(load_era5([year_dir], target_tz=TZ), timezone=TZ)
        era5_dir = self.root / "era5_icon" / "c5"
        prepare_era5.write_history(era5_dir, tables, timezone=TZ, source="ERA5")

        def columns(directory, start):
            hourly, quarter = load_consolidated_dwd(
                directory, start_issue_date=start, end_issue_date=start,
                required_run="06", skipped_issue_dates=[], target_tz=TZ,
            )
            return set(build_dwd_features(hourly, quarter, target_tz=TZ).columns)

        self.assertEqual(columns(era5_dir, date(2026, 6, 1)), columns(icon_dir, date(2026, 6, 1)))

    def test_combine_keeps_icon_and_fills_gaps_with_era5(self):
        icon_dir = self.root / "icon" / "c3"
        icon_tables = prepare_era5.select_days(self.tables, ["2026-03-30"])
        icon_tables = {
            field: frame.assign(**{c: -1.0 for c in frame.columns if c.startswith("cluster_")})
            for field, frame in icon_tables.items()
        }
        update_history(icon_dir, icon_tables, timezone=TZ)
        combined_dir = self.root / "combined" / "c3"
        added = prepare_era5.combine_with_icon(icon_dir, self.tables, combined_dir, timezone=TZ)
        self.assertEqual(added, ["2026-03-28", "2026-03-29", "2026-03-31"])
        combined = read_history_tables(combined_dir)["u10"]
        self.assertTrue((combined.loc[combined.delivery_date == "2026-03-30", "cluster_0"] == -1.0).all())
        self.assertTrue((combined.loc[combined.delivery_date == "2026-03-31", "cluster_0"] != -1.0).all())
        self.assertEqual(
            json.loads((combined_dir / "era5_days.json").read_text())["delivery_dates"], added
        )
        self.assertTrue(history_is_complete(combined_dir))


if __name__ == "__main__":
    unittest.main()
