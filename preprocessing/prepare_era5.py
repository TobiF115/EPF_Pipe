#!/usr/bin/env python3
"""Aggregate downloaded ERA5 data and export it in the ICON-D2 history format.

Steps (each can be skipped):

1. ``aggregate``: run ``aggregate_era5.py`` for every year and cluster count.
   Output: ``ERA5_DATA_ROOT/c{C}/{year}/*.csv`` (used by the LEAR ERA5 runs).
2. ``history``: convert those CSVs into the consolidated history format of
   ``data/icon`` (``u10``, ``v10``, ``ASWDIR_S``, ``ASWDIFD_S`` Parquet files
   plus ``dwd_history.json``) under ``data/era5_icon/c{C}``. Every model that
   reads ICON-D2 histories can use it via ``ICON_DATA_ROOT=data/era5_icon``.
3. ``--combine``: additionally write ``data/era5_icon_combined/c{C}``: all
   ICON-D2 days from ``data/icon`` plus ERA5 for every delivery day ICON-D2
   does not cover. ``era5_days.json`` lists the days taken from ERA5.

Variable mapping (ERA5 -> ICON-D2 field)::

    u10, v10        -> u10, v10                (hourly, instantaneous)
    fdir            -> ASWDIR_S                (direct, horizontal plane)
    max(ssrd-fdir,0)-> ASWDIFD_S               (diffuse = global - direct)

ERA5 radiation is an hourly mean flux labelled with its interval start; each
hour is repeated for its four quarter-hours, which preserves the energy.
ERA5 rows use ``issue_date = delivery_date - 1`` and ``run_hour = "06"`` so
that the existing loaders select them.

Usage::

    python preprocessing/prepare_era5.py                 # aggregate + history
    python preprocessing/prepare_era5.py --combine       # also merge with ICON-D2
    python preprocessing/prepare_era5.py --skip-aggregate --combine
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiment_config import load_experiment_config
from pipeline.dwd_history import (
    FORECAST_FIELDS,
    HISTORY_MANIFEST,
    METADATA_COLUMNS,
    read_history_tables,
    update_history,
    validate_delivery_tables,
)
from pipeline.lear.lear_model import load_era5

CLUSTERS = (1, 5, 25)
ERA5_RUN_HOUR = "06"
ERA5_DAYS_FILE = "era5_days.json"
DEFAULT_HISTORY_ROOT = REPO_ROOT / "data" / "era5_icon"
DEFAULT_COMBINED_ROOT = REPO_ROOT / "data" / "era5_icon_combined"


# ====================================================================
# Conversion (pure functions, unit tested)
# ====================================================================
def _cluster_ids(frame: pd.DataFrame, prefix: str) -> list[str]:
    ids = [c[len(prefix) + 1:] for c in frame.columns if c.startswith(prefix + "_cluster_")]
    return sorted(ids, key=lambda value: int(value.rsplit("_", 1)[-1]))


def era5_to_history_tables(
    df_era5: pd.DataFrame,
    *,
    timezone: str,
) -> dict[str, pd.DataFrame]:
    """Convert ``load_era5`` output into the four ICON-D2 history tables.

    Only local delivery days with complete hourly coverage are emitted.
    """
    if df_era5.index.tz is None:
        raise ValueError("ERA5 frame needs a timezone-aware index.")
    frame = df_era5.sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="first")]
    clusters = _cluster_ids(frame, "u10")
    for name in ("v10", "ssrd", "fdir"):
        if _cluster_ids(frame, name) != clusters:
            raise ValueError(f"ERA5 variable {name} is missing or has other clusters.")
    cluster_cols = clusters  # e.g. "cluster_0"

    utc = frame.index.tz_convert("UTC")
    tables_hourly = {
        "u10": frame[[f"u10_{c}" for c in cluster_cols]].to_numpy(dtype=float),
        "v10": frame[[f"v10_{c}" for c in cluster_cols]].to_numpy(dtype=float),
    }
    ssrd = frame[[f"ssrd_{c}" for c in cluster_cols]].to_numpy(dtype=float)
    fdir = frame[[f"fdir_{c}" for c in cluster_cols]].to_numpy(dtype=float)
    tables_hourly["ASWDIR_S"] = np.clip(fdir, 0.0, None)
    tables_hourly["ASWDIFD_S"] = np.clip(ssrd - fdir, 0.0, None)

    local_dates = utc.tz_convert(timezone).normalize()
    candidate_days = sorted({ts.date() for ts in local_dates})
    available = set(utc)

    out: dict[str, list[pd.DataFrame]] = {field: [] for field in FORECAST_FIELDS}
    for day in candidate_days:
        start = pd.Timestamp(day, tz=timezone)
        hours = pd.date_range(start, start + pd.DateOffset(days=1), freq="1h", inclusive="left")
        hours_utc = hours.tz_convert("UTC")
        if not set(hours_utc).issubset(available):
            continue
        positions = utc.get_indexer(hours_utc)
        meta = {
            "delivery_date": day.isoformat(),
            "issue_date": (day - timedelta(days=1)).isoformat(),
            "run_hour": ERA5_RUN_HOUR,
        }
        day_values = {field: values[positions] for field, values in tables_hourly.items()}
        if not all(np.isfinite(v).all() for v in day_values.values()):
            continue
        for field, values in day_values.items():
            if field in ("ASWDIR_S", "ASWDIFD_S"):
                stamps = hours_utc.repeat(4) + pd.to_timedelta(
                    np.tile([0, 15, 30, 45], len(hours_utc)), unit="min"
                )
                values = np.repeat(values, 4, axis=0)
            else:
                stamps = hours_utc
            table = pd.DataFrame(values, columns=cluster_cols)
            table.insert(0, "timestamp", stamps)
            for key in reversed(("delivery_date", "issue_date", "run_hour")):
                table.insert(0, key, meta[key])
            out[field].append(table.loc[:, [*METADATA_COLUMNS, *cluster_cols]])

    if not out["u10"]:
        raise ValueError("ERA5 data contain no complete delivery day.")
    return {
        field: pd.concat(parts, ignore_index=True) for field, parts in out.items()
    }


def missing_delivery_days(
    era5_tables: Mapping[str, pd.DataFrame],
    icon_tables: Mapping[str, pd.DataFrame],
) -> list[str]:
    """Delivery days present in ERA5 but absent from (any field of) ICON-D2."""
    era5_days = set(era5_tables["u10"]["delivery_date"].astype(str))
    if any(frame.empty for frame in icon_tables.values()):
        return sorted(era5_days)
    icon_days = set.intersection(
        *(set(frame["delivery_date"].astype(str)) for frame in icon_tables.values())
    )
    return sorted(era5_days - icon_days)


def select_days(tables: Mapping[str, pd.DataFrame], days: Iterable[str]) -> dict[str, pd.DataFrame]:
    wanted = set(days)
    return {
        field: frame.loc[frame["delivery_date"].astype(str).isin(wanted)].reset_index(drop=True)
        for field, frame in tables.items()
    }


def validate_all_days(tables: Mapping[str, pd.DataFrame], *, timezone: str) -> int:
    days = sorted(set(tables["u10"]["delivery_date"].astype(str)))
    for value in days:
        validate_delivery_tables(
            tables,
            delivery_date=date.fromisoformat(value),
            run_hour=ERA5_RUN_HOUR,
            timezone=timezone,
        )
    return len(days)


# ====================================================================
# Writing
# ====================================================================
def write_history(cluster_dir: Path, tables: Mapping[str, pd.DataFrame], *, timezone: str,
                  source: str) -> None:
    """Replace ``cluster_dir`` with a fresh consolidated history."""
    if cluster_dir.exists():
        for field in FORECAST_FIELDS:
            (cluster_dir / f"{field}.parquet").unlink(missing_ok=True)
        (cluster_dir / HISTORY_MANIFEST).unlink(missing_ok=True)
    update_history(cluster_dir, tables, timezone=timezone)
    manifest_path = cluster_dir / HISTORY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = source
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def combine_with_icon(
    icon_dir: Path,
    era5_tables: Mapping[str, pd.DataFrame],
    combined_dir: Path,
    *,
    timezone: str,
) -> list[str]:
    """Copy the ICON-D2 history and add ERA5 for days ICON-D2 lacks."""
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True)
    for field in FORECAST_FIELDS:
        source = icon_dir / f"{field}.parquet"
        if source.is_file():
            shutil.copy2(source, combined_dir / source.name)
    if (icon_dir / HISTORY_MANIFEST).is_file():
        shutil.copy2(icon_dir / HISTORY_MANIFEST, combined_dir / HISTORY_MANIFEST)
    days = missing_delivery_days(era5_tables, read_history_tables(combined_dir))
    if days:
        update_history(combined_dir, select_days(era5_tables, days), timezone=timezone)
    (combined_dir / ERA5_DAYS_FILE).write_text(
        json.dumps({"source": "ERA5", "delivery_dates": days}, indent=2) + "\n",
        encoding="utf-8",
    )
    return days


# ====================================================================
# Orchestration
# ====================================================================
def run_aggregation(raw_root: Path, era5_root: Path, years: list[int], clusters: Iterable[int]) -> list[int]:
    """Run aggregate_era5.py per cluster count and year; return years with data."""
    from preprocessing.aggregate_era5 import NO_DATA_EXIT_CODE

    script = REPO_ROOT / "preprocessing" / "aggregate_era5.py"
    with_data: set[int] = set()
    for cluster in clusters:
        for year in years:
            out_dir = era5_root / f"c{cluster}" / str(year)
            # Old CSVs with another start date would be loaded twice.
            if out_dir.is_dir():
                for old in out_dir.glob("*.csv"):
                    old.unlink()
            env = os.environ.copy()
            env.update({
                "ERA5_RAW_ARCHIVE": str(raw_root),
                "ERA5_DATA_ROOT": str(era5_root),
                "ERA5_PREPROCESS_YEAR": str(year),
                "ERA5_N_CLUSTERS": str(cluster),
                # aggregate_era5 cuts at min(year end, EVALUATION_END_DATE);
                # the real end of the data is detected from the files.
                "EVALUATION_END_DATE": date(year, 12, 31).isoformat(),
                "PYTHONIOENCODING": "utf-8",
            })
            print(f"\n=== aggregate_era5 C={cluster} year={year}", flush=True)
            result = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, env=env)
            if result.returncode == NO_DATA_EXIT_CODE:
                print(f"    keine ERA5-Daten für {year}, übersprungen")
                continue
            if result.returncode != 0:
                raise RuntimeError(f"aggregate_era5.py C={cluster} {year} failed ({result.returncode}).")
            with_data.add(year)
    return sorted(with_data)


def _years_with_csv(era5_root: Path, cluster: int, years: Iterable[int]) -> list[int]:
    return [y for y in years if any((era5_root / f"c{cluster}" / str(y)).glob("*.csv"))]


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-root", default=None,
                        help="Ordner mit ERA5-GRIB-Dateien (Standard: ERA5_RAW_ARCHIVE bzw. data/era5_raw)")
    parser.add_argument("--clusters", default="1,5,25")
    parser.add_argument("--years", default=None,
                        help="Kommagetrennt (Standard: 2025 bis aktuelles Jahr)")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    parser.add_argument("--combine", action="store_true",
                        help="Zusätzlich ICON-D2 + ERA5-Lückenfüllung schreiben.")
    parser.add_argument("--combined-root", default=str(DEFAULT_COMBINED_ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_experiment_config(REPO_ROOT)
    clusters = [int(v) for v in args.clusters.split(",")]
    from preprocessing.download_era5 import resolve_raw_root

    raw_root = resolve_raw_root(args.raw_root)
    era5_root = config.era5_data_root
    timezone = config.timezone

    if args.years:
        years = [int(v) for v in args.years.split(",")]
    else:
        years = list(range(2025, date.today().year + 1))
    if not args.skip_aggregate:
        run_aggregation(raw_root, era5_root, years, clusters)

    history_root = _resolve(args.history_root)
    for cluster in clusters:
        cluster_years = _years_with_csv(era5_root, cluster, years)
        if not cluster_years:
            raise FileNotFoundError(f"Keine aggregierten ERA5-CSVs für C={cluster} unter {era5_root}.")
        dirs = [era5_root / f"c{cluster}" / str(year) for year in cluster_years]
        tables = era5_to_history_tables(load_era5(dirs, target_tz=timezone), timezone=timezone)
        count = validate_all_days(tables, timezone=timezone)
        write_history(history_root / f"c{cluster}", tables, timezone=timezone, source="ERA5")
        first, last = tables["u10"]["delivery_date"].iloc[[0, -1]]
        print(f"ERA5-Verlauf C={cluster}: {count} Tage, {first} .. {last} -> {history_root / f'c{cluster}'}")

        if args.combine:
            days = combine_with_icon(
                config.icon_cluster_dir(cluster),
                tables,
                _resolve(args.combined_root) / f"c{cluster}",
                timezone=timezone,
            )
            print(f"Kombiniert C={cluster}: {len(days)} ERA5-Tage ergänzt "
                  f"({days[0] if days else '-'} .. {days[-1] if days else '-'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
