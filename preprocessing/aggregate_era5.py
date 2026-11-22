#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_era5.py
=================

Aggregates ERA5 weather data to ICON-D2 spatial clusters and saves
the results as CSV files (one per variable and year).

Input files
-----------
Any number of ERA5 GRIB files, each holding any mix of variables and any
time range (e.g. one CDS download with u10, v10, ssrd, fdir for 2025-2026,
plus monthly follow-up downloads). Files are taken from, in this order:

1. ``ERA5_FILES``: list of paths separated by ``os.pathsep`` (``;`` on Windows);
2. the legacy pair ``ERA5_MAIN_FILE`` / ``ERA5_SOLAR_FILE`` (or
   ``era5_main_YYYY.grib`` / ``era5_solar_YYYY.grib``) if both exist;
3. every ``*.grib`` below ``ERA5_RAW_ARCHIVE``.

Instantaneous fields (wind, ...) and accumulated fields (radiation) are read
separately via the GRIB ``stepType`` key, so they may share one file.
Overlapping timestamps across files are de-duplicated.

Solar handling
--------------
ERA5 ssrd / fdir have dims (time, step, lat, lon) where each step
represents one hourly interval energy [J/m²]. The (time, step)
dimensions are flattened to a true hourly time axis via valid_time,
and values are converted from J/m² to W/m² by division by 3600.
Hours that are absent from the download (all-NaN fields at the edges of
the forecast hypercube) are dropped instead of being filled with zeros.
"""

# ====================================================================
# IMPORTS
# ====================================================================
import gc
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


# Do not let optional Unicode status symbols abort long Windows runs when the
# active console uses a legacy encoding such as cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


# ====================================================================
# CONFIG
# ====================================================================
BASE_DIR = Path(__file__).parent.parent  # points to repository root
sys.path.insert(0, str(BASE_DIR))
from experiment_config import load_experiment_config

EXPERIMENT = load_experiment_config(BASE_DIR)

# Output year and cluster count.  The historical launcher overrides both for
# each required combination.
ERA5_PREPROCESS_YEAR = int(os.getenv("ERA5_PREPROCESS_YEAR", "2026"))
N_CLUSTERS = int(os.getenv("ERA5_N_CLUSTERS", "25"))

# Raw ERA5 GRIB files are not included in the repository due to file size.
_raw_archive = os.getenv("ERA5_RAW_ARCHIVE", "")
if not _raw_archive or _raw_archive.startswith("/path/to"):
    _raw_archive = "data/era5_raw"
ERA5_RAW_ARCHIVE = Path(_raw_archive)
if not ERA5_RAW_ARCHIVE.is_absolute():
    ERA5_RAW_ARCHIVE = BASE_DIR / ERA5_RAW_ARCHIVE
ERA5_MAIN_FILE = Path(os.getenv(
    "ERA5_MAIN_FILE", str(ERA5_RAW_ARCHIVE / f"era5_main_{ERA5_PREPROCESS_YEAR}.grib")
))
ERA5_SOLAR_FILE = Path(os.getenv(
    "ERA5_SOLAR_FILE", str(ERA5_RAW_ARCHIVE / f"era5_solar_{ERA5_PREPROCESS_YEAR}.grib")
))

CLUSTER_FILE = BASE_DIR / "data" / "clustering" / f"icon_d2_clustering_c{N_CLUSTERS}.parquet"

# Output directory for aggregated CSVs.
OUTPUT_PARENT = EXPERIMENT.era5_year_dir(N_CLUSTERS, ERA5_PREPROCESS_YEAR)

# ERA5 points farther than this (degrees) from every clustered ICON-D2 point
# are outside Germany + 50 km and are ignored, matching the ICON-D2 mask.
# Set ERA5_MAX_CLUSTER_DISTANCE_DEG=none to assign every point (old behaviour).
_max_distance = os.getenv("ERA5_MAX_CLUSTER_DISTANCE_DEG", "0.1").strip().lower()
MAX_CLUSTER_DISTANCE_DEG = None if _max_distance in ("", "none", "inf") else float(_max_distance)

# Required variables are used by the models; optional ones are aggregated
# only if they are present in the downloaded files.
MAIN_VARIABLES = ["u10", "v10"]
OPTIONAL_MAIN_VARIABLES = ["t2m", "sp", "sd"]
SOLAR_VARIABLES = ["ssrd", "fdir"]

# Exit code used when the requested year contains no ERA5 data at all.
NO_DATA_EXIT_CODE = 3

_year_start = date(ERA5_PREPROCESS_YEAR, 1, 1)
_year_end = date(ERA5_PREPROCESS_YEAR, 12, 31)
START_TIME = pd.Timestamp(max(_year_start, EXPERIMENT.weather_training_start))
END_TIME = pd.Timestamp(min(_year_end, EXPERIMENT.evaluation_end)) + pd.Timedelta(hours=23, minutes=45)


# ====================================================================
# HELPERS
# ====================================================================
def safe_unit_for_filename(units: str) -> str:
    """
    Convert a physical unit string into a filesystem-safe representation.

    Replaces characters that are illegal or awkward in file names with
    safe alternatives (e.g. ``"/"`` → ``"-per-"``, ``"²"`` → ``"2"``).
    """
    return (
        str(units)
        .replace("/", "-per-")
        .replace(" ", "_")
        .replace("*", "")
        .replace("²", "2")
    )


def build_era5_cluster_grid(cluster_df, lats, lons, max_distance_deg=MAX_CLUSTER_DISTANCE_DEG):
    """
    Map every ERA5 grid point to the nearest ICON-D2 cluster centroid.

    Uses a KD-tree nearest-neighbour search in lon/lat space to assign
    each ERA5 (lat, lon) cell to a cluster ID from the ICON-D2 clustering.
    ERA5 points farther than ``max_distance_deg`` from every clustered ICON
    point lie outside Germany + buffer and get ID ``-1`` (ignored).
    ``None`` disables the mask.

    Returns
    -------
    numpy.ndarray of shape (Y, X)
        Integer cluster IDs, one per ERA5 grid cell (``-1`` = ignored).
    """
    icon_coords = cluster_df[["lon", "lat"]].to_numpy()
    icon_ids = cluster_df["cluster_id"].to_numpy()
    tree = cKDTree(icon_coords)

    lon_g, lat_g = np.meshgrid(lons, lats)
    era5_coords = np.column_stack([lon_g.ravel(), lat_g.ravel()])
    distance, idx = tree.query(era5_coords)

    ids = icon_ids[idx].astype(int)
    if max_distance_deg is not None:
        ids[distance > max_distance_deg] = -1
    return ids.reshape(lat_g.shape)


def extract_main_values_and_times(da: xr.DataArray):
    """Return values ``(T, lat, lon)`` and timestamps of an instantaneous field."""
    times = pd.to_datetime(np.atleast_1d(da["valid_time"].values if "valid_time" in da.coords
                                         else da.time.values))
    values = da.values
    if values.ndim == 2:  # a single timestep
        values = values[np.newaxis]
    return values, times


def extract_solar_values_and_times(ds: xr.Dataset, da: xr.DataArray):
    """
    Flatten the (time, step) dimensions of an ERA5 solar DataArray.

    ERA5 solar variables (ssrd, fdir) have shape ``(time, step, lat, lon)``
    where each step represents one hourly interval energy in J/m².  This
    function merges the two leading dimensions into a single true hourly
    time axis derived from ``valid_time``.

    Returns
    -------
    values : numpy.ndarray of shape (T*S, lat, lon)
        Flattened interval energies in J/m².
    times : pandas.DatetimeIndex of length T*S
        True hourly timestamps, shifted back by one hour so that each
        value is labelled with the start of its accumulation interval.
    """
    vals = da.values  # (T, S, lat, lon)
    T, S, Y, X = vals.shape

    values = vals.reshape(T * S, Y, X)
    # Implementation for: solar_rate (t) = solar_energy( t to t+1) / 3600
    times = pd.to_datetime(ds["valid_time"].values.reshape(T * S)) - pd.Timedelta(hours=1)

    return values, times


def aggregate_to_clusters(values_tll, cluster_grid, n_clusters):
    """
    Average ERA5 grid-point values within each cluster for every timestep.

    Returns
    -------
    numpy.ndarray of shape (T, n_clusters)
        Spatially averaged values per cluster and timestep.  Clusters
        with no grid points (or all-NaN fields) remain ``NaN``.
    """
    T = values_tll.shape[0]
    out = np.full((T, n_clusters), np.nan)

    for c in range(n_clusters):
        mask = cluster_grid == c
        if np.any(mask):
            with np.errstate(invalid="ignore"):
                with_nan = values_tll[:, mask]
                valid = np.isfinite(with_nan).any(axis=1)
                out[valid, c] = np.nanmean(with_nan[valid], axis=1)

    return out


def combine_parts(parts):
    """
    Concatenate ``(times, values)`` pieces from several files.

    Rows whose clusters are all NaN (hours missing from a download) are
    dropped, the result is sorted, and repeated timestamps keep the first
    complete occurrence.
    """
    if not parts:
        return pd.DatetimeIndex([]), np.empty((0, 0))
    times = pd.DatetimeIndex(np.concatenate([np.asarray(t, dtype="datetime64[ns]") for t, _ in parts]))
    values = np.concatenate([v for _, v in parts], axis=0)
    keep = ~np.all(np.isnan(values), axis=1)
    times, values = times[keep], values[keep]
    order = np.argsort(times.values, kind="stable")
    times, values = times[order], values[order]
    unique = ~times.duplicated(keep="first")
    return times[unique], values[unique]


def write_icon_style_csv(outpath, times, out, var, long_name, units, n_clusters, transformation):
    """
    Write cluster-aggregated ERA5 data to a CSV file with a metadata header.

    The file begins with comment lines (prefixed ``"#"``) describing the
    source, variable, units, and transformation applied.  The data block
    follows in standard CSV format with a ``"timestamp"`` index column and
    one column per cluster (``"cluster_0"``, ``"cluster_1"``, …).
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(out, index=times, columns=[f"cluster_{i}" for i in range(n_clusters)])
    df.index.name = "timestamp"

    header = [
        "# Source: ERA5",
        f"# Variable: {var}",
        f"# Long name: {long_name}",
        f"# Units: {units}",
        f"# Transformation: {transformation}",
        f"# Clusters: {n_clusters}",
    ]

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(f)

    print(f"✅ Saved {outpath.name} ({df.shape[0]} × {df.shape[1]})")


# ====================================================================
# INPUT DISCOVERY
# ====================================================================
def input_files():
    """Return the GRIB files to read (see module docstring for the order)."""
    explicit = os.getenv("ERA5_FILES")
    if explicit:
        files = [Path(p) for p in explicit.split(os.pathsep) if p.strip()]
    elif ERA5_MAIN_FILE.is_file() and ERA5_SOLAR_FILE.is_file():
        files = [ERA5_MAIN_FILE, ERA5_SOLAR_FILE]
    else:
        files = sorted(p for p in ERA5_RAW_ARCHIVE.rglob("*.grib") if p.is_file())
    missing = [p for p in files if not p.is_file()]
    if missing or not files:
        raise FileNotFoundError(
            f"No ERA5 GRIB input found (looked in {ERA5_RAW_ARCHIVE}); missing: {missing}"
        )
    return files


def open_step_type(path, step_type):
    """Open only the GRIB messages of one stepType; ``None`` if there are none."""
    try:
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": "", "filter_by_keys": {"stepType": step_type}},
        )
    except Exception as exc:  # cfgrib raises various errors for empty selections
        print(f"   ({path.name}: no '{step_type}' fields: {type(exc).__name__})")
        return None
    if not ds.data_vars:
        ds.close()
        return None
    return ds


# ====================================================================
# PROCESSING
# ====================================================================
def collect_variables(files, step_type, variables, cluster_df, n_clusters, solar_mode):
    """Aggregate ``variables`` from every file to clusters, combined over files."""
    parts = {var: [] for var in variables}
    attrs = {}
    grid_cache = {}
    for path in files:
        ds = open_step_type(path, step_type)
        if ds is None:
            continue
        try:
            key = (tuple(ds.latitude.values), tuple(ds.longitude.values))
            if key not in grid_cache:
                grid_cache[key] = build_era5_cluster_grid(
                    cluster_df, ds.latitude.values, ds.longitude.values
                )
            cluster_grid = grid_cache[key]
            for var in variables:
                if var not in ds:
                    continue
                da = ds[var]
                if solar_mode:
                    values_tll, times = extract_solar_values_and_times(ds, da)
                else:
                    values_tll, times = extract_main_values_and_times(da)
                parts[var].append((times, aggregate_to_clusters(values_tll, cluster_grid, n_clusters)))
                attrs.setdefault(var, dict(da.attrs))
                print(f"   {path.name}: {var} {len(times)} Zeitschritte")
                del values_tll
                gc.collect()
        finally:
            ds.close()
    return {var: combine_parts(p) for var, p in parts.items() if p}, attrs


def write_year(var, times, out, attrs, n_clusters, solar_mode, start_time, end_time):
    """Write one variable for the configured year; return False if no data."""
    mask = (times >= start_time) & (times <= end_time)
    if not np.any(mask):
        print(f"⚠️  {var}: keine Daten zwischen {start_time} und {end_time}")
        return False
    times = times[mask]
    out = out[mask]
    if solar_mode:
        out = np.nan_to_num(out) / 3600.0
        units = "W m-2"
        transformation = "hourly mean flux (ERA5 interval energy)"
    else:
        units = attrs.get("units", "")
        transformation = "raw"
    expected = pd.date_range(times[0], times[-1], freq="1h")
    if len(expected) != len(times):
        print(f"⚠️  {var}: {len(expected) - len(times)} fehlende Stunden im Zeitraum")
    outname = f"{var}_{safe_unit_for_filename(units)}_{times[0].strftime('%Y%m%d%H')}.csv"
    write_icon_style_csv(
        OUTPUT_PARENT / outname, times, out, var,
        attrs.get("long_name", var), units, n_clusters, transformation,
    )
    return True


# ====================================================================
# MAIN
# ====================================================================
def main():
    """
    Aggregate all ERA5 inputs for ``ERA5_PREPROCESS_YEAR`` and ``ERA5_N_CLUSTERS``.

    Returns ``NO_DATA_EXIT_CODE`` if the year has no ERA5 data.
    """
    cluster_df = pd.read_parquet(CLUSTER_FILE)
    n_clusters = int(cluster_df["cluster_id"].max()) + 1
    files = input_files()
    print(f"ERA5 C={N_CLUSTERS}, Jahr {ERA5_PREPROCESS_YEAR}: {len(files)} Datei(en)")

    main_data, main_attrs = collect_variables(
        files, "instant", MAIN_VARIABLES + OPTIONAL_MAIN_VARIABLES,
        cluster_df, n_clusters, solar_mode=False,
    )
    solar_data, solar_attrs = collect_variables(
        files, "accum", SOLAR_VARIABLES, cluster_df, n_clusters, solar_mode=True,
    )
    missing = [v for v in MAIN_VARIABLES if v not in main_data] + [
        v for v in SOLAR_VARIABLES if v not in solar_data
    ]
    if missing:
        raise RuntimeError(f"Required ERA5 variables missing from the input files: {missing}")
    skipped = [v for v in OPTIONAL_MAIN_VARIABLES if v not in main_data]
    if skipped:
        print(f"ℹ️  Optionale Variablen nicht vorhanden, übersprungen: {skipped}")

    written = 0
    for data, attrs, solar_mode in ((main_data, main_attrs, False), (solar_data, solar_attrs, True)):
        for var, (times, out) in data.items():
            written += write_year(var, times, out, attrs[var], n_clusters, solar_mode,
                                  START_TIME, END_TIME)
    if written == 0:
        return NO_DATA_EXIT_CODE
    print("\n🎉 ERA5 → ICON aggregation finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
