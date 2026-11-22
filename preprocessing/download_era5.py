#!/usr/bin/env python3
"""Download ERA5 inputs in the format read by ``aggregate_era5.py``.

Each request contains all four model variables (10 m wind components,
global and direct solar radiation) for one month as a single GRIB file::

    ERA5_RAW_ARCHIVE/monthly/era5_YYYY_MM.grib

``aggregate_era5.py`` reads every ``*.grib`` below ``ERA5_RAW_ARCHIVE``, so a
manual CDS download in the same format (any time range) can simply be placed
in that folder as well; overlapping hours are de-duplicated.

Requirements: ``pip install cdsapi`` and a CDS API key in ``~/.cdsapirc``
(https://cds.climate.copernicus.eu/how-to-api). The ERA5 licence must be
accepted once on the dataset page.

Usage::

    python preprocessing/download_era5.py                  # 2025-01-01 .. today-6d
    python preprocessing/download_era5.py --start 2026-11-01
    python preprocessing/download_era5.py --dry-run
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET = "reanalysis-era5-single-levels"

# N, W, S, E: bounding box of Germany (lat 47.27-55.06, lon 5.87-15.04).
# Every C=1/5/25 cluster receives >= 33 ERA5 points with this box.
AREA = [55, 5.75, 47.25, 15]

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
]

# ERA5T is published with roughly five days of delay.
DEFAULT_LATENCY_DAYS = 6
DEFAULT_START = date(2025, 1, 1)


@dataclass(frozen=True)
class MonthRequest:
    year: int
    month: int
    days: tuple[int, ...]

    @property
    def filename(self) -> str:
        return f"era5_{self.year}_{self.month:02d}.grib"

    def payload(self) -> dict:
        return {
            "product_type": ["reanalysis"],
            "variable": list(VARIABLES),
            "year": [str(self.year)],
            "month": [f"{self.month:02d}"],
            "day": [f"{day:02d}" for day in self.days],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": list(AREA),
        }


def month_requests(start: date, end: date) -> list[MonthRequest]:
    """Return one request per month covering ``start..end`` inclusive."""
    if end < start:
        raise ValueError(f"End date {end} is before start date {start}.")
    requests: list[MonthRequest] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        last_day = calendar.monthrange(year, month)[1]
        first = start.day if (year, month) == (start.year, start.month) else 1
        last = end.day if (year, month) == (end.year, end.month) else last_day
        requests.append(MonthRequest(year, month, tuple(range(first, last + 1))))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return requests


def _is_grib(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as handle:
        return handle.read(4) == b"GRIB"


def _days_marker(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".days")


def monthly_file_complete(path: Path, request: MonthRequest) -> bool:
    """A monthly file is reused only if it holds exactly the requested days."""
    marker = _days_marker(path)
    if not _is_grib(path) or not marker.is_file():
        return False
    return marker.read_text(encoding="utf-8").strip() == ",".join(map(str, request.days))


def download_month(client, request: MonthRequest, monthly_dir: Path) -> Path:
    target = monthly_dir / request.filename
    if monthly_file_complete(target, request):
        print(f"  vorhanden: {target.name}", flush=True)
        return target
    monthly_dir.mkdir(parents=True, exist_ok=True)
    # The suffix keeps an unfinished file out of aggregate_era5's *.grib glob.
    partial = target.with_name(target.name + ".part")
    partial.unlink(missing_ok=True)
    print(
        f"  lade {target.name} (Tage {request.days[0]}-{request.days[-1]}) ...",
        flush=True,
    )
    client.retrieve(DATASET, request.payload()).download(str(partial))
    if not _is_grib(partial):
        head = partial.read_bytes()[:200] if partial.is_file() else b""
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"CDS lieferte keine GRIB-Datei für {target.name} "
            f"(evtl. Zip oder Fehlermeldung): {head!r}"
        )
    os.replace(partial, target)
    _days_marker(target).write_text(",".join(map(str, request.days)), encoding="utf-8")
    return target


def resolve_raw_root(value: str | None) -> Path:
    if value:
        path = Path(value)
    else:
        from experiment_config import _load_env_file

        _load_env_file(REPO_ROOT / ".env")
        configured = os.getenv("ERA5_RAW_ARCHIVE", "")
        if not configured or configured.startswith("/path/to"):
            configured = "data/era5_raw"
        path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help=f"Letzter Tag (Standard: heute - {DEFAULT_LATENCY_DAYS} Tage).",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Zielordner (Standard: ERA5_RAW_ARCHIVE aus .env, sonst data/era5_raw).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    end = args.end or date.today() - timedelta(days=DEFAULT_LATENCY_DAYS)
    monthly_dir = resolve_raw_root(args.raw_root) / "monthly"
    requests = month_requests(args.start, end)

    print(f"ERA5 {args.start} .. {end} -> {monthly_dir}", flush=True)
    print(f"{len(requests)} Monatsanfragen, Gebiet {AREA}", flush=True)
    if args.dry_run:
        for request in requests:
            done = monthly_file_complete(monthly_dir / request.filename, request)
            print(f"  {request.filename}: {len(request.days)} Tage [{'ok' if done else 'fehlt'}]")
        return 0

    import cdsapi

    client = cdsapi.Client()
    for request in requests:
        download_month(client, request, monthly_dir)
    print(f"Fertig. Letzter Tag: {end}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
