#!/usr/bin/env python3
"""Add ICON-D2 delivery days published in the upstream repository.

Fetches the ``upstream`` git remote and merges its consolidated
``data/icon/c{1,5,25}`` histories into the local ones. Only delivery days
that are missing locally are added; existing local days are never replaced.

Usage:
    python sync_upstream_weather.py            # fetch and merge
    python sync_upstream_weather.py --dry-run  # only report what would be added
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from experiment_config import load_experiment_config
from pipeline.dwd_history import (
    FORECAST_FIELDS,
    HISTORY_MANIFEST,
    history_path,
    read_history_tables,
    update_history,
)

REMOTE = "upstream"
BRANCH = "main"
CLUSTERS = (1, 5, 25)


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout


def _upstream_tables(cluster: int) -> dict[str, pd.DataFrame] | None:
    tables: dict[str, pd.DataFrame] = {}
    for field in FORECAST_FIELDS:
        spec = f"{REMOTE}/{BRANCH}:data/icon/c{cluster}/{field}.parquet"
        try:
            raw = _git("show", spec)
        except subprocess.CalledProcessError:
            return None
        frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
        frame["delivery_date"] = frame["delivery_date"].astype(str)
        frame["issue_date"] = frame["issue_date"].astype(str)
        frame["run_hour"] = frame["run_hour"].astype(str).str.zfill(2)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        tables[field] = frame
    return tables


def _timezone(cluster_dir: Path) -> str:
    try:
        manifest = json.loads((cluster_dir / HISTORY_MANIFEST).read_text("utf-8"))
        return str(manifest["timezone"])
    except (OSError, KeyError, ValueError):
        return "Europe/Berlin"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Fetching {REMOTE}/{BRANCH} ...", flush=True)
    _git("fetch", "--no-tags", REMOTE, BRANCH)

    config = load_experiment_config(REPO_ROOT)
    for cluster in CLUSTERS:
        cluster_dir = config.icon_cluster_dir(cluster)
        remote = _upstream_tables(cluster)
        if remote is None:
            print(f"C={cluster}: not published upstream, skipped.")
            continue

        local = read_history_tables(cluster_dir)
        local_keys = set()
        if all(not frame.empty for frame in local.values()):
            # A day counts as present only if every field has it locally.
            per_field = [
                set(zip(frame["delivery_date"], frame["run_hour"]))
                for frame in local.values()
            ]
            local_keys = set.intersection(*per_field)

        additions = {
            field: frame[
                [key not in local_keys for key in zip(frame["delivery_date"], frame["run_hour"])]
            ].reset_index(drop=True)
            for field, frame in remote.items()
        }
        new_days = sorted(set(additions[FORECAST_FIELDS[0]]["delivery_date"]))
        if not new_days:
            print(f"C={cluster}: up to date.")
            continue
        print(f"C={cluster}: {len(new_days)} new day(s): {', '.join(new_days)}")
        if args.dry_run:
            continue
        update_history(cluster_dir, additions, timezone=_timezone(cluster_dir))

    if not args.dry_run:
        for cluster in CLUSTERS:
            path = config.icon_cluster_dir(cluster) / HISTORY_MANIFEST
            if path.is_file():
                manifest = json.loads(path.read_text("utf-8"))
                print(
                    f"C={cluster}: {manifest['first_delivery_date']} .. "
                    f"{manifest['last_delivery_date']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
