#!/usr/bin/env python3
"""Backtest the online LSTM against the rolling EXAA LSTM and LEAR D56 C5.

All models are evaluated on the same delivery days with the same inputs
(C5 weather, load, EPEX lags, target-day EXAA):

* ``online``: pretrained once on ``--weather-root`` up to ``--pretrain-end``,
  then fine-tuned day by day (shifting window) exactly as in the pipeline.
* ``rolling``: the existing ``LSTM_exaa_enriched``, refitted from scratch for
  every day on all days since ``--train-start`` (expanding window).
* ``lear``: LEAR D56 C5 point forecasts read from ``--lear-forecast``.

Without ERA5 data, ``--weather-root data/icon`` pretrains on ICON-D2, which
gives the like-for-like comparison with the existing DWD-trained models.

Usage::

    python evaluation/backtest_lstm_era5_online.py --weather-root data/icon \
        --train-start 2026-07-13 --pretrain-end 2026-10-10 \
        --start 2026-10-11 --end 2026-11-19
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.mlp.LSTM_era5_online import (  # noqa: E402
    OnlineLSTMConfig,
    OnlineLSTMStore,
    advance_state,
    era5_pretrain_config,
    forecast_state,
    pretrain_state,
)
from pipeline.mlp.LSTM_exaa_enriched import (  # noqa: E402
    LSTMConfig,
    LSTMExaaEnrichedConfig,
    rolling_lstm_exaa_enriched_forecast,
)
from pipeline.operational import runner  # noqa: E402
from pipeline.operational.config import load_operational_config  # noqa: E402


def _metrics(errors: np.ndarray) -> dict:
    errors = errors[np.isfinite(errors)]
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "Bias": float(np.mean(errors)),
        "observations": int(errors.size),
    }


def _lear_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["delivery_date"] = frame["delivery_date"].astype(str)
    return frame.pivot(index="delivery_date", columns="mtu", values="y_pred")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weather-root", default="data/era5_icon",
                        help="History used for pretraining (data/icon = ICON-D2 stand-in).")
    parser.add_argument("--train-start", type=date.fromisoformat, required=True)
    parser.add_argument("--pretrain-end", type=date.fromisoformat, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--lear-forecast",
        default="results/operational/lear_op_results/dwd/d56/c5/fundamental/forecast.csv",
    )
    parser.add_argument("--skip-rolling", action="store_true",
                        help="Do not refit the rolling EXAA LSTM (slow).")
    parser.add_argument(
        "--pretrain-configs", default="era5",
        help="Comma-separated optimiser settings to compare: 'era5' (mini-batches, "
             "stepwise decay, forget bias, clipping) and/or 'legacy' (full batch, constant rate).",
    )
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated random seeds; each config is pretrained once per seed.")
    parser.add_argument("--output", default=None, help="Output stem (default results/operational/mlp/...).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.train_start <= args.pretrain_end < args.start <= args.end:
        raise SystemExit("Require train_start <= pretrain_end < start <= end.")
    weather_root = Path(args.weather_root)
    weather_root = weather_root if weather_root.is_absolute() else REPO_ROOT / weather_root
    os.environ.setdefault("LSTM_ERA5_WEATHER_ROOT", str(weather_root))

    config = replace(load_operational_config(REPO_ROOT), point_variant="exaa_enriched")
    online_config = OnlineLSTMConfig.from_env()
    pretrain_bases = {"era5": era5_pretrain_config(), "legacy": LSTMConfig()}
    config_names = [name.strip() for name in args.pretrain_configs.split(",") if name.strip()]
    unknown = sorted(set(config_names) - set(pretrain_bases))
    if unknown or not config_names:
        raise SystemExit(f"Unknown --pretrain-configs {unknown}; choose from {sorted(pretrain_bases)}.")
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    print(f"Fine-tuning: {online_config.finetune_epochs} epochs, "
          f"lr {online_config.finetune_learning_rate:g}, window {online_config.finetune_window_days} d")

    # ---- inputs (same for all models) ---------------------------------
    prices, exaa, load = runner.load_market_data(
        config, start_date=args.train_start - timedelta(days=7), target_date=args.end
    )
    pretrain_weather = runner._load_weather_features(
        replace(config, icon_data_root=weather_root), 5, args.train_start, args.pretrain_end
    )
    X_pre, Y_pre = runner._online_lstm_matrices(
        config, prices=prices, exaa=exaa, load=load, weather=pretrain_weather,
        first_feature_date=args.train_start, target_date=args.pretrain_end,
    )
    live_weather, sources = runner._online_lstm_weather(config, args.train_start, args.end)
    X, Y = runner._online_lstm_matrices(
        config, prices=prices, exaa=exaa, load=load, weather=live_weather,
        first_feature_date=args.train_start, target_date=args.end,
    )
    days = [args.start + timedelta(days=k) for k in range((args.end - args.start).days + 1)]
    days = [d for d in days if sources.get(d) == "ICON-D2"]

    # ---- online LSTM: one pretraining per (config, seed) -------------------
    results: dict[str, dict[date, np.ndarray]] = {}
    runs: dict[str, dict] = {}
    for config_name in config_names:
        for seed in seeds:
            run_config = replace(
                online_config, pretrain=replace(pretrain_bases[config_name], random_seed=seed)
            )
            tag = f"{config_name}_s{seed}"
            online_name, frozen_name = f"online_{tag}", f"online_pretrained_only_{tag}"
            results[online_name], results[frozen_name] = {}, {}
            with tempfile.TemporaryDirectory(prefix="lstm_online_backtest_") as tmp:
                store = OnlineLSTMStore(Path(tmp))
                started = time.time()
                pretrained = pretrain_state(X_pre, Y_pre, run_config)
                pretrain_seconds = time.time() - started
                store.save_pretrained(pretrained)
                print(f"[{tag}] pretrained on {pretrained.metadata['pretrain_days']} days "
                      f"in {pretrain_seconds:.0f}s", flush=True)
                started = time.time()
                for day in days:
                    state, _ = advance_state(store, X, Y, day, run_config)
                    results[online_name][day] = (
                        forecast_state(state, X, Y, day, run_config)[0]["y_pred"].to_numpy()
                    )
                    results[frozen_name][day] = (
                        forecast_state(pretrained, X, Y, day, run_config)[0]["y_pred"].to_numpy()
                    )
            runs[tag] = {
                "config": config_name,
                "seed": seed,
                "pretrain_seconds": round(pretrain_seconds, 1),
                "finetune_and_forecast_seconds": round(time.time() - started, 1),
            }

    # ---- rolling EXAA LSTM ---------------------------------------------
    if not args.skip_rolling:
        started = time.time()
        rolling = rolling_lstm_exaa_enriched_forecast(
            X, Y, [pd.Timestamp(d, tz=config.timezone) for d in days], LSTMExaaEnrichedConfig()
        )
        # Output rows are sorted by delivery day, 96 MTUs each.
        values = rolling["y_pred"].to_numpy().reshape(len(days), 96)
        results["rolling_lstm_exaa"] = dict(zip(sorted(days), values))
        print(f"Rolling EXAA LSTM refitted for {len(days)} days in {time.time() - started:.0f}s", flush=True)

    # ---- LEAR ------------------------------------------------------------
    lear_path = Path(args.lear_forecast)
    lear_path = lear_path if lear_path.is_absolute() else REPO_ROOT / lear_path
    lear = _lear_frame(lear_path)
    results["lear_d56_c5"] = {
        d: lear.loc[d.isoformat()].to_numpy(dtype=float) for d in days if d.isoformat() in lear.index
    }

    # ---- evaluation on common days ----------------------------------------
    actual = {d: Y.loc[pd.Timestamp(d, tz=config.timezone)].to_numpy(dtype=float) for d in days}
    common = [d for d in days if np.isfinite(actual[d]).all() and all(d in r for r in results.values())]
    if not common:
        raise SystemExit("No common evaluation days with forecasts for every model.")
    summary = {
        name: _metrics(np.concatenate([forecasts[d] - actual[d] for d in common]))
        for name, forecasts in results.items()
    }
    daily = pd.DataFrame(
        {name: [float(np.mean(np.abs(forecasts[d] - actual[d]))) for d in common]
         for name, forecasts in results.items()},
        index=pd.Index([d.isoformat() for d in common], name="delivery_date"),
    )

    # Seed-averaged metrics per optimiser config (mean and spread over seeds).
    by_config: dict[str, dict] = {}
    for config_name in config_names:
        for prefix in ("online", "online_pretrained_only"):
            maes = [summary[f"{prefix}_{config_name}_s{seed}"]["MAE"] for seed in seeds]
            rmses = [summary[f"{prefix}_{config_name}_s{seed}"]["RMSE"] for seed in seeds]
            by_config[f"{prefix}_{config_name}"] = {
                "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
                "RMSE_mean": float(np.mean(rmses)), "seeds": seeds,
            }

    stem = Path(args.output) if args.output else (
        REPO_ROOT / "results" / "operational" / "mlp"
        / f"backtest_lstm_era5_online_{args.start}_to_{args.end}"
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(stem.with_suffix(".csv"))
    payload = {
        "pretrain_weather_root": str(weather_root),
        "train_start": args.train_start.isoformat(),
        "pretrain_end": args.pretrain_end.isoformat(),
        "evaluation_days": [d.isoformat() for d in common],
        "online_config": online_config.to_dict(),
        "pretrain_configs": {name: asdict(pretrain_bases[name]) for name in config_names},
        "runs": runs,
        "metrics_eur_per_mwh": summary,
        "metrics_by_config_seed_averaged": by_config,
        "models": {
            "online_<config>_s<seed>": "pretrained once, then fine-tuned daily on the last week",
            "online_pretrained_only_<config>_s<seed>": "same pretrained model without any update",
            "rolling_lstm_exaa": "existing LSTM_exaa_enriched, refitted daily (expanding window)",
            "lear_d56_c5": f"LEAR D56 C5 fundamental from {lear_path}",
        },
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(common)} gemeinsame Tage ({common[0]} .. {common[-1]})")
    for name, values in sorted(summary.items(), key=lambda item: item[1]["MAE"]):
        print(f"  {name:40s} MAE {values['MAE']:7.2f}  RMSE {values['RMSE']:7.2f}  Bias {values['Bias']:7.2f}")
    print("\nSeed-averaged:")
    for name, values in sorted(by_config.items(), key=lambda item: item[1]["MAE_mean"]):
        print(f"  {name:40s} MAE {values['MAE_mean']:7.2f} +- {values['MAE_std']:.2f}  "
              f"RMSE {values['RMSE_mean']:7.2f}")
    print(f"-> {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
