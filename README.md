# An Open, Operational Probabilistic Forecasting Pipeline for Day-Ahead Electricity Prices


This repository implements an open, automated pipeline for 15-minute point
and probabilistic day-ahead electricity-price forecasts for the EPEX DE-LU
bidding zone. The daily workflow maintains market and ICON-D2 inputs, runs the
selected forecasting models, and can submit forecasts to Energy Arena.

All automated computation is implemented in ordinary Python modules. The
notebooks are optional interfaces for interactive inspection and do not contain
an alternative pipeline implementation.

## Repository structure

```text
DA_Price_Forecasting_Pipeline_DE_LU/
├── run_pipeline.py                 # daily operational entry point
├── prepare_dwd_data.py             # daily ICON-D2 preparation
├── pipeline/
│   ├── operational/                # downloads, validation, submission, locking
│   ├── lear/
│   │   └── lear_model.py           # LEAR and feature functions
│   ├── mlp/
│   │   └── ...                     # operational neural point models
│   ├── sqra/
│   │   └── sqra_model.py           # probabilistic post-processing
│   ├── delivery_index.py           # canonical delivery-day/MTU indexing
│   └── dwd_history.py              # appendable ICON-D2 history format
├── data/
│   ├── clustering/                 # spatial ICON-D2 cluster definitions
│   ├── icon/                       # consolidated operational weather histories
│   └── shapefile/                  # geographic boundary assets
├── experiment_config.py            # shared pipeline configuration
├── experiment_manifest.py          # canonical model configurations
├── market_cache.py                 # persistent market-data caches
├── .env.example                    # configuration template
└── requirements.txt                # pipeline dependencies
```

## Setup

### Python environment

Python 3.9 is recommended. In PowerShell, from the repository root:

```powershell
py -3.9 -m venv forecast
.\forecast\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

The `requirements.txt` installs the operational pipeline dependencies.
Registering a Jupyter kernel is optional:

```powershell
python -m ipykernel install --user --name forecast --display-name "forecast (Python 3.9)"
```

### Environment configuratio

Create the local configuration file:

```powershell
Copy-Item .env.example .env
```

At minimum, configure:

```dotenv
ENTSOE_API_KEY=...
ENERGY_ARENA_API_KEY=...

ICON_DATA_ROOT=data/icon
DWD_OPERATIONAL_RAW_ROOT=data/dwd_raw
OPERATIONAL_MARKET_DATA_ROOT=data/market
OPERATIONAL_RESULTS_ROOT=results/operational
OPERATIONAL_OUTPUT_ROOT=output/operational
```

`ENTSOE_API_KEY` is required for DE-LU prices, EXAA prices, and load forecasts.
`ENERGY_ARENA_API_KEY` is required only when forecasts are submitted. Input,
cache, result, and output paths can be absolute or relative to the repository
root. Energy Arena submissions use a 60-second timeout and three attempts by
default. These can be changed with `ENERGY_ARENA_SUBMISSION_TIMEOUT_SECONDS`,
`ENERGY_ARENA_SUBMISSION_MAX_ATTEMPTS`, and
`ENERGY_ARENA_SUBMISSION_RETRY_SECONDS`.

Before the first operational run, inspect the resolved configuration and live
Energy Arena target:

```powershell
python run_pipeline.py --check-setup
python run_pipeline.py --dry-run
```

## Daily operational pipeline

### Operational data

The operational pipeline uses only the data required for the next daily
forecast and its rolling calibration:

| Input | Purpose | Acquisition |
|---|---|---|
| EPEX DE-LU day-ahead prices | LEAR targets and lagged price features | ENTSO-E API |
| ENTSO-E DE-LU load forecast | Fundamental-model feature | ENTSO-E API |
| EXAA day-ahead prices | EXAA-Enriched and EXAA-Only features | ENTSO-E API, sequence 2 |
| DWD ICON-D2 forecasts | Wind and solar features | DWD Open Data |
| ICON-D2 cluster definitions | Spatial aggregation for `C=1,5,25` | Versioned under `data/clustering` |

The market caches are updated automatically whenever coverage is missing:

```text
OPERATIONAL_MARKET_DATA_ROOT/
├── entsoe/
│   ├── prices_da.csv
│   └── load_forecast.csv
└── exaa/
    └── prices_exaa.csv
```

ENTSO-E prices are always maintained. Load forecasts are required by
weather-based models, while EXAA prices are loaded only when the selected model
uses them or `DOWNLOAD_EXAA=true`. Optional variables
`OPERATIONAL_ENTSOE_PRICE_CACHE_DIR`,
`OPERATIONAL_ENTSOE_LOAD_CACHE_DIR`, and `OPERATIONAL_EXAA_CACHE_DIR` can place
the three caches elsewhere. When an ENTSO-E or EXAA request fails, the pipeline
waits `MARKET_DATA_RETRY_SECONDS` (300 seconds by default) before its next
attempt. EXAA-Only makes eight total EXAA attempts. The Fundamental pipeline
makes four total load-forecast attempts and, if the fourth still fails,
continues with a consistent feature set that omits the load block.

### Maintaining the ICON-D2 calibration history

The DWD operational archive does not provide the complete sequence of past
ICON-D2 forecasts needed to reconstruct a rolling model calibration later.
Consequently, the relevant forecast run must be collected every day. The
repository downloads the preceding 06 UTC ICON-D2 run, verifies it, aggregates
it to the spatial resolutions `C=1,5,25`, and appends the new delivery day to
the persistent weather histories.

The consolidated cluster histories under the default `data/icon` path are
updated and published to this GitHub repository every day. A fresh clone
therefore contains the collected historical ICON-D2 cluster information up to
the repository's latest update. Users do not need to obtain all past raw
forecasts themselves before calibrating the model; they only need to continue
the daily download and preparation process for new delivery days. The large raw
GRIB/ZIP downloads are not versioned—only the compact, preprocessed Parquet
histories required by the forecasting models are maintained in the repository.
For operational robustness, a complete target-day ICON-D2 forecast remains
mandatory, while unavailable historical ICON-D2 delivery days are skipped in
LEAR and SQRA calibration and recorded in each run's metadata. The equivalent
policy applies to EXAA: target-day EXAA prices remain mandatory, but, when
automatic EXAA downloading is enabled, an unavailable historical EXAA delivery
day is retried on every pipeline run and skipped in calibration until the cache
can be repaired.

This produces four appendable Parquet histories for every spatial resolution:

```text
ICON_DATA_ROOT/
├── c1/
│   ├── u10.parquet
│   ├── v10.parquet
│   ├── ASWDIR_S.parquet
│   ├── ASWDIFD_S.parquet
│   └── dwd_history.json
├── c5/
│   ├── u10.parquet
│   ├── v10.parquet
│   ├── ASWDIR_S.parquet
│   ├── ASWDIFD_S.parquet
│   └── dwd_history.json
└── c25/
    ├── u10.parquet
    ├── v10.parquet
    ├── ASWDIR_S.parquet
    ├── ASWDIFD_S.parquet
    └── dwd_history.json
```

`u10` and `v10` contain the wind components. `ASWDIR_S` and `ASWDIFD_S`
contain direct and diffuse short-wave radiation. Every history records the
delivery date, issue date, run hour, valid timestamp, and spatial cluster
values. `dwd_history.json` records and validates the history coverage.

Prepare tomorrow's weather data independently before model execution:

```powershell
python .\prepare_dwd_data.py
```

The preparation script performs the following steps:

1. download or locate only the four required ICON-D2 variables for the 06 UTC
   run;
2. validate the ZIP, BZip2, and GRIB content;
3. retry failed or incomplete downloads;
4. preprocess all required fields for `C=1,5,25`;
5. deduplicate repeated valid timestamps;
6. append or replace the target delivery day in each Parquet history;
7. verify all twelve updated histories; and
8. delete the consumed raw data only after successful validation.

The default Fundamental/SQRA system needs approximately 116 preceding delivery
days for its first complete run: 56 days for LEAR plus 60 days for SQRA. After
that bootstrap, one new weather day is appended on every operational day.

### Running the complete daily pipeline

The default configuration follows the paper's selected operational setup:

- SQRA inputs: Fundamental LEAR forecasts with ICON-D2, `D_LEAR=56`, and
  `C={1,5,25}`;
- probabilistic forecast: one pooled SQRA fit per requested quantile using the
  preceding 60 delivery days and all valid MTUs;
- point submission: the SQRA median forecast (`q=0.5`); and
- probabilistic submission: all quantiles requested by Energy Arena.

The `C=5` Fundamental LEAR forecast is therefore no longer submitted directly.
It remains one of the three point forecasts used to estimate SQRA.

Run the complete workflow and submit both forecasts:

```powershell
python run_pipeline.py
```

The runner then:

1. resolves and validates the current DE-LU Energy Arena point and quantile
   challenges;
2. verifies that the consolidated ICON-D2 histories contain the target day,
   invoking the DWD preparation step if necessary;
3. updates the ENTSO-E price/load caches and, when required, the EXAA cache;
4. fits any missing rolling LEAR point forecasts and updates their histories;
5. fits the target-day SQRA models and generates non-crossing quantiles;
6. uses SQRA `q=0.5` for the point payload and all requested SQRA quantiles for
   the probabilistic payload;
7. creates DST-aware Energy Arena payloads with 92, 96, or 100 physical MTUs;
   and
8. submits the point and quantile payloads with `ENERGY_ARENA_API_KEY`.

Point and quantile submissions are handled independently. Transient network
timeouts, connection failures, rate limits, and server errors are retried with
backoff. A final failure for one challenge does not prevent the other challenge
from being attempted. Each challenge's stable `latest.json` receipt records the
most recent confirmed success or final failed attempt; the command exits with a
nonzero status if either challenge remains unsuccessful so an external scheduler
can retry the run.

If the target-day 06 UTC ICON-D2 data remain unavailable after the configured
DWD download and verification attempts, the default Fundamental pipeline uses
operational persistence fallbacks. The point forecast first uses the realized
EPEX price at the same MTU on delivery day d-1; if that complete day is
unavailable, it uses d-7. The probabilistic fallback uses the same lag and adds
pooled empirical residual quantiles calculated from the preceding 60 calendar
days. Residual offsets are median-centered, so q=0.5 is exactly the submitted
point fallback. At least 14 complete calibration days are required; otherwise
only the point fallback is submitted. If neither d-1 nor d-7 is complete, the
pipeline fails without submitting a fallback.

Fallback values are written only to the current payloads and submission
receipts. They are never added to Fundamental LEAR or SQRA forecasts, metrics,
runtimes, or calibration histories and therefore cannot affect later model
calibration.

Generate and validate everything without submitting:

```powershell
python run_pipeline.py --no-submit
```

After the live cutoff, Energy Arena may already advertise the day after
tomorrow. For a local test only, shift that target back by one delivery day:

```powershell
python run_pipeline.py --no-submit --d-1
```

`--d-1` is intentionally restricted to `--no-submit`.

### Operational model variants

Select the model on the command line or through `.env`:

```powershell
# Default ICON-D2 Fundamental model
python run_pipeline.py --fundamental

# ICON-D2 Fundamental model enriched with EXAA prices
python run_pipeline.py --exaa-enriched

# EXAA-only model without weather or load features
python run_pipeline.py --exaa-only
```

The EXAA-Only SQRA forecast combines the `D_LEAR={56,112,364}` EXAA-Only point
models and submits its `q=0.5` median to the point challenge. The former
`D_LEAR=364` point forecast remains an SQRA member but is no longer submitted
directly. Weather-based SQRA likewise combines all three cluster resolutions.
`--cluster 1|5|25` and `POINT_WEATHER_CLUSTERS` retain the reference LEAR run
reported in logs and metadata; they do not change the submitted SQRA median.

For a direct one-layer neural-network point forecast with C=5 weather and all
96 target-day EXAA prices, use:

```powershell
python run_pipeline.py --nn-exaa-enriched
```

This mode prepares only the C=5 DWD history and submits the NN point forecast
directly. The quantile payload is centered on that NN forecast and uses pooled
empirical d-1 persistence residuals from the preceding 60 days; it is not an
NN- or SQRA-trained quantile model. Use `--no-submit` to create the forecasts
and payloads locally without sending them to Energy Arena.

To run the recurrent LSTM variant over the 96 quarter-hour MTUs of the target
delivery day, use:

```powershell
python run_pipeline.py --lstm-exaa-enriched
```

Use `--lstm-exaa-enriched-start-date YYYY-MM-DD` to set the first training
feature date explicitly; the pipeline still loads the preceding seven days for
price-lag features and checks that all inputs cover the requested date.

The LSTM uses the same C=5 weather, EPEX price lags, load forecast, and target-
day EXAA features as the EXAA-enriched NN. It also uses the mean EPEX price
from the seven preceding complete delivery days in both the LSTM inputs and a
linear skip connection to every MTU output. It submits a point forecast and
pooled empirical D-1 persistence-residual quantiles. Add `--no-submit` to
generate and validate the forecasts locally only. The latest point forecast,
quantile forecast, and run metadata overwrite the fixed files
`results/operational/mlp/forecast_lstm_exaa_enriched.csv`,
`results/operational/mlp/forecast_lstm_exaa_enriched_quantiles.csv`, and
`results/operational/mlp/forecast_lstm_exaa_enriched.json`.

To train the same recurrent architecture without any EXAA input features, use:

```powershell
python run_pipeline.py --lstm-fundamental --no-submit
```

This mode uses C=5 weather, EPEX price lags, load forecast, temporal features,
and the mean EPEX price from the seven preceding complete delivery days. The
weekly mean is also connected to all 96 outputs. It does not request or use
EXAA prices. `--no-submit` keeps the generated point and quantile forecasts
local.

For an EXAA-enriched LSTM without weather inputs, but with ENTSO-E load,
use:

```powershell
python run_pipeline.py --lstm-exaa-enriched-price-load
```

This model uses only the EPEX D-1, D-2, and D-7 price vectors and the target-
day EXAA prices plus the target-day ENTSO-E load forecast. Its expanding
training history starts on 2025-10-01; it fetches the preceding seven days of
EPEX data for the lag features. Historical EPEX and load requests are split
into 90-day intervals and returned on the 15-minute grid. EXAA history is
requested through the target day. Historical training days without an
available load forecast are omitted rather than imputed. Use `--no-submit` to
save forecasts without sending them to Energy Arena. The former
`--lstm-exaa-enriched-price-only` option remains an alias.

The NN uses an expanding training window: its feature history starts one week
after the latest first-recorded date among the C=5 weather, EPEX, EXAA, and load
data. It starts fitting once seven complete training days are available and
adds each subsequent complete day without dropping earlier samples. Incomplete
days are excluded.

The EXAA NN sends each target-day EXAA price both through the hidden layer and
through direct linear skip connections to the same MTU output and the two
adjacent MTUs on either side (cyclically across midnight). The existing EPEX
skip connections remain unchanged.

The equivalent `.env` settings are:

```dotenv
POINT_FORECAST_VARIANT=fundamental
POINT_WEATHER_CLUSTERS=5
SQRA_FORECAST_VARIANT=auto
SQRA_TRAIN_DAYS=60
SQRA_MTU_SPECIFIC=false
```

For a faster C=5-only Fundamental SQRA run, select
`SQRA_FORECAST_VARIANT=fundamental_c5`. This fits only the C=5 Fundamental
LEAR member and derives SQRA quantiles from that single member; the default
`auto` setting continues to use all three cluster resolutions.
A practical local-time schedule is:

- approximately 10:00: `prepare_dwd_data.py`;
- approximately 10:30: Fundamental `run_pipeline.py`;
- approximately 11:15: `run_pipeline.py --exaa-only`.

## Tests

Run the synthetic parity, schema, validation, and orchestration tests from the
activated environment:

```powershell
python -m unittest discover -s tests -v
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
