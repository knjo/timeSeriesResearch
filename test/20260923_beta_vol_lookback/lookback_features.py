"""Prior-only TXF beta and residual vol at 60, 20 and 5 day lookbacks.

Research-only sensitivity variant of `src.features.market_beta`. The production
estimator `estimate_txf_beta` and the intraday vol clock are reused unchanged;
only `MarketBetaConfig` differs per window. Nothing here writes preMarket,
timeSeries `data/` or `data_index.parquet`.

Date contract matches preMarket: the row for decision date T uses the last N
`*_marketData.parquet` files up to and including the source date S, the
trading day before T.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from src.dataloader.txfDataLoader import TxfDataLoader
from src.features.market_beta import (
    TXF_REMAIN_SECONDS_COLUMN,
    TXF_SESSION_SECONDS,
    MarketBetaConfig,
    estimate_txf_beta,
    residual_vol_time_scale,
)
from src.pipeline_storage import load_pipeline_storage


STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[4]
ARTIFACT_DIR = STUDY_DIR / "artifacts" / "v1"
TABLE_PATH = ARTIFACT_DIR / "lookback_beta_vol.parquet"
EVENT_DATA_DIR = PROJECT_ROOT / "src" / "research" / "timeSeries" / "data"

# 60 is the production config. Shorter windows keep the same full-window
# shrink weight (n / (n + prior) = 0.75) and roughly the same 1/3 coverage
# floor, so the comparison isolates window length instead of prior strength.
# The 5-day floor is raised to 4 so the residual std has at least 3 dof.
LOOKBACK_CONFIGS: dict[int, MarketBetaConfig] = {
    60: MarketBetaConfig(),
    20: MarketBetaConfig(lookback_days=20, min_observations=7, prior_strength=20 / 3),
    5: MarketBetaConfig(lookback_days=5, min_observations=4, prior_strength=5 / 3),
}


def beta_column(window: int) -> str:
    return f"txf_beta_{window}"


def full_vol_column(window: int) -> str:
    return f"txf_residual_vol_0900_1330_{window}"


def event_vol_column(window: int) -> str:
    return f"txf_residual_vol_to_1330_{window}"


def storage_dirs() -> tuple[Path, Path]:
    storage = load_pipeline_storage(
        project_root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / "config" / "pipeline.yaml",
    )
    return Path(storage.market_dir), Path(storage.txf_tick_dir)


def market_dates(market_dir: Path) -> list[str]:
    return sorted(
        path.name.split("_", 1)[0] for path in market_dir.glob("*_marketData.parquet")
    )


def event_dates(event_dir: Path = EVENT_DATA_DIR) -> list[str]:
    return sorted(path.stem for path in event_dir.glob("[0-9]" * 8 + ".parquet"))


def event_codes_by_date(
    dates: list[str], event_dir: Path = EVENT_DATA_DIR
) -> dict[str, list[str]]:
    return {
        date: pl.read_parquet(event_dir / f"{date}.parquet", columns=["QuoteCode"])[
            "QuoteCode"
        ]
        .cast(pl.Utf8)
        .unique()
        .to_list()
        for date in dates
    }


def load_return_history(
    dates: list[str],
    codes: list[str],
    market_dir: Path,
    txf_loader: TxfDataLoader,
) -> pl.DataFrame:
    """Stock and TXF open-to-close returns, filtered as in `add_txf_beta_features`."""
    parts: list[pl.DataFrame] = []
    for date in dates:
        factor = txf_loader.daily_open_to_close_return(date)
        if factor is None:
            continue
        market_path = market_dir / f"{date}_marketData.parquet"
        schema = set(pl.scan_parquet(market_path).collect_schema().names())
        if not {"quote_code", "open_price", "close_price"} <= schema:
            continue
        tradable = (
            (pl.col("high_price") != pl.col("low_price"))
            if {"high_price", "low_price"} <= schema
            else pl.lit(True)
        )
        parts.append(
            pl.scan_parquet(market_path)
            .filter(pl.col("quote_code").cast(pl.Utf8).is_in(codes))
            .select(
                pl.lit(date).alias("Date"),
                pl.col("quote_code").cast(pl.Utf8).alias("QuoteCode"),
                pl.when(
                    (pl.col("open_price") > 0.0)
                    & (pl.col("close_price") > 0.0)
                    & tradable
                )
                .then(pl.col("close_price") / pl.col("open_price") - 1.0)
                .otherwise(None)
                .cast(pl.Float64)
                .alias("stock_return"),
                pl.lit(factor.return_value).alias("txf_return"),
            )
            .unique("QuoteCode", keep="last")
            .collect()
        )
    return pl.concat(parts, how="vertical")


def _window_features(
    history: pl.DataFrame,
    window_dates: list[str],
    source_date: str,
    window: int,
) -> pl.DataFrame:
    config = LOOKBACK_CONFIGS[window]
    in_window = history.filter(pl.col("Date").is_in(window_dates))
    observations = (
        in_window.filter(
            pl.col("stock_return").is_finite() & pl.col("txf_return").is_finite()
        )
        .group_by("QuoteCode")
        .agg(pl.len().cast(pl.Int32).alias(f"txf_obs_{window}"))
    )
    estimates = estimate_txf_beta(in_window, source_date, config)
    if "txf_corr_60d" not in estimates.columns:
        # The empty-history schema omits corr.
        estimates = estimates.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("txf_corr_60d")
        )
    return (
        estimates.select(
            "QuoteCode",
            "txf_beta_raw_60d",
            "txf_beta_60d",
            "txf_corr_60d",
            "txf_residual_vol_0900_1330_60d",
        )
        .rename(
            {
                "txf_beta_raw_60d": f"txf_beta_raw_{window}",
                "txf_beta_60d": beta_column(window),
                "txf_corr_60d": f"txf_corr_{window}",
                "txf_residual_vol_0900_1330_60d": full_vol_column(window),
            }
        )
        .join(observations, on="QuoteCode", how="left")
    )


def build_lookback_table(
    decision_dates: list[str] | None = None,
    *,
    event_dir: Path = EVENT_DATA_DIR,
    windows: tuple[int, ...] = (60, 20, 5),
    history: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per (decision Date, QuoteCode) present in timeSeries `data/`.

    `window_full_{N}` is False while fewer than N marketData files precede the
    source date, so early-2023 rows can be excluded from comparisons.
    """
    market_dir, txf_dir = storage_dirs()
    calendar = market_dates(market_dir)
    decision_dates = decision_dates or event_dates(event_dir)

    codes_by_date = event_codes_by_date(decision_dates, event_dir)
    if history is None:
        all_codes = sorted({c for codes in codes_by_date.values() for c in codes})
        history = load_return_history(
            calendar, all_codes, market_dir, TxfDataLoader(txf_dir)
        )

    frames: list[pl.DataFrame] = []
    for date in decision_dates:
        prior = [day for day in calendar if day < date]
        if not prior:
            continue
        source_date = prior[-1]
        codes = codes_by_date[date]
        day_history = history.filter(pl.col("QuoteCode").is_in(codes))
        out = pl.DataFrame({"QuoteCode": codes}).with_columns(
            pl.lit(date).alias("Date"),
            pl.lit(source_date).alias("source_date"),
        )
        for window in windows:
            window_dates = prior[-window:]
            out = out.join(
                _window_features(day_history, window_dates, source_date, window),
                on="QuoteCode",
                how="left",
            ).with_columns(
                # Same fallback as production: missing beta -> prior beta.
                pl.col(beta_column(window)).fill_null(
                    LOOKBACK_CONFIGS[window].prior_beta
                ),
                pl.lit(len(window_dates) == window).alias(f"window_full_{window}"),
            )
        frames.append(out)
    return pl.concat(frames, how="vertical").select(
        "Date", "QuoteCode", "source_date", pl.exclude("Date", "QuoteCode", "source_date")
    )


def _date_key(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.strftime("%Y%m%d")
    return values.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)


def attach_lookback_variants(
    frame: pd.DataFrame | pl.DataFrame,
    table: pl.DataFrame | None = None,
    *,
    windows: tuple[int, ...] = (60, 20, 5),
    table_path: Path = TABLE_PATH,
) -> pd.DataFrame | pl.DataFrame:
    """Join `txf_beta_{N}` and scaled `txf_residual_vol_to_1330_{N}` onto events.

    Needs `Date`, `QuoteCode` and `RemainSeconds_1330` (written by the
    timeSeries loader). Returns the same frame type it was given, row order
    kept. The vol clock is the production `residual_vol_time_scale`.
    """
    if table is None:
        table = pl.read_parquet(table_path)
    keep = ["Date", "QuoteCode"]
    for window in windows:
        keep += [beta_column(window), full_vol_column(window)]
    table = table.select(keep).rename({"Date": "_lookback_date"})

    is_pandas = isinstance(frame, pd.DataFrame)
    if is_pandas:
        events = pl.from_pandas(
            pd.DataFrame(
                {
                    "_lookback_date": _date_key(frame["Date"]).to_numpy(),
                    "QuoteCode": frame["QuoteCode"].astype(str).to_numpy(),
                    TXF_REMAIN_SECONDS_COLUMN: frame[TXF_REMAIN_SECONDS_COLUMN]
                    .astype(float)
                    .to_numpy(),
                }
            )
        )
    else:
        events = frame.select(
            pl.col("Date").cast(pl.Utf8).str.replace_all("-", "").str.slice(0, 8)
            .alias("_lookback_date"),
            pl.col("QuoteCode").cast(pl.Utf8),
            pl.col(TXF_REMAIN_SECONDS_COLUMN).cast(pl.Float64),
        )

    scale = residual_vol_time_scale(
        pl.col(TXF_REMAIN_SECONDS_COLUMN).clip(0.0, TXF_SESSION_SECONDS)
        / TXF_SESSION_SECONDS
    )
    added = events.join(
        table, on=["_lookback_date", "QuoteCode"], how="left", validate="m:1"
    ).select(
        *[pl.col(beta_column(window)) for window in windows],
        *[
            pl.when(
                pl.col(full_vol_column(window)).is_finite()
                & (pl.col(full_vol_column(window)) > 0.0)
            )
            .then(pl.col(full_vol_column(window)) * scale)
            .otherwise(None)
            .alias(event_vol_column(window))
            for window in windows
        ],
    )
    if is_pandas:
        out = frame.copy()
        for column in added.columns:
            out[column] = added[column].to_numpy()
        return out
    return frame.drop([c for c in added.columns if c in frame.columns]).hstack(added)
