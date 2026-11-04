"""Build the 60/20/5 lookback beta/vol table and measure window sensitivity.

Three questions, each on the common sample where all three windows have a vol:

1. Agreement: how differently do the windows rank stocks on beta and vol?
2. Next-session fit (out of sample): with day-T open-to-close returns, which
   window's beta removes more market variance and which vol has the lowest
   QLIKE / best-calibrated z?
3. Event label: how much does the timeSeries residual-z label
   (Close vs event mid, minus beta x txf_to_1330_return, over vol_to_1330)
   move when the window changes?

Outputs go to `artifacts/v1/` only; nothing production-facing is touched.

    uv run python src/research/timeSeries/test/20260923_beta_vol_lookback/run_lookback_sensitivity.py
    uv run python .../run_lookback_sensitivity.py --rebuild --start 20250101
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lookback_features import (  # noqa: E402
    ARTIFACT_DIR,
    EVENT_DATA_DIR,
    TABLE_PATH,
    attach_lookback_variants,
    beta_column,
    build_lookback_table,
    event_codes_by_date,
    event_dates,
    event_vol_column,
    full_vol_column,
    load_return_history,
    market_dates,
    storage_dirs,
)
from src.dataloader.txfDataLoader import TxfDataLoader  # noqa: E402

WINDOWS = (60, 20, 5)
SHORT = (20, 5)
ROBUST = 0.6745
EVENT_COLUMNS = (
    "Date",
    "QuoteCode",
    "RemainSeconds_1330",
    "BidPrice1",
    "AskPrice1",
    "Close",
    "txf_to_1330_return",
    "txf_beta_60d",
    "txf_residual_vol_0900_1330_60d",
)
SESSION_BUCKETS = ("09-10", "10-11", "11-12", "12-13", "13-1330")


def _all_vols_present(frame: pl.DataFrame, vol_of) -> pl.DataFrame:
    return frame.filter(
        pl.all_horizontal(pl.col(vol_of(w)).is_not_null() for w in WINDOWS)
    )


def agreement_by_date(table: pl.DataFrame) -> pl.DataFrame:
    common = _all_vols_present(table.filter(pl.col("window_full_60")), full_vol_column)
    coverage = (
        table.filter(pl.col("window_full_60"))
        .group_by("Date")
        .agg(
            pl.len().alias("n_codes"),
            *[
                pl.col(full_vol_column(w)).is_not_null().mean().alias(f"vol_cov_{w}")
                for w in WINDOWS
            ],
        )
    )
    pairs = [(60, 20), (60, 5), (20, 5)]
    stats = common.group_by("Date").agg(
        pl.len().alias("n_common"),
        *[
            pl.corr(beta_column(a), beta_column(b), method="spearman").alias(
                f"beta_rho_{a}_{b}"
            )
            for a, b in pairs
        ],
        *[
            pl.corr(full_vol_column(a), full_vol_column(b), method="spearman").alias(
                f"vol_rho_{a}_{b}"
            )
            for a, b in pairs
        ],
        *[
            (pl.col(beta_column(w)) - pl.col(beta_column(60)))
            .abs()
            .median()
            .alias(f"beta_mad_{w}")
            for w in SHORT
        ],
        *[
            (pl.col(full_vol_column(w)) / pl.col(full_vol_column(60)))
            .log()
            .median()
            .alias(f"vol_logratio_med_{w}")
            for w in SHORT
        ],
        *[
            (
                (pl.col(full_vol_column(w)) / pl.col(full_vol_column(60)))
                .log()
                .quantile(0.75)
                - (pl.col(full_vol_column(w)) / pl.col(full_vol_column(60)))
                .log()
                .quantile(0.25)
            ).alias(f"vol_logratio_iqr_{w}")
            for w in SHORT
        ],
    )
    return coverage.join(stats, on="Date", how="left").sort("Date")


def nextday_by_date(table: pl.DataFrame, realized: pl.DataFrame) -> pl.DataFrame:
    """Day-T open-to-close fit of features known before T's open."""
    frame = (
        _all_vols_present(table.filter(pl.col("window_full_60")), full_vol_column)
        .join(realized, on=["Date", "QuoteCode"], how="inner")
        .filter(pl.col("stock_return").is_finite() & pl.col("txf_return").is_finite())
    )
    r, f = pl.col("stock_return"), pl.col("txf_return")
    exprs = [
        (r * r).mean().alias("msq_beta0"),
        ((r - f) ** 2).mean().alias("msq_beta1"),
    ]
    for w in WINDOWS:
        e = r - pl.col(beta_column(w)) * f
        z = e / pl.col(full_vol_column(w))
        exprs += [
            (e * e).mean().alias(f"msq_{w}"),
            (z * z).mean().sqrt().alias(f"rms_z_{w}"),
            (z.abs().median() / ROBUST).alias(f"robust_z_{w}"),
            ((pl.col(full_vol_column(w)) ** 2).log() + z * z).mean().alias(f"qlike_{w}"),
        ]
    return frame.group_by("Date").agg(pl.len().alias("n"), *exprs).sort("Date")


def _event_frame(date: str, day_table: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    raw = pl.read_parquet(EVENT_DATA_DIR / f"{date}.parquet", columns=list(EVENT_COLUMNS))

    # Parity: recomputed 60 vs the preMarket columns already in data/.
    code_day = raw.unique("QuoteCode").join(
        day_table.select(
            "QuoteCode", beta_column(60), full_vol_column(60)
        ),
        on="QuoteCode",
        how="left",
    )
    vol_a, vol_b = pl.col("txf_residual_vol_0900_1330_60d"), pl.col(full_vol_column(60))
    parity = code_day.select(
        pl.len().alias("code_days"),
        ((pl.col("txf_beta_60d") - pl.col(beta_column(60))).abs() > 1e-12)
        .fill_null(True)
        .sum()
        .alias("beta_mismatch"),
        (
            (vol_a.is_null() != vol_b.is_null())
            | ((vol_a - vol_b).abs() > 1e-12).fill_null(False)
        )
        .sum()
        .alias("vol_mismatch"),
    ).row(0, named=True)
    parity["Date"] = date

    events = attach_lookback_variants(raw, day_table).filter(
        (pl.col("BidPrice1") > 0.0)
        & (pl.col("AskPrice1") >= pl.col("BidPrice1"))
        & (pl.col("Close") > 0.0)
        & pl.col("txf_to_1330_return").is_finite()
        & (pl.col("RemainSeconds_1330") >= 60.0)
    )
    events = _all_vols_present(events, event_vol_column)
    mid = (pl.col("BidPrice1") + pl.col("AskPrice1")) / 2.0
    events = events.with_columns(
        *[
            (
                pl.col("Close") / mid
                - 1.0
                - pl.col(beta_column(w)) * pl.col("txf_to_1330_return")
            ).alias(f"e_{w}")
            for w in WINDOWS
        ],
        (
            ((16_200.0 - pl.col("RemainSeconds_1330")) // 3_600.0)
            .cast(pl.Int32)
            .clip(0, len(SESSION_BUCKETS) - 1)
        ).alias("bucket"),
    ).with_columns(
        *[(pl.col(f"e_{w}") / pl.col(event_vol_column(w))).alias(f"z_{w}") for w in WINDOWS]
    )
    return events, parity


def event_metrics(events: pl.DataFrame, date: str) -> tuple[dict, pl.DataFrame]:
    row = events.select(
        pl.len().alias("n_events"),
        *[
            pl.corr("z_60", f"z_{w}", method="spearman").alias(f"z_rho_60_{w}")
            for w in SHORT
        ],
        *[
            (pl.col("e_60").sign() != pl.col(f"e_{w}").sign())
            .mean()
            .alias(f"e_signflip_{w}")
            for w in SHORT
        ],
        *[
            (pl.col(f"z_{w}").abs() / pl.col("z_60").abs())
            .filter(pl.col("z_60").abs() > 1e-9)
            .log()
            .median()
            .alias(f"z_logratio_med_{w}")
            for w in SHORT
        ],
        *[
            (pl.col(f"z_{w}").abs().median() / ROBUST).alias(f"robust_z_{w}")
            for w in WINDOWS
        ],
        *[
            ((pl.col(event_vol_column(w)) ** 2).log() + pl.col(f"z_{w}") ** 2)
            .mean()
            .alias(f"qlike_{w}")
            for w in WINDOWS
        ],
    ).row(0, named=True)
    row["Date"] = date
    buckets = (
        events.group_by("bucket")
        .agg(
            pl.len().alias("n_events"),
            *[
                (pl.col(f"z_{w}").abs().median() / ROBUST).alias(f"robust_z_{w}")
                for w in WINDOWS
            ],
            *[
                ((pl.col(event_vol_column(w)) ** 2).log() + pl.col(f"z_{w}") ** 2)
                .mean()
                .alias(f"qlike_{w}")
                for w in WINDOWS
            ],
        )
        .with_columns(pl.lit(date).alias("Date"))
    )
    return row, buckets


def _paired_t(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff)]
    if len(diff) < 3 or diff.std(ddof=1) == 0.0:
        return float("nan")
    return float(diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))))


def _md_table(frame: pl.DataFrame, digits: int = 4) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [header, rule]
    for row in frame.iter_rows():
        cells = [
            f"{value:.{digits}f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(
    agreement: pl.DataFrame,
    nextday: pl.DataFrame,
    event_daily: pl.DataFrame,
    event_bucket: pl.DataFrame,
    parity: pl.DataFrame,
) -> str:
    out: list[str] = []
    years = pl.col("Date").str.slice(0, 4).alias("year")

    out.append("## Parity (recomputed 60 vs preMarket in data/)")
    out.append(
        _md_table(
            parity.select(
                pl.len().alias("dates"),
                pl.col("code_days").sum(),
                pl.col("beta_mismatch").sum(),
                pl.col("vol_mismatch").sum(),
                (pl.col("beta_mismatch") + pl.col("vol_mismatch") > 0)
                .sum()
                .alias("dates_with_mismatch"),
            )
        )
    )

    out.append("\n## 1. Agreement (median over dates)")
    agree_cols = [c for c in agreement.columns if c not in ("Date", "n_codes", "n_common")]
    out.append(
        _md_table(
            agreement.group_by(years)
            .agg(pl.len().alias("dates"), *[pl.col(c).median() for c in agree_cols])
            .sort("year")
            .vstack(
                agreement.select(
                    pl.lit("all").alias("year"),
                    pl.len().cast(pl.UInt32).alias("dates"),
                    *[pl.col(c).median() for c in agree_cols],
                )
            ),
            3,
        )
    )

    out.append("\n## 2. Next-session open-to-close fit (equal-day mean)")
    hedge = nextday.select(
        pl.len().alias("dates"),
        pl.col("n").sum().alias("stock_days"),
        *[
            (pl.col(f"msq_{k}").mean() / pl.col("msq_beta0").mean()).alias(
                f"resid_var_share_{k}"
            )
            for k in ("beta1", *WINDOWS)
        ],
    )
    out.append("Residual variance as share of unhedged variance (lower is better):")
    out.append(_md_table(hedge))
    fit_rows = []
    for w in WINDOWS:
        fit_rows.append(
            {
                "window": w,
                "rms_z": float(nextday[f"rms_z_{w}"].mean()),
                "robust_z": float(nextday[f"robust_z_{w}"].median()),
                "qlike": float(nextday[f"qlike_{w}"].mean()),
                "qlike_minus_60": float(
                    (nextday[f"qlike_{w}"] - nextday["qlike_60"]).mean()
                ),
                "t_qlike_vs_60": _paired_t(
                    (nextday[f"qlike_{w}"] - nextday["qlike_60"]).to_numpy()
                ),
                "msq_minus_60_bp2": float(
                    (nextday[f"msq_{w}"] - nextday["msq_60"]).mean() * 1e8
                ),
                "t_msq_vs_60": _paired_t(
                    (nextday[f"msq_{w}"] - nextday["msq_60"]).to_numpy()
                ),
            }
        )
    out.append("\nVol fit (z = own-beta residual / own vol; calibrated rms_z/robust_z near 1):")
    out.append(_md_table(pl.DataFrame(fit_rows)))
    out.append("\nQLIKE minus 60 by year (negative = shorter window fits better):")
    out.append(
        _md_table(
            nextday.group_by(years)
            .agg(
                pl.len().alias("dates"),
                *[(pl.col(f"qlike_{w}") - pl.col("qlike_60")).mean().alias(f"dqlike_{w}") for w in SHORT],
                *[
                    ((pl.col(f"msq_{w}") - pl.col("msq_60")).mean() * 1e8).alias(f"dmsq_bp2_{w}")
                    for w in SHORT
                ],
            )
            .sort("year"),
        )
    )

    out.append("\n## 3. Event residual-z label (median over dates)")
    ev_cols = [c for c in event_daily.columns if c not in ("Date", "n_events")]
    out.append(
        _md_table(
            event_daily.group_by(years)
            .agg(pl.len().alias("dates"), *[pl.col(c).median() for c in ev_cols])
            .sort("year")
            .vstack(
                event_daily.select(
                    pl.lit("all").alias("year"),
                    pl.len().cast(pl.UInt32).alias("dates"),
                    *[pl.col(c).median() for c in ev_cols],
                )
            ),
            3,
        )
    )
    out.append("\nEvent z calibration by session hour (median over dates):")
    out.append(
        _md_table(
            event_bucket.group_by("bucket")
            .agg(
                pl.len().alias("dates"),
                pl.col("n_events").sum(),
                *[pl.col(f"robust_z_{w}").median() for w in WINDOWS],
                *[(pl.col(f"qlike_{w}") - pl.col("qlike_60")).mean().alias(f"dqlike_{w}") for w in SHORT],
            )
            .sort("bucket")
            .with_columns(
                pl.col("bucket").map_elements(
                    lambda b: SESSION_BUCKETS[b], return_dtype=pl.Utf8
                )
            ),
            3,
        )
    )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rebuild", action="store_true", help="recompute the table")
    parser.add_argument("--start", default=None, help="first decision date to evaluate")
    parser.add_argument("--end", default=None, help="last decision date to evaluate")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    market_dir, txf_dir = storage_dirs()
    decisions = event_dates()
    codes_by_date = event_codes_by_date(decisions)
    all_codes = sorted({c for codes in codes_by_date.values() for c in codes})
    print(f"loading return history for {len(all_codes)} codes")
    history = load_return_history(
        market_dates(market_dir), all_codes, market_dir, TxfDataLoader(txf_dir)
    )

    if args.rebuild or not TABLE_PATH.exists():
        print(f"building lookback table for {len(decisions)} decision dates")
        table = build_lookback_table(decisions, history=history)
        table.write_parquet(TABLE_PATH)
    else:
        table = pl.read_parquet(TABLE_PATH)

    eval_dates = [
        d
        for d in table.filter(pl.col("window_full_60"))["Date"].unique().sort().to_list()
        if (args.start is None or d >= args.start) and (args.end is None or d <= args.end)
    ]
    table = table.filter(pl.col("Date").is_in(eval_dates))
    print(f"evaluating {len(eval_dates)} dates {eval_dates[0]}..{eval_dates[-1]}")

    agreement = agreement_by_date(table)
    nextday = nextday_by_date(table, history)

    parity_rows, event_rows, bucket_frames = [], [], []
    tables_by_date = table.partition_by("Date", as_dict=True)
    for date in tqdm(eval_dates, desc="events"):
        events, parity = _event_frame(date, tables_by_date[(date,)])
        parity_rows.append(parity)
        if events.is_empty():
            continue
        row, buckets = event_metrics(events, date)
        event_rows.append(row)
        bucket_frames.append(buckets)
    parity = pl.DataFrame(parity_rows)
    event_daily = pl.DataFrame(event_rows)
    event_bucket = pl.concat(bucket_frames, how="vertical")

    agreement.write_csv(ARTIFACT_DIR / "agreement_by_date.csv")
    nextday.write_csv(ARTIFACT_DIR / "nextday_by_date.csv")
    parity.write_csv(ARTIFACT_DIR / "parity_by_date.csv")
    event_daily.write_csv(ARTIFACT_DIR / "event_by_date.csv")
    event_bucket.write_csv(ARTIFACT_DIR / "event_by_bucket.csv")

    summary = summarize(agreement, nextday, event_daily, event_bucket, parity)
    (ARTIFACT_DIR / "summary.md").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
