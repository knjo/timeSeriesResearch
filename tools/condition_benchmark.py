"""Reference CLI for the generic condition-study framework.

negFill is a calibration event, not a framework primitive.  This adapter pins
the historical Taiwan-equity eligibility coordinates and label profiles while
the core comparison guarantees that condition and control come from one
eligible, label-valid population.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from src.dataloader.txfDataLoader import TxfDataLoader
from src.research.intradaySignal.framework.condition_study import (
    ConditionComparisonSpec,
    EligiblePoolSpec,
    PrimaryEstimand,
    build_eligible_labeled_pool,
    compare_condition_to_control,
    read_day_event_keys,
    summarize_condition_day,
)
from src.research.intradaySignal.framework.events import DataRoots
from src.research.intradaySignal.framework.labels import DirectionalLabelSpec
from src.research.intradaySignal.framework.profiles.benchmark import LONG_TAKER_TP109_V1
from src.research.intradaySignal.framework.profiles.fhl import LONG_MAKER_TP109_V1
from src.research.intradaySignal.framework.profiles.negfill import SHORT_TAKER_STOP108_V1


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVENT_INDEX = Path(__file__).with_name("data_index.parquet")

T_EDGES = [3300.0, 6900.0, 10500.0, 14100.0]
T_LABEL = {
    "1": "11:30-12:30",
    "2": "10:30-11:30",
    "3": "09:30-10:30",
    "4": "09:00-09:30",
}
R_EDGES = [-0.005, 0.0, 0.01, 0.02, 0.03]
R_LABEL = {
    "0": "-1.5~-0.5",
    "1": "-0.5~0",
    "2": "0~1",
    "3": "1~2",
    "4": "2~3",
    "5": "3~5",
}

LABEL_PROFILES: dict[str, DirectionalLabelSpec] = {
    "short-taker-stop108-v1": SHORT_TAKER_STOP108_V1,
    "long-taker-tp109-v1": LONG_TAKER_TP109_V1,
    "long-maker-tp109-v1": LONG_MAKER_TP109_V1,
}
LEGACY_SIDE_PROFILE = {
    "short": "short-taker-stop108-v1",
    "long": "long-taker-tp109-v1",
}


def tick_size(price: pl.Expr) -> pl.Expr:
    return (
        pl.when(price < 10)
        .then(0.01)
        .when(price < 50)
        .then(0.05)
        .when(price < 100)
        .then(0.1)
        .when(price < 500)
        .then(0.5)
        .when(price < 1000)
        .then(1.0)
        .otherwise(5.0)
    )


def _reference_pool_transform(entry_col: str):
    def transform(frame):
        bid_to_ref = (
            (pl.col("BidPrice1") - pl.col("RefPrice")) / pl.col("RefPrice")
        ).round(6)
        seconds_to_1325 = (
            pl.lit(dt.time(13, 25)) - pl.col("TransTime").dt.time()
        ).dt.total_seconds()
        return (
            frame.filter(
                (pl.col(entry_col) > 0)
                & (pl.col("RefPrice") > 0)
                & (pl.col("TrialMatch") == 0)
                & (pl.col("TransTime").dt.time() > dt.time(9, 0, 30))
                & (pl.col("TransTime").dt.time() < dt.time(12, 30))
                & (pl.col("nextday_allow_day_trade_mark") == "X")
                & (pl.col("day_amount_rank") <= 150)
            )
            .with_columns(
                tick_size(pl.col("BidPrice1")).alias("TickSize"),
                bid_to_ref.alias("bid_to_ref"),
                seconds_to_1325.alias("seconds_to_1325"),
            )
            .filter(
                (pl.col("bid_to_ref") > -0.015)
                & (pl.col("bid_to_ref") < 0.05)
            )
            .with_columns(
                pl.col("seconds_to_1325")
                .cut(T_EDGES, labels=[str(i) for i in range(5)])
                .alias("legacy_time_bucket_1325_v1"),
                pl.col("bid_to_ref")
                .cut(R_EDGES, labels=[str(i) for i in range(6)])
                .alias("bid_to_ref_bucket_v1"),
            )
        )

    return transform


def _raw_quality_filter(frame):
    return frame.filter(
        pl.col("label_raw_bp").is_finite()
        & (pl.col("label_raw_bp").abs() < 10_000)
    )


def reference_pool_spec(label_spec: DirectionalLabelSpec) -> EligiblePoolSpec:
    """Pinned eligibility contract used to calibrate the negFill reference."""

    return EligiblePoolSpec(
        pool_id="tw_equity_daytrade_rank150_090030_123000_bidref_v1",
        pool_version="1",
        tick_columns=(
            "QuoteCode",
            "ChannelSeq",
            "TransTime",
            "BidPrice1",
            "AskPrice1",
            "Close",
            "FutureHigh",
            "RefPrice",
            "TrialMatch",
        ),
        pre_market_columns=(
            "QuoteCode",
            "day_amount_rank",
            "nextday_allow_day_trade_mark",
            "txf_beta_60d",
            "txf_residual_vol_0900_1330_60d",
        ),
        transform_and_filter=_reference_pool_transform(
            label_spec.entry.price_col
        ),
        post_label_filter=_raw_quality_filter,
    )


def _study_dates(
    *,
    event_index: Path | str | None,
    events_dir: Path | str | None,
    start: str,
    end: str,
    step: int,
) -> list[str]:
    if events_dir is not None:
        dates = sorted(
            path.stem
            for path in Path(events_dir).glob("*.parquet")
            if start <= path.stem <= end
        )
    else:
        if event_index is None:
            raise ValueError("event_index is required when events_dir is absent")
        path = Path(event_index)
        if not path.exists():
            raise FileNotFoundError(path)
        all_dates = (
            pl.read_parquet(path, columns=["Date"])["Date"]
            .cast(pl.String)
            .unique()
            .sort()
            .to_list()
        )
        dates = [date for date in all_dates if start <= date <= end]
    if step < 1:
        raise ValueError("step must be positive")
    return dates[::step]


def scan(
    events_dir: Path | str | None,
    start: str,
    end: str,
    step: int,
    sample: int,
    side: str = "short",
    *,
    event_index: Path | str | None = None,
    label_profile: str | None = None,
    fail_fast: bool = True,
) -> list[dict[str, object]]:
    """Run the negFill reference study through generic shared-pool contracts."""

    profile_name = label_profile or LEGACY_SIDE_PROFILE[side]
    profile = LABEL_PROFILES[profile_name]
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    loader = TxfDataLoader(roots.txf_tick_dir)
    pool_spec = reference_pool_spec(profile)
    comparison_spec = ConditionComparisonSpec(
        condition_id="negfill_reference",
        condition_version="shared_valid_pool_v2",
        control_sample_size=sample,
        control_population="eligible_pool",
        primary_estimand=PrimaryEstimand.SHARED_POOL,
        bucket_columns=(
            "legacy_time_bucket_1325_v1",
            "bid_to_ref_bucket_v1",
        ),
    )
    dates = _study_dates(
        event_index=event_index,
        events_dir=events_dir,
        start=start,
        end=end,
        step=step,
    )
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        try:
            pool = build_eligible_labeled_pool(
                date,
                roots=roots,
                pool_spec=pool_spec,
                label_spec=profile,
                txf_loader=loader,
            )
            keys = read_day_event_keys(
                date,
                event_index_path=(event_index if events_dir is None else None),
                events_dir=events_dir,
            )
            result = compare_condition_to_control(
                date,
                pool,
                keys,
                spec=comparison_spec,
                seed=int(date),
            )
            day_rows = summarize_condition_day(
                result, bucket_columns=comparison_spec.bucket_columns
            )
            for row in day_rows:
                row["label_profile_id"] = profile.profile_id
                row["label_profile_hash"] = profile.stable_hash()
                row["pool_id"] = pool_spec.pool_id
                row["study_version"] = comparison_spec.condition_version
                rows.append(row)
            if index % 20 == 0:
                print(
                    f"[{index + 1}/{len(dates)}] {date}: "
                    f"condition={result.condition_rows:,}, "
                    f"pool={result.eligible_pool_rows:,}",
                    flush=True,
                )
        except Exception as exc:
            if fail_fast:
                raise RuntimeError(f"{date}: condition study failed") from exc
            print(f"{date}: ERROR {exc}", flush=True)
    return rows


def report(rows: list[dict[str, object]]) -> None:
    import pandas as pd

    if not rows:
        raise ValueError("condition study produced no rows")
    frame = pd.DataFrame(rows)
    frame["ym"] = frame.Date.str[:6]
    condition = frame[frame.cohort == "condition"]
    control = frame[frame.cohort == "control"]
    selection = condition.set_index("Date").n / condition.set_index(
        "Date"
    ).n_eligible_pool
    daily_n = condition.n
    profile = frame.label_profile_id.iloc[0]
    study_version = frame.study_version.iloc[0]
    print(f"\nprofile={profile} | study={study_version}")
    print(
        f"[A] 選擇率 median {selection.median() * 100:.2f}% "
        f"(p10-p90 {selection.quantile(.1) * 100:.2f}~"
        f"{selection.quantile(.9) * 100:.2f}%) | "
        f"樣本/日 mean {daily_n.mean():,.0f} "
        f"cv {daily_n.std() / daily_n.mean():.2f}"
    )

    def pooled(sub):
        n = sub.n_z.sum()
        z_mean = sub.sum_z.sum() / n
        variance = max(sub.sum_z2.sum() / n - z_mean**2, 0.0)
        return {
            "raw_bp": sub.sum_raw_bp.sum() / n,
            "residual_bp": sub.sum_residual_bp.sum() / n,
            "z_mean": z_mean,
            "z_std": float(np.sqrt(variance)),
            "market_component_bp": sub.sum_market_component_bp.sum() / n,
            "hedge_pnl_bp": sub.sum_hedge_pnl_bp.sum() / n,
        }

    condition_pooled, control_pooled = pooled(condition), pooled(control)
    sum_z = frame.pivot_table(index="Date", columns="cohort", values="sum_z")
    n_z = frame.pivot_table(index="Date", columns="cohort", values="n_z")
    delta_z = (sum_z.condition / n_z.condition) - (
        sum_z.control / n_z.control
    )
    month_for_date = frame.groupby("Date").ym.first()
    monthly = delta_z.groupby(month_for_date).mean()
    print(
        f"[B] Δz {condition_pooled['z_mean'] - control_pooled['z_mean']:+.3f}σ | "
        f"Δres {condition_pooled['residual_bp'] - control_pooled['residual_bp']:+.1f}bp | "
        f"Δraw {condition_pooled['raw_bp'] - control_pooled['raw_bp']:+.1f}bp | "
        f"Δmarket {condition_pooled['market_component_bp'] - control_pooled['market_component_bp']:+.1f}bp | "
        f"ΔhedgePnL {condition_pooled['hedge_pnl_bp'] - control_pooled['hedge_pnl_bp']:+.1f}bp"
    )
    daily_t = delta_z.mean() / delta_z.std() * np.sqrt(len(delta_z))
    print(
        f"    日級 t={daily_t:.1f} | 正Δz日 {(delta_z > 0).mean():.0%} | "
        f"正Δz月 {(monthly > 0).sum()}/{len(monthly)} | "
        f"control z_std {control_pooled['z_std']:.2f} (應≈1)"
    )
    print("    月度Δz:", {key: round(value, 3) for key, value in monthly.items()})

    def share(sub, axis):
        totals: dict[str, int] = {}
        for grid in sub.grid:
            for key, value in grid.items():
                totals[key] = totals.get(key, 0) + value
        series = pd.Series(totals, dtype=float)
        groups = [key.split("|")[axis] for key in series.index]
        series = series.groupby(groups).sum()
        return series / series.sum()

    for axis, label, names in (
        (0, "時段", T_LABEL),
        (1, "Bid/Ref", R_LABEL),
    ):
        condition_share = share(condition, axis)
        control_share = share(control, axis)
        union = control_share.index.union(condition_share.index)
        tv = (
            condition_share.reindex(union, fill_value=0)
            - control_share.reindex(union, fill_value=0)
        ).abs().sum() / 2
        table = pd.DataFrame(
            {"condition%": condition_share * 100, "control%": control_share * 100}
        ).rename(index=names)
        print(f"[C] {label} TV={tv:.3f}\n{table.round(1).to_string()}")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-index", type=Path, default=DEFAULT_EVENT_INDEX)
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=None,
        help="Legacy alternative: read event keys from merged per-day files",
    )
    parser.add_argument("--start", default="20250211")
    parser.add_argument("--end", default="20991231")
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--sample", type=int, default=25_000)
    parser.add_argument("--side", choices=["short", "long"], default=None)
    parser.add_argument("--label-profile", choices=sorted(LABEL_PROFILES), default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.side is not None and args.label_profile is not None:
        parser.error("--side and --label-profile are mutually exclusive")
    side = args.side or "short"
    report(
        scan(
            args.events_dir,
            args.start,
            args.end,
            args.step,
            args.sample,
            side,
            event_index=(None if args.events_dir is not None else args.event_index),
            label_profile=args.label_profile,
            fail_fast=not args.continue_on_error,
        )
    )


if __name__ == "__main__":
    _main()
