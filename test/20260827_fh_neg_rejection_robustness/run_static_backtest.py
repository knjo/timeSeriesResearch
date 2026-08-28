"""Keep a retrospective static BackTest snapshot for the conservative anchor.

This runner performs no condition or model selection.  It reconstructs the
2026-08-27 conservative D/G/B cohort from pinned artifacts, reproduces the
robustness pool's rounded boundary and legal tick-size label, then applies the
previously used static execution and one-stock capacity contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from src.research.backTest import BackTest


PROJECT_ROOT = Path(__file__).resolve().parents[5]
STUDY_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = STUDY_DIR / "artifacts"
OUTPUT_DIR = ARTIFACT_ROOT / "static_backtest_retrospective_v1"
CONTRACT_PATH = STUDY_DIR / "static_backtest_contract.json"

PARENT_FACTORIAL_DIR = ARTIFACT_ROOT / "dgb_factorial_v1"
MEMBERSHIP_DIR = PARENT_FACTORIAL_DIR / "membership_daily"
PARENT_DAILY_STATS_PATH = PARENT_FACTORIAL_DIR / "daily_stats.parquet"
PARENT_RUN_MANIFEST_PATH = PARENT_FACTORIAL_DIR / "run_manifest.json"
PARENT_DECISION_PATH = ARTIFACT_ROOT / "final_decision_v1" / "decision.json"
PARENT_REFERENCE_LINEAGE_PATH = (
    ARTIFACT_ROOT / "reference_lineage_v1" / "reference_lineage.parquet"
)
CALENDAR_PATH = (
    ARTIFACT_ROOT
    / "calendar_retrospective_v1"
    / "requested_calendar.parquet"
)

LEGACY_SIGNAL_STUDY_DIR = (
    PROJECT_ROOT
    / "src/research/timeSeries/test/20260826_filllots_at_high_signal"
)
MODEL_INPUT_PATH = (
    LEGACY_SIGNAL_STUDY_DIR / "artifacts/model_v1/model_input.parquet"
)
LEGACY_STUDY_PATH = LEGACY_SIGNAL_STUDY_DIR / "study.py"
BACKTEST_SOURCE_PATH = PROJECT_ROOT / "src/research/backTest.py"
LABEL_SOURCE_PATH = (
    PROJECT_ROOT / "src/research/timeSeries/framework/labels.py"
)
POOL_SOURCE_PATH = PROJECT_ROOT / "src/research/timeSeries/condition_benchmark.py"

START = "20250211"
END = "20260825"
LEGACY_MODEL_ERA_START = "20250513"
PRIMARY_FEE_BP = 19.3
FEE_GRID_BP = (0.0, 10.0, 19.3, 25.0, 35.0)
KEY_COLUMNS = ("Date", "QuoteCode", "ChannelSeq")
EXPECTED_PRELABEL_MEMBERSHIP = 6_422_686
EXPECTED_LABELED_SUPPORT = 850_376
EXPECTED_LABELED_KEY_SHA256 = (
    "eff1e68b5017e82a2dc044262e8747bb5513012b4239d8528eb43616a58fc993"
)
EXPECTED_ALIGNED_SIGNAL_ROWS = 58_518
EXPECTED_ALIGNED_SIGNAL_KEY_SHA256 = (
    "d81e26ebb2457b27d20d3e1317dfefcd134068899a5489d83cc0f13d947015d6"
)

ANCHOR_KEYS_PATH = OUTPUT_DIR / "anchor_keys.parquet"
SIGNALS_PATH = OUTPUT_DIR / "accepted_signals.parquet"
DAILY_PATH = OUTPUT_DIR / "daily.parquet"
RECONCILIATION_PATH = OUTPUT_DIR / "cohort_reconciliation.parquet"
PERIOD_SUMMARY_PATH = OUTPUT_DIR / "period_summary.parquet"
MONTHLY_SUMMARY_PATH = OUTPUT_DIR / "monthly_summary.parquet"
FEE_SENSITIVITY_PATH = OUTPUT_DIR / "fee_sensitivity.parquet"
FULL_PLOT_PATH = OUTPUT_DIR / "backtest_full.png"
HISTORICAL_2026_PLOT_PATH = OUTPUT_DIR / "backtest_2026_historical.png"
DECISION_PATH = OUTPUT_DIR / "snapshot_decision.json"
VERIFICATION_PATH = OUTPUT_DIR / "verification.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _load_legacy_study() -> ModuleType:
    name = "_fh_signal_study_20260826_static_snapshot"
    spec = importlib.util.spec_from_file_location(name, LEGACY_STUDY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import execution contract: {LEGACY_STUDY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for required in (
        "add_threshold_buckets",
        "static_execution_filter",
        "apply_capacity",
    ):
        if not hasattr(module, required):
            raise RuntimeError(f"legacy execution contract lacks {required}")
    return module


def _assert_unique_keys(frame: pl.DataFrame, context: str) -> None:
    missing = sorted(set(KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{context}: missing key columns {missing}")
    nulls = frame.select(
        pl.any_horizontal([pl.col(column).is_null() for column in KEY_COLUMNS])
        .sum()
        .alias("n")
    ).item()
    if nulls:
        raise RuntimeError(f"{context}: null keys={nulls}")
    duplicates = frame.select(pl.struct(KEY_COLUMNS).is_duplicated().sum()).item()
    if duplicates:
        raise RuntimeError(f"{context}: duplicate keys={duplicates}")


def _key_sha256(frame: pl.DataFrame) -> str:
    """Hash sorted canonical keys with an explicit, language-neutral encoding."""

    digest = hashlib.sha256()
    ordered = frame.select(KEY_COLUMNS).sort(KEY_COLUMNS)
    for date, quote_code, channel_seq in ordered.iter_rows():
        digest.update(
            f"{date}\x1f{quote_code}\x1f{int(channel_seq)}\n".encode("utf-8")
        )
    return digest.hexdigest()


def legal_tick_size(price: pl.Expr) -> pl.Expr:
    """Match the tick-size producer frozen by the robustness eligible pool."""

    return (
        pl.when(price < 10.0)
        .then(0.01)
        .when(price < 50.0)
        .then(0.05)
        .when(price < 100.0)
        .then(0.1)
        .when(price < 500.0)
        .then(0.5)
        .when(price < 1000.0)
        .then(1.0)
        .otherwise(5.0)
    )


def apply_robustness_pool_boundary(frame: pl.DataFrame) -> pl.DataFrame:
    bid_to_ref = (
        (pl.col("BidPrice1") - pl.col("RefPrice")) / pl.col("RefPrice")
    ).round(6)
    return (
        frame.with_columns(bid_to_ref.alias("bid_to_ref_robustness"))
        .filter(
            (pl.col("bid_to_ref_robustness") > -0.015)
            & (pl.col("bid_to_ref_robustness") < 0.05)
        )
        .with_columns(pl.col("bid_to_ref_robustness").alias("ToRef"))
    )


def recompute_robustness_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """Replace the legacy tickFeature tick label with the robustness label."""

    required = {
        "BidPrice1",
        "Close",
        "FutureHigh",
        "RefPrice",
        "TickSize",
        "label_raw_bp",
        "label_residual_bp",
        "label_z",
        "txf_residual_vol_to_1330",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"label reconstruction missing columns: {missing}")

    output = frame.with_columns(
        pl.col("TickSize").alias("TickSize_legacy_feature"),
        pl.col("label_raw_bp").alias("label_raw_bp_legacy_tick"),
        pl.col("label_residual_bp").alias(
            "label_residual_bp_legacy_tick"
        ),
        pl.col("label_z").alias("label_z_legacy_tick"),
        legal_tick_size(pl.col("BidPrice1")).alias("TickSize"),
        (pl.col("RefPrice") * 1.08).alias("_barrier_price"),
    )
    output = output.with_columns(
        pl.when(pl.col("FutureHigh") > pl.col("_barrier_price"))
        .then(pl.col("_barrier_price") + 2.0 * pl.col("TickSize"))
        .otherwise(pl.col("Close"))
        .alias("label_exit_price_robustness"),
        (
            pl.col("label_residual_bp_legacy_tick")
            - pl.col("label_raw_bp_legacy_tick")
        ).alias("_hedge_pnl_bp"),
    )
    output = output.with_columns(
        (
            (pl.col("BidPrice1") - pl.col("label_exit_price_robustness"))
            / pl.col("BidPrice1")
            * 10_000.0
        ).alias("label_raw_bp"),
        (
            pl.col("txf_residual_vol_to_1330").cast(pl.Float64) * 10_000.0
        ).alias("_label_vol_bp"),
    )
    output = output.with_columns(
        (pl.col("label_raw_bp") + pl.col("_hedge_pnl_bp")).alias(
            "label_residual_bp"
        )
    )
    output = output.with_columns(
        pl.when(pl.col("_label_vol_bp").is_finite() & (pl.col("_label_vol_bp") > 1e-12))
        .then(pl.col("label_residual_bp") / pl.col("_label_vol_bp"))
        .alias("label_z")
    )
    return output.filter(
        pl.col("label_raw_bp").is_finite()
        & (pl.col("label_raw_bp").abs() < 10_000.0)
        & pl.col("label_residual_bp").is_finite()
        & pl.col("label_z").is_finite()
    ).drop("_barrier_price", "_hedge_pnl_bp", "_label_vol_bp")


def _read_calendar() -> list[str]:
    calendar = pl.read_parquet(CALENDAR_PATH)
    dates = (
        calendar.filter(
            (pl.col("Date") >= START)
            & (pl.col("Date") <= END)
            & (pl.col("evidence_class") == "retrospective_diagnostic")
        )["Date"]
        .unique()
        .sort()
        .to_list()
    )
    if not dates or dates[0] != START or dates[-1] != END:
        raise RuntimeError("retrospective calendar does not cover frozen dates")
    if len(dates) != len(set(dates)):
        raise RuntimeError("retrospective calendar contains duplicate dates")
    return dates


def _membership_paths(calendar: Iterable[str]) -> list[Path]:
    dates = list(calendar)
    paths = [MEMBERSHIP_DIR / f"{date}.parquet" for date in dates]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing membership partitions: {missing[:5]}")
    return paths


def build_anchor_cohort(
    calendar: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any], list[Path]]:
    membership_paths = _membership_paths(calendar)
    membership = (
        pl.scan_parquet(membership_paths)
        .filter(pl.col("conservative_dgb").fill_null(False))
        .select(KEY_COLUMNS)
        .collect()
    )
    _assert_unique_keys(membership, "conservative pre-label membership")
    if len(membership) != EXPECTED_PRELABEL_MEMBERSHIP:
        raise RuntimeError(
            f"pre-label anchor drift: {len(membership)} != "
            f"{EXPECTED_PRELABEL_MEMBERSHIP}"
        )

    model_input = pl.read_parquet(MODEL_INPUT_PATH).filter(
        (pl.col("Date") >= START) & (pl.col("Date") <= END)
    )
    _assert_unique_keys(model_input, "legacy model input")
    intersection = model_input.join(
        membership, on=KEY_COLUMNS, how="inner", validate="1:1"
    )
    bounded = apply_robustness_pool_boundary(intersection)
    cohort = recompute_robustness_labels(bounded).sort(
        ["Date", "QuoteCode", "TransTime", "ChannelSeq"]
    )
    _assert_unique_keys(cohort, "reconstructed robustness labeled anchor")
    if len(cohort) != EXPECTED_LABELED_SUPPORT:
        raise RuntimeError(
            f"labeled anchor drift: {len(cohort)} != {EXPECTED_LABELED_SUPPORT}"
        )
    cohort_hash = _key_sha256(cohort)
    if cohort_hash != EXPECTED_LABELED_KEY_SHA256:
        raise RuntimeError(
            f"labeled anchor key hash drift: {cohort_hash} != "
            f"{EXPECTED_LABELED_KEY_SHA256}"
        )

    counts = {
        "model_input_rows": len(model_input),
        "prelabel_membership_rows": len(membership),
        "model_membership_intersection_rows": len(intersection),
        "rounded_boundary_removed_rows": len(intersection) - len(bounded),
        "labeled_anchor_rows": len(cohort),
        "labeled_anchor_key_sha256": cohort_hash,
    }
    return cohort, membership, counts, membership_paths


def reconcile_parent_moments(
    cohort: pl.DataFrame,
    membership: pl.DataFrame,
    calendar: list[str],
    model_membership_intersection: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, float]]:
    actual = cohort.group_by("Date").agg(
        pl.len().alias("reconstructed_condition_n_z"),
        pl.col("label_raw_bp").sum().alias("reconstructed_sum_raw_bp"),
        pl.col("label_residual_bp")
        .sum()
        .alias("reconstructed_sum_residual_bp"),
        pl.col("label_z").sum().alias("reconstructed_sum_z"),
        (pl.col("label_z") ** 2).sum().alias("reconstructed_sum_z2"),
    )
    expected = (
        pl.read_parquet(PARENT_DAILY_STATS_PATH)
        .filter(
            (pl.col("candidate_id") == "d1_g1_b1")
            & (pl.col("Date") >= START)
            & (pl.col("Date") <= END)
        )
        .select(
            "Date",
            pl.col("condition_n_z").alias("expected_condition_n_z"),
            pl.col("condition_sum_raw_bp").alias("expected_sum_raw_bp"),
            pl.col("condition_sum_residual_bp").alias(
                "expected_sum_residual_bp"
            ),
            pl.col("condition_sum_z").alias("expected_sum_z"),
            pl.col("condition_sum_z2").alias("expected_sum_z2"),
        )
    )
    if len(expected) != len(calendar):
        raise RuntimeError(
            f"parent daily row count drift: {len(expected)} != {len(calendar)}"
        )

    membership_daily = membership.group_by("Date").agg(
        pl.len().alias("prelabel_membership_n")
    )
    intersection_daily = model_membership_intersection.group_by("Date").agg(
        pl.len().alias("model_membership_intersection_n")
    )
    reconciliation = (
        pl.DataFrame({"Date": calendar})
        .join(membership_daily, on="Date", how="left")
        .join(intersection_daily, on="Date", how="left")
        .join(actual, on="Date", how="left")
        .join(expected, on="Date", how="left")
        .with_columns(
            pl.col("prelabel_membership_n").fill_null(0),
            pl.col("model_membership_intersection_n").fill_null(0),
            pl.col("reconstructed_condition_n_z").fill_null(0),
            pl.col("reconstructed_sum_raw_bp").fill_null(0.0),
            pl.col("reconstructed_sum_residual_bp").fill_null(0.0),
            pl.col("reconstructed_sum_z").fill_null(0.0),
            pl.col("reconstructed_sum_z2").fill_null(0.0),
        )
        .with_columns(
            (
                pl.col("model_membership_intersection_n")
                - pl.col("reconstructed_condition_n_z")
            ).alias("rounded_boundary_removed_n"),
            (
                pl.col("reconstructed_condition_n_z")
                - pl.col("expected_condition_n_z")
            ).alias("condition_n_diff"),
            (
                pl.col("reconstructed_sum_raw_bp")
                - pl.col("expected_sum_raw_bp")
            ).alias("sum_raw_bp_diff"),
            (
                pl.col("reconstructed_sum_residual_bp")
                - pl.col("expected_sum_residual_bp")
            ).alias("sum_residual_bp_diff"),
            (
                pl.col("reconstructed_sum_z") - pl.col("expected_sum_z")
            ).alias("sum_z_diff"),
            (
                pl.col("reconstructed_sum_z2") - pl.col("expected_sum_z2")
            ).alias("sum_z2_diff"),
        )
        .sort("Date")
    )

    if reconciliation.filter(pl.col("condition_n_diff") != 0).height:
        raise RuntimeError("daily anchor support does not match parent robustness")
    moment_pairs = (
        ("sum_raw_bp_diff", "expected_sum_raw_bp"),
        ("sum_residual_bp_diff", "expected_sum_residual_bp"),
        ("sum_z_diff", "expected_sum_z"),
        ("sum_z2_diff", "expected_sum_z2"),
    )
    max_errors: dict[str, float] = {}
    for diff_column, expected_column in moment_pairs:
        diff = reconciliation[diff_column].to_numpy()
        expected_values = reconciliation[expected_column].to_numpy()
        actual_values = diff + expected_values
        if not np.allclose(
            actual_values,
            expected_values,
            rtol=1e-10,
            atol=1e-6,
            equal_nan=False,
        ):
            bad = reconciliation.filter(pl.col(diff_column).abs() > 1e-6)
            raise RuntimeError(
                f"parent moment parity failed for {diff_column}; "
                f"sample={bad.head(5).to_dicts()}"
            )
        max_errors[diff_column] = float(np.max(np.abs(diff)))
    return reconciliation, max_errors


def build_static_signals(
    cohort: pl.DataFrame, legacy_study: ModuleType
) -> tuple[pl.DataFrame, pl.DataFrame]:
    bucketed = legacy_study.add_threshold_buckets(cohort)
    pre_cap = legacy_study.static_execution_filter(bucketed).sort(
        ["Date", "QuoteCode", "TransTime", "ChannelSeq"]
    )
    accepted = (
        legacy_study.apply_capacity(pre_cap)
        .with_columns(
            pl.lit("static_anchor_no_model_gate").alias("signal_variant"),
            pl.col("label_raw_bp").alias("TakerSell_CloseBP"),
            pl.lit(1, dtype=pl.Int8).alias("lots"),
        )
        .sort(["Date", "QuoteCode", "TransTime", "ChannelSeq"])
    )
    _assert_unique_keys(accepted, "accepted static signals")
    aligned = accepted.filter(pl.col("Date") >= LEGACY_MODEL_ERA_START)
    if len(aligned) != EXPECTED_ALIGNED_SIGNAL_ROWS:
        raise RuntimeError(
            f"legacy-era accepted-row parity failed: {len(aligned)} != "
            f"{EXPECTED_ALIGNED_SIGNAL_ROWS}"
        )
    aligned_hash = _key_sha256(aligned)
    if aligned_hash != EXPECTED_ALIGNED_SIGNAL_KEY_SHA256:
        raise RuntimeError(
            f"legacy-era signal key hash drift: {aligned_hash} != "
            f"{EXPECTED_ALIGNED_SIGNAL_KEY_SHA256}"
        )
    return pre_cap, accepted


def _daily_frame(
    signals: pl.DataFrame, calendar: list[str], fee_bp: float
) -> pl.DataFrame:
    subset = signals.filter(pl.col("Date").is_in(calendar))
    aggregated = (
        subset.with_columns(
            (pl.col("BidPrice1") * pl.col("label_raw_bp") / 10.0).alias(
                "gross_pnl"
            ),
            (
                pl.col("BidPrice1")
                * (pl.col("label_raw_bp") - fee_bp)
                / 10.0
            ).alias("net_pnl"),
            (
                pl.col("BidPrice1") * pl.col("label_residual_bp") / 10.0
            ).alias("residual_gross_pnl"),
            (
                pl.col("BidPrice1")
                * (pl.col("label_residual_bp") - fee_bp)
                / 10.0
            ).alias("residual_net_pnl"),
            (pl.col("BidPrice1") * 1000.0).alias("position"),
        )
        .group_by("Date")
        .agg(
            pl.col("gross_pnl").sum(),
            pl.col("net_pnl").sum(),
            pl.col("residual_gross_pnl").sum(),
            pl.col("residual_net_pnl").sum(),
            pl.col("position").sum(),
            pl.len().alias("trades"),
            pl.col("QuoteCode").n_unique().alias("stocks"),
        )
    )
    daily = (
        pl.DataFrame({"Date": calendar})
        .join(aggregated, on="Date", how="left")
        .with_columns(
            pl.col("gross_pnl").fill_null(0.0),
            pl.col("net_pnl").fill_null(0.0),
            pl.col("residual_gross_pnl").fill_null(0.0),
            pl.col("residual_net_pnl").fill_null(0.0),
            pl.col("position").fill_null(0.0),
            pl.col("trades").fill_null(0),
            pl.col("stocks").fill_null(0),
            pl.lit(float(fee_bp)).alias("fee_bp"),
        )
        .sort("Date")
    )
    cumulative = np.cumsum(daily["net_pnl"].to_numpy())
    high_water = np.maximum.accumulate(cumulative) if len(cumulative) else cumulative
    drawdown = cumulative - high_water
    return daily.with_columns(
        pl.Series("cumulative_net_pnl", cumulative),
        pl.Series("high_water_mark", high_water),
        pl.Series("drawdown", drawdown),
    )


def _period_summary(
    signals: pl.DataFrame,
    *,
    period: str,
    calendar: list[str],
    fee_bp: float,
) -> dict[str, Any]:
    subset = signals.filter(pl.col("Date").is_in(calendar))
    daily = _daily_frame(signals, calendar, fee_bp)
    pnl = daily["net_pnl"].to_numpy()
    residual_pnl = daily["residual_net_pnl"].to_numpy()
    position = float(daily["position"].sum())
    gross_pnl = float(daily["gross_pnl"].sum())
    net_pnl = float(daily["net_pnl"].sum())
    residual_gross_pnl = float(daily["residual_gross_pnl"].sum())
    residual_net_pnl = float(daily["residual_net_pnl"].sum())

    def sharpe(values: np.ndarray) -> float:
        if len(values) <= 1 or np.std(values, ddof=1) <= 0.0:
            return math.nan
        return float(
            np.mean(values) / np.std(values, ddof=1) * np.sqrt(252.0)
        )

    return {
        "signal_variant": "static_anchor_no_model_gate",
        "period": period,
        "fee_bp": float(fee_bp),
        "start": calendar[0] if calendar else None,
        "end": calendar[-1] if calendar else None,
        "calendar_days": len(calendar),
        "active_days": int((daily["trades"] > 0).sum()),
        "zero_signal_days": int((daily["trades"] == 0).sum()),
        "trades": int(daily["trades"].sum()),
        "distinct_stocks": int(subset["QuoteCode"].n_unique()) if len(subset) else 0,
        "stock_days": int(
            subset.select(pl.struct(["Date", "QuoteCode"]).n_unique()).item()
        )
        if len(subset)
        else 0,
        "gross_pnl": gross_pnl,
        "cost_pnl": gross_pnl - net_pnl,
        "net_pnl": net_pnl,
        "residual_gross_pnl": residual_gross_pnl,
        "residual_net_pnl": residual_net_pnl,
        "cap_weighted_gross_bp": gross_pnl / position * 10_000.0
        if position > 0.0
        else math.nan,
        "cap_weighted_net_bp": net_pnl / position * 10_000.0
        if position > 0.0
        else math.nan,
        "cap_weighted_residual_gross_bp": residual_gross_pnl
        / position
        * 10_000.0
        if position > 0.0
        else math.nan,
        "cap_weighted_residual_net_bp": residual_net_pnl
        / position
        * 10_000.0
        if position > 0.0
        else math.nan,
        "mean_event_raw_bp": float(subset["label_raw_bp"].mean())
        if len(subset)
        else math.nan,
        "mean_event_net_bp": float(subset["label_raw_bp"].mean() - fee_bp)
        if len(subset)
        else math.nan,
        "calendar_daily_sharpe": sharpe(pnl),
        "calendar_residual_daily_sharpe": sharpe(residual_pnl),
        "calendar_daily_win_rate": float(np.mean(pnl > 0.0))
        if len(pnl)
        else math.nan,
        "max_drawdown": float(daily["drawdown"].min()) if len(daily) else math.nan,
        "mean_daily_position": float(daily["position"].mean())
        if len(daily)
        else math.nan,
        "max_daily_position": float(daily["position"].max())
        if len(daily)
        else math.nan,
    }


def _period_calendars(
    all_dates: list[str], contract: dict[str, Any]
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for period, bounds in contract["periods"].items():
        start, end = bounds
        dates = [date for date in all_dates if start <= date <= end]
        if not dates:
            raise RuntimeError(f"period {period} has no calendar sessions")
        output[period] = dates
    return output


def _monthly_summary(daily: pl.DataFrame) -> pl.DataFrame:
    return (
        daily.with_columns(pl.col("Date").str.slice(0, 6).alias("month"))
        .group_by("month")
        .agg(
            pl.len().alias("calendar_days"),
            (pl.col("trades") > 0).sum().alias("active_days"),
            pl.col("trades").sum(),
            pl.col("gross_pnl").sum(),
            pl.col("net_pnl").sum(),
            pl.col("residual_gross_pnl").sum(),
            pl.col("residual_net_pnl").sum(),
            pl.col("position").sum(),
        )
        .with_columns(
            pl.lit(PRIMARY_FEE_BP).alias("fee_bp"),
            (pl.col("gross_pnl") / pl.col("position") * 10_000.0).alias(
                "cap_weighted_gross_bp"
            ),
            (pl.col("net_pnl") / pl.col("position") * 10_000.0).alias(
                "cap_weighted_net_bp"
            ),
        )
        .sort("month")
    )


def _save_project_backtest_plot(
    signals: pl.DataFrame,
    *,
    path: Path,
    title: str,
    fee_bp: float,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
    if signals.is_empty():
        raise RuntimeError(f"cannot plot empty signal set: {title}")
    pdf = signals.to_pandas()
    backtest = BackTest(pdf, fee=fee_bp)
    stats, _ = backtest.run_analysis(return_daily=True, plot=False)
    # get_daily_result() intentionally exposes a narrow API and drops the
    # HighWaterMark/Is_New_High fields required by the project's own plotter.
    # The complete, already-computed daily_result is the plot contract used by
    # the earlier signal study as well.
    daily = backtest.daily_result.copy()
    expected = float(
        (pdf["BidPrice1"] * (pdf["TakerSell_CloseBP"] - fee_bp) / 10.0).sum()
    )
    actual = float(backtest.df["PnL"].sum())
    if not np.isclose(expected, actual, rtol=1e-12, atol=1e-8):
        raise RuntimeError(
            f"BackTest PnL parity failed for {title}: {expected} != {actual}"
        )
    backtest._plot(daily)
    figure = plt.gcf()
    if figure.axes:
        figure.axes[0].set_title(title, fontsize=14, fontweight="bold")
        figure.axes[0].text(
            0.99,
            0.02,
            f"1 lot/event · fixed cost {fee_bp:.1f} bp",
            transform=figure.axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#555555",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=170, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    if path.stat().st_size == 0:
        raise RuntimeError(f"empty BackTest plot: {path}")
    return _json_value(stats), daily, actual


def _artifact_identities(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        output[str(path.relative_to(STUDY_DIR))] = _file_identity(path)
    return output


def run() -> dict[str, Any]:
    if RUN_MANIFEST_PATH.exists():
        raise FileExistsError(
            f"frozen snapshot already exists: {RUN_MANIFEST_PATH}; use a new version"
        )
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["contract"] != "fh_neg_rejection_static_backtest/retrospective_v1":
        raise RuntimeError("unexpected static backtest contract id")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calendar = _read_calendar()
    print(f"[cohort] calendar sessions={len(calendar)} {calendar[0]}-{calendar[-1]}")
    cohort, membership, cohort_counts, membership_paths = build_anchor_cohort(
        calendar
    )
    print(
        "[cohort] "
        f"prelabel={cohort_counts['prelabel_membership_rows']:,} "
        f"labeled={cohort_counts['labeled_anchor_rows']:,}"
    )

    model_input = pl.read_parquet(MODEL_INPUT_PATH).filter(
        (pl.col("Date") >= START) & (pl.col("Date") <= END)
    )
    model_membership_intersection = model_input.select(KEY_COLUMNS).join(
        membership, on=KEY_COLUMNS, how="inner", validate="1:1"
    )
    reconciliation, max_moment_errors = reconcile_parent_moments(
        cohort, membership, calendar, model_membership_intersection
    )
    print(
        "[cohort] parent daily support and label moments matched; "
        f"rounded removals={int(reconciliation['rounded_boundary_removed_n'].sum()):,}"
    )

    legacy_study = _load_legacy_study()
    pre_cap, signals = build_static_signals(cohort, legacy_study)
    full_signal_key_hash = _key_sha256(signals)
    aligned_signal_key_hash = _key_sha256(
        signals.filter(pl.col("Date") >= LEGACY_MODEL_ERA_START)
    )
    print(
        f"[execution] pre_cap={len(pre_cap):,} post_cap={len(signals):,} "
        f"legacy_aligned={len(signals.filter(pl.col('Date') >= LEGACY_MODEL_ERA_START)):,}"
    )

    anchor_keys = cohort.select(KEY_COLUMNS).sort(KEY_COLUMNS)
    signal_columns = tuple(
        column
        for column in (
            "Date",
            "QuoteCode",
            "ChannelSeq",
            "TransTime",
            "BidPrice1",
            "Close",
            "FutureHigh",
            "RefPrice",
            "TickSize",
            "TickSize_legacy_feature",
            "label_exit_price_robustness",
            "label_raw_bp",
            "label_residual_bp",
            "label_z",
            "TakerSell_CloseBP",
            "OTC",
            "MD_L1Rate_30_re",
            "SpreadPairElapsed",
            "RemainSeconds",
            "ToRef",
            "time_bucket",
            "to_ref_bucket",
            "AmountRank_canDayTrade",
            "day_amount_rank",
            "avg_bidLots1",
            "avg_askLots1",
            "hft_strick_makerSpreadBP",
            "accLots",
            "stock_position_wan",
            "lots",
            "signal_variant",
        )
        if column in signals.columns
    )
    kept_signals = signals.select(signal_columns)
    _atomic_parquet(anchor_keys, ANCHOR_KEYS_PATH)
    _atomic_parquet(kept_signals, SIGNALS_PATH)
    _atomic_parquet(reconciliation, RECONCILIATION_PATH)

    period_calendars = _period_calendars(calendar, contract)
    period_rows = [
        _period_summary(
            signals,
            period=period,
            calendar=dates,
            fee_bp=PRIMARY_FEE_BP,
        )
        for period, dates in period_calendars.items()
    ]
    period_summary = pl.DataFrame(period_rows).sort("period")
    _atomic_parquet(period_summary, PERIOD_SUMMARY_PATH)

    full_daily = _daily_frame(signals, calendar, PRIMARY_FEE_BP)
    monthly = _monthly_summary(full_daily)
    _atomic_parquet(full_daily, DAILY_PATH)
    _atomic_parquet(monthly, MONTHLY_SUMMARY_PATH)

    fee_rows = [
        _period_summary(
            signals,
            period=period,
            calendar=dates,
            fee_bp=fee,
        )
        for fee in FEE_GRID_BP
        for period, dates in period_calendars.items()
    ]
    fee_sensitivity = pl.DataFrame(fee_rows).sort(["fee_bp", "period"])
    _atomic_parquet(fee_sensitivity, FEE_SENSITIVITY_PATH)

    full_stats, full_backtest_daily, full_backtest_pnl = _save_project_backtest_plot(
        signals,
        path=FULL_PLOT_PATH,
        title="fh_neg_rejection static anchor · full retrospective",
        fee_bp=PRIMARY_FEE_BP,
    )
    historical_2026_signals = signals.filter(pl.col("Date") >= "20260101")
    (
        historical_2026_stats,
        historical_2026_backtest_daily,
        historical_2026_backtest_pnl,
    ) = _save_project_backtest_plot(
        historical_2026_signals,
        path=HISTORICAL_2026_PLOT_PATH,
        title="fh_neg_rejection static anchor · 2026 historical slice",
        fee_bp=PRIMARY_FEE_BP,
    )

    summary_by_period = {
        str(row["period"]): _json_value(row)
        for row in period_summary.to_dicts()
    }
    full_summary_pnl = float(summary_by_period["full_retrospective"]["net_pnl"])
    historical_2026_summary_pnl = float(
        summary_by_period["year_2026_historical"]["net_pnl"]
    )
    backtest_full_parity = bool(
        np.isclose(full_backtest_pnl, full_summary_pnl, rtol=1e-12, atol=1e-8)
    )
    backtest_2026_parity = bool(
        np.isclose(
            historical_2026_backtest_pnl,
            historical_2026_summary_pnl,
            rtol=1e-12,
            atol=1e-8,
        )
    )
    if not backtest_full_parity or not backtest_2026_parity:
        raise RuntimeError("project BackTest and calendar summary PnL differ")

    decision = {
        "contract": contract["contract"],
        "status": "RESEARCH_SNAPSHOT_ONLY",
        "condition_id": "fh_neg_rejection",
        "selector": "conservative_dgb/d1_g1_b1",
        "selection_mode": "fixed_no_selection",
        "evidence_class": "retrospective_diagnostic",
        "deployment_authorized": False,
        "deployment_handoff_authorized": False,
        "model_gate": "none",
        "primary_fee_bp": PRIMARY_FEE_BP,
        "position_semantics": (
            "one lot per accepted event; chart lower panel is daily summed entry "
            "notional, not concurrent peak exposure"
        ),
        "period_results": summary_by_period,
        "project_backtest_stats": {
            "full_retrospective_active_days_only": full_stats,
            "year_2026_historical_active_days_only": historical_2026_stats,
        },
        "interpretation": [
            "The snapshot confirms the fixed static anchor under the stated idealized execution and cost contract.",
            "The full-window result is a new no-model-gate execution contract; only the 20250513+ slice is directly comparable with the old static-only model era.",
            "Historical profitability does not change the parent RESEARCH_SHADOW_ONLY decision or supply prospective first-live evidence.",
        ],
        "known_limitations": [
            "no portfolio-wide capital or concurrent-exposure cap",
            "no inventory netting, borrow availability, market impact, or latency model",
            "fixed 19.3bp cost is sensitivity-tested but not a fill simulator",
            "condition and execution rules were exposed to the same historical era",
        ],
    }
    _atomic_json(decision, DECISION_PATH)

    checks = {
        "contract_is_retrospective_and_non_deployable": (
            contract["evidence_class"] == "retrospective_diagnostic"
            and contract["promotion_authorized"] is False
            and contract["deployment_handoff_authorized"] is False
        ),
        "prelabel_membership_exact": len(membership)
        == EXPECTED_PRELABEL_MEMBERSHIP,
        "labeled_anchor_support_exact": len(cohort) == EXPECTED_LABELED_SUPPORT,
        "labeled_anchor_key_hash_exact": cohort_counts[
            "labeled_anchor_key_sha256"
        ]
        == EXPECTED_LABELED_KEY_SHA256,
        "parent_daily_support_exact": reconciliation.filter(
            pl.col("condition_n_diff") != 0
        ).is_empty(),
        "parent_daily_label_moments_close": all(
            value <= 1e-6 for value in max_moment_errors.values()
        ),
        "legacy_aligned_signal_rows_exact": len(
            signals.filter(pl.col("Date") >= LEGACY_MODEL_ERA_START)
        )
        == EXPECTED_ALIGNED_SIGNAL_ROWS,
        "legacy_aligned_signal_key_hash_exact": aligned_signal_key_hash
        == EXPECTED_ALIGNED_SIGNAL_KEY_SHA256,
        "accepted_signal_keys_unique": not bool(
            signals.select(pl.struct(KEY_COLUMNS).is_duplicated().any()).item()
        ),
        "full_project_backtest_pnl_parity": backtest_full_parity,
        "historical_2026_project_backtest_pnl_parity": backtest_2026_parity,
        "full_plot_nonempty": FULL_PLOT_PATH.stat().st_size > 0,
        "historical_2026_plot_nonempty": HISTORICAL_2026_PLOT_PATH.stat().st_size
        > 0,
        "fee_grid_contains_primary": PRIMARY_FEE_BP in FEE_GRID_BP,
        "calendar_zero_days_retained": len(full_daily) == len(calendar),
    }
    verification = {
        "contract": "fh_neg_rejection_static_backtest_verification/v1",
        "all_passed": all(checks.values()),
        "checks": checks,
        "counts": {
            **cohort_counts,
            "calendar_sessions": len(calendar),
            "pre_cap_rows": len(pre_cap),
            "accepted_signal_rows_full": len(signals),
            "accepted_signal_rows_legacy_aligned": len(
                signals.filter(pl.col("Date") >= LEGACY_MODEL_ERA_START)
            ),
            "accepted_signal_rows_2026": len(historical_2026_signals),
            "full_zero_signal_days": int((full_daily["trades"] == 0).sum()),
        },
        "key_hashes": {
            "anchor_labeled": cohort_counts["labeled_anchor_key_sha256"],
            "accepted_signals_full": full_signal_key_hash,
            "accepted_signals_legacy_aligned": aligned_signal_key_hash,
        },
        "parent_moment_max_abs_error": max_moment_errors,
        "pnl_parity": {
            "full_backtest": full_backtest_pnl,
            "full_calendar_summary": full_summary_pnl,
            "historical_2026_backtest": historical_2026_backtest_pnl,
            "historical_2026_calendar_summary": historical_2026_summary_pnl,
        },
        "backtest_active_day_rows": {
            "full": len(full_backtest_daily),
            "historical_2026": len(historical_2026_backtest_daily),
        },
    }
    if not verification["all_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"static snapshot verification failed: {failed}")
    _atomic_json(verification, VERIFICATION_PATH)

    membership_identities = _artifact_identities(membership_paths)
    membership_digest = hashlib.sha256()
    for relative, identity in membership_identities.items():
        membership_digest.update(
            f"{relative}\x1f{identity['size']}\x1f{identity['sha256']}\n".encode(
                "utf-8"
            )
        )

    output_paths = (
        ANCHOR_KEYS_PATH,
        SIGNALS_PATH,
        DAILY_PATH,
        RECONCILIATION_PATH,
        PERIOD_SUMMARY_PATH,
        MONTHLY_SUMMARY_PATH,
        FEE_SENSITIVITY_PATH,
        FULL_PLOT_PATH,
        HISTORICAL_2026_PLOT_PATH,
        DECISION_PATH,
        VERIFICATION_PATH,
    )
    source_paths = (
        CONTRACT_PATH,
        Path(__file__),
        LEGACY_STUDY_PATH,
        BACKTEST_SOURCE_PATH,
        LABEL_SOURCE_PATH,
        POOL_SOURCE_PATH,
        MODEL_INPUT_PATH,
        PARENT_DAILY_STATS_PATH,
        PARENT_RUN_MANIFEST_PATH,
        PARENT_DECISION_PATH,
        PARENT_REFERENCE_LINEAGE_PATH,
        CALENDAR_PATH,
    )
    manifest = {
        "contract": contract["contract"],
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": "RESEARCH_SNAPSHOT_ONLY",
        "evidence_class": "retrospective_diagnostic",
        "selection_mode": "fixed_no_selection",
        "deployment_authorized": False,
        "calendar": {
            "start": calendar[0],
            "end": calendar[-1],
            "sessions": len(calendar),
            "zero_signal_sessions_retained": True,
        },
        "selector": {
            "condition_id": "fh_neg_rejection",
            "anchor": "d1_g1_b1",
            "membership_column": "conservative_dgb",
            "parent_contract_sha256": contract["parent_robustness_contract"][
                "contract_sha256"
            ],
            "factorial_design_fingerprint": contract[
                "parent_robustness_contract"
            ]["factorial_design_fingerprint"],
            "labeled_anchor_key_sha256": cohort_counts[
                "labeled_anchor_key_sha256"
            ],
        },
        "execution": contract["execution"],
        "cost": contract["cost"],
        "verification_all_passed": True,
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): _file_identity(path)
            for path in source_paths
        },
        "membership_inputs": {
            "count": len(membership_paths),
            "combined_sha256": membership_digest.hexdigest(),
            "files": membership_identities,
        },
        "outputs": _artifact_identities(output_paths),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "polars": importlib.metadata.version("polars"),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "matplotlib": importlib.metadata.version("matplotlib"),
        },
    }
    _atomic_json(manifest, RUN_MANIFEST_PATH)
    print(
        f"[saved] {OUTPUT_DIR} manifest_sha256={_sha256(RUN_MANIFEST_PATH)}",
        flush=True,
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    decision = run()
    full = decision["period_results"]["full_retrospective"]
    historical_2026 = decision["period_results"]["year_2026_historical"]
    print(
        "[result] full "
        f"trades={full['trades']:,} net_pnl={full['net_pnl']:,.2f} "
        f"net_bp={full['cap_weighted_net_bp']:.4f} "
        f"sharpe={full['calendar_daily_sharpe']:.4f}"
    )
    print(
        "[result] 2026 "
        f"trades={historical_2026['trades']:,} "
        f"net_pnl={historical_2026['net_pnl']:,.2f} "
        f"net_bp={historical_2026['cap_weighted_net_bp']:.4f} "
        f"sharpe={historical_2026['calendar_daily_sharpe']:.4f}"
    )


if __name__ == "__main__":
    main()
