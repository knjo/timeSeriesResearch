"""Train causal FillHigh models, apply prior thresholds, and run BackTest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from src.research.backTest import BackTest
from src.research.timeSeries.framework.labels import (
    add_centered_labels,
    add_directional_labels,
)
from src.research.timeSeries.framework.model import train_and_predict_causal_v2
from src.research.timeSeries.framework.prior_thresholds import (
    PriorQuantileSpec,
    WindowBasis,
    apply_prior_threshold,
    build_prior_session_thresholds,
)
from src.research.timeSeries.framework.profiles.negfill import (
    SHORT_TAKER_STOP108_V1,
)

from study import (
    ARTIFACT_DIR,
    BACKTEST_DIR,
    DEVELOPMENT_END,
    EVENT_DATASET_DIR,
    EVENT_INDEX_PATH,
    FEATURE_MANIFEST,
    HOLDOUT_START,
    MODEL_DIR,
    PROJECT_ROOT,
    REQUIRED_MODEL_INPUT_COLUMNS,
    RIDGE_ALPHA,
    SELECTION_LOOKBACK_DAYS,
    SELECTION_QUANTILE,
    SELECTION_TARGET,
    STUDY_DIR,
    STUDY_END,
    STUDY_START,
    TAKER_FEE_BP,
    THRESHOLD_LOOKBACK_SESSIONS,
    THRESHOLD_MIN_OBSERVATIONS,
    THRESHOLD_MIN_SESSIONS,
    TIMING_LOOKBACK_DAYS,
    TIMING_QUANTILE,
    TIMING_TARGET,
    TRAINING_POSITION_CAP_WAN,
    add_study_features,
    add_threshold_buckets,
    apply_capacity,
    data_roots,
    static_execution_filter,
)


MODEL_INPUT_PATH = MODEL_DIR / "model_input.parquet"
PREDICTION_PATH = MODEL_DIR / "predictions.parquet"
TIMING_THRESHOLD_PATH = MODEL_DIR / "timing_thresholds.parquet"
SELECTION_THRESHOLD_PATH = MODEL_DIR / "selection_thresholds.parquet"
THRESHOLDED_PATH = MODEL_DIR / "thresholded_events.parquet"
VARIANT_SUMMARY_PATH = BACKTEST_DIR / "variant_period_summary.parquet"
MONTHLY_SUMMARY_PATH = BACKTEST_DIR / "monthly_summary.parquet"
FEE_SENSITIVITY_PATH = BACKTEST_DIR / "fee_sensitivity.parquet"
SELECTED_SIGNALS_PATH = BACKTEST_DIR / "selected_signals.parquet"
SELECTED_DAILY_PATH = BACKTEST_DIR / "selected_daily.parquet"
SELECTION_DECISION_PATH = BACKTEST_DIR / "selection_decision.json"
RUN_MANIFEST_PATH = BACKTEST_DIR / "run_manifest.json"

SELECTABLE_VARIANTS = ("timing_only", "selection_only", "dual")
ALL_VARIANTS = ("static_only", *SELECTABLE_VARIANTS)
FEE_GRID_BP = (0.0, 10.0, 19.3, 25.0)

BACKTEST_COLUMNS = (
    "Date",
    "QuoteCode",
    "ChannelSeq",
    "TransTime",
    "BidPrice1",
    "Close",
    "FutureHigh",
    "RefPrice",
    "TickSize",
    "label_raw_bp",
    "label_residual_bp",
    "label_z",
    "label_z_within_stock_day",
    "label_z_within_day",
    "OTC",
    "MD_L1Rate_30_re",
    "SpreadPairElapsed",
    "RemainSeconds",
    "ToRef",
    "AmountRank_canDayTrade",
    "day_amount_rank",
    "avg_bidLots1",
    "avg_askLots1",
    "hft_strick_makerSpreadBP",
)


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
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_source_data() -> pl.DataFrame:
    paths = sorted(
        path
        for path in EVENT_DATASET_DIR.glob("*.parquet")
        if STUDY_START <= path.stem <= STUDY_END
    )
    if not paths:
        raise FileNotFoundError(f"no event dataset partitions: {EVENT_DATASET_DIR}")
    parts: list[pl.DataFrame] = []
    for index, path in enumerate(paths, start=1):
        frame = pl.read_parquet(path, columns=list(REQUIRED_MODEL_INPUT_COLUMNS))
        parts.append(frame)
        if index == 1 or index % 50 == 0:
            print(f"[model-input] read={index}/{len(paths)} date={path.stem}", flush=True)
    return pl.concat(parts, how="vertical_relaxed")


def build_model_input(*, force: bool) -> pl.DataFrame:
    if MODEL_INPUT_PATH.exists() and not force:
        return pl.read_parquet(MODEL_INPUT_PATH)

    raw = _read_source_data()
    featured = add_study_features(raw)
    labeled = add_directional_labels(featured, spec=SHORT_TAKER_STOP108_V1)
    labeled = labeled.filter(
        pl.col("label_valid")
        & pl.col("label_raw_bp").is_finite()
        & (pl.col("label_raw_bp").abs() < 10_000.0)
    )
    labeled = add_centered_labels(labeled)
    FEATURE_MANIFEST.validate_columns(labeled.columns)

    keep = tuple(
        dict.fromkeys(
            (*BACKTEST_COLUMNS, *FEATURE_MANIFEST.ordered_names)
        )
    )
    output = labeled.select(keep).sort(
        ["Date", "QuoteCode", "TransTime", "ChannelSeq"]
    )
    if output.select(pl.struct(["Date", "QuoteCode", "ChannelSeq"]).is_duplicated().any()).item():
        raise ValueError("model input contains duplicate event keys")
    _atomic_parquet(output, MODEL_INPUT_PATH)
    print(f"[model-input] rows={len(output)} saved={MODEL_INPUT_PATH}", flush=True)
    return output


def _save_model_audit(name: str, result: dict[str, object]) -> None:
    audit = {
        "feature_manifest": result["feature_manifest"],
        "feature_names": result["feature_names"],
        "screening_log": result["screening_log"],
        "coeff_log": result["coeff_log"],
        "pls_log": result["pls_log"],
    }
    _atomic_json(_json_value(audit), MODEL_DIR / f"{name}_model_audit.json")


def train_models(
    model_input: pl.DataFrame,
    *,
    n_jobs: int,
    force: bool,
) -> pl.DataFrame:
    if PREDICTION_PATH.exists() and not force:
        return pl.read_parquet(PREDICTION_PATH)

    columns = tuple(
        dict.fromkeys(
            (
                *BACKTEST_COLUMNS,
                *FEATURE_MANIFEST.ordered_names,
            )
        )
    )
    signal_pdf = model_input.select(columns).to_pandas()
    signal_pdf["Date"] = pd.to_datetime(signal_pdf["Date"], format="%Y%m%d")
    common = {
        "n_jobs": n_jobs,
        "feature_screening": False,
        "max_position_per_stock": TRAINING_POSITION_CAP_WAN,
        "position_sample_mode": "hard",
    }
    timing = train_and_predict_causal_v2(
        signal_pdf,
        FEATURE_MANIFEST,
        TIMING_TARGET,
        n_training_days=TIMING_LOOKBACK_DAYS,
        alpha=RIDGE_ALPHA,
        **common,
    )
    _save_model_audit("timing", timing)
    selection = train_and_predict_causal_v2(
        timing["df"],
        FEATURE_MANIFEST,
        SELECTION_TARGET,
        n_training_days=SELECTION_LOOKBACK_DAYS,
        alpha=RIDGE_ALPHA,
        **common,
    )
    _save_model_audit("selection", selection)

    timing_pred = f"{TIMING_TARGET}_pred"
    selection_pred = f"{SELECTION_TARGET}_pred"
    output_columns = tuple(
        dict.fromkeys(
            (*BACKTEST_COLUMNS, timing_pred, selection_pred)
        )
    )
    output = pl.from_pandas(selection["df"][list(output_columns)]).with_columns(
        pl.col("Date").dt.strftime("%Y%m%d")
    )
    output = output.sort(["Date", "QuoteCode", "TransTime", "ChannelSeq"])
    _atomic_parquet(output, PREDICTION_PATH)
    print(f"[model] predictions saved rows={len(output)}", flush=True)
    return output


def _calendar_sessions() -> list[str]:
    roots = data_roots()
    return sorted(
        path.name[:8]
        for path in roots.tick_data_dir.glob("*_StockTick.parquet")
        if STUDY_START <= path.name[:8] <= STUDY_END
    )


def apply_thresholds(predictions: pl.DataFrame, *, force: bool) -> pl.DataFrame:
    if THRESHOLDED_PATH.exists() and not force:
        return pl.read_parquet(THRESHOLDED_PATH)

    timing_pred = f"{TIMING_TARGET}_pred"
    selection_pred = f"{SELECTION_TARGET}_pred"
    bucketed = add_threshold_buckets(predictions)
    sessions = _calendar_sessions()
    event_dates = bucketed["Date"].unique().sort().to_list()
    if len(event_dates) <= TIMING_LOOKBACK_DAYS:
        raise RuntimeError("event calendar is shorter than the timing warmup")
    # The rolling engine treats n_training_days as a maximum window and emits
    # predictions as early as event-date two.  Require 60 prior event-active
    # dates before a score may enter thresholding or execution.  This differs
    # from a market calendar when an event dataset partition is empty.
    model_warmup_start = event_dates[TIMING_LOOKBACK_DAYS]
    threshold_scores = bucketed.filter(pl.col("Date") >= model_warmup_start)
    bucket_cols = ("time_bucket", "to_ref_bucket")
    timing_spec = PriorQuantileSpec(
        score_col=timing_pred,
        threshold_col="timing_threshold",
        q=TIMING_QUANTILE,
        bucket_cols=bucket_cols,
        lookback_sessions=THRESHOLD_LOOKBACK_SESSIONS,
        min_sessions=THRESHOLD_MIN_SESSIONS,
        min_observations=THRESHOLD_MIN_OBSERVATIONS,
        window_basis=WindowBasis.CALENDAR_SESSIONS,
        score_unit="residual_z_centered_model_score",
    )
    selection_spec = PriorQuantileSpec(
        score_col=selection_pred,
        threshold_col="selection_threshold",
        q=SELECTION_QUANTILE,
        bucket_cols=bucket_cols,
        lookback_sessions=THRESHOLD_LOOKBACK_SESSIONS,
        min_sessions=THRESHOLD_MIN_SESSIONS,
        min_observations=THRESHOLD_MIN_OBSERVATIONS,
        window_basis=WindowBasis.CALENDAR_SESSIONS,
        score_unit="residual_z_centered_model_score",
    )
    timing_thresholds = build_prior_session_thresholds(
        threshold_scores,
        sessions=sessions,
        spec=timing_spec,
    )
    selection_thresholds = build_prior_session_thresholds(
        threshold_scores,
        sessions=sessions,
        spec=selection_spec,
    )
    _atomic_parquet(timing_thresholds, TIMING_THRESHOLD_PATH)
    _atomic_parquet(selection_thresholds, SELECTION_THRESHOLD_PATH)

    timing_join = timing_thresholds.select(
        "Date", *bucket_cols, "timing_threshold"
    )
    selection_join = selection_thresholds.select(
        "Date", *bucket_cols, "selection_threshold"
    )
    output = apply_prior_threshold(
        bucketed,
        timing_join,
        score_col=timing_pred,
        threshold_col="timing_threshold",
        bucket_cols=bucket_cols,
        output_col="passes_timing",
    )
    output = apply_prior_threshold(
        output,
        selection_join,
        score_col=selection_pred,
        threshold_col="selection_threshold",
        bucket_cols=bucket_cols,
        output_col="passes_selection",
    )
    output = output.with_columns(
        (
            pl.col(timing_pred).is_finite()
            & pl.col(selection_pred).is_finite()
            & (pl.col("Date") >= model_warmup_start)
        ).alias("model_era"),
        (pl.col("passes_timing") & pl.col("passes_selection")).alias(
            "passes_dual"
        ),
    )
    _atomic_parquet(output, THRESHOLDED_PATH)
    return output


def build_variant_signals(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    gates = {
        "static_only": pl.col("model_era"),
        "timing_only": pl.col("model_era") & pl.col("passes_timing"),
        "selection_only": pl.col("model_era") & pl.col("passes_selection"),
        "dual": pl.col("model_era") & pl.col("passes_dual"),
    }
    output: dict[str, pl.DataFrame] = {}
    for name, gate in gates.items():
        gated = static_execution_filter(frame.filter(gate.fill_null(False)))
        signal = apply_capacity(gated).with_columns(
            pl.lit(name).alias("signal_variant"),
            pl.col("label_raw_bp").alias("TakerSell_CloseBP"),
        )
        output[name] = signal
        print(
            f"[signal] variant={name} pre_cap={len(gated)} post_cap={len(signal)}",
            flush=True,
        )
    return output


def _period_calendar(
    sessions: list[str], period: str, model_start: str
) -> list[str]:
    if period == "development":
        return [date for date in sessions if model_start <= date <= DEVELOPMENT_END]
    if period == "holdout_2026_ytd":
        return [date for date in sessions if HOLDOUT_START <= date <= STUDY_END]
    if period == "full_model_era":
        return [date for date in sessions if model_start <= date <= STUDY_END]
    raise ValueError(period)


def _summary_row(
    signals: pl.DataFrame,
    *,
    variant: str,
    period: str,
    calendar: list[str],
    fee_bp: float,
) -> dict[str, object]:
    subset = signals.filter(pl.col("Date").is_in(calendar))
    daily = (
        subset.with_columns(
            (pl.col("BidPrice1") * (pl.col("label_raw_bp") - fee_bp) / 10.0)
            .alias("net_pnl"),
            (pl.col("BidPrice1") * pl.col("label_raw_bp") / 10.0).alias(
                "gross_pnl"
            ),
            (
                pl.col("BidPrice1")
                * (pl.col("label_residual_bp") - fee_bp)
                / 10.0
            ).alias("residual_net_pnl"),
            (pl.col("BidPrice1") * pl.col("label_residual_bp") / 10.0).alias(
                "residual_gross_pnl"
            ),
            (pl.col("BidPrice1") * 1000.0).alias("position"),
        )
        .group_by("Date")
        .agg(
            pl.col("net_pnl").sum(),
            pl.col("gross_pnl").sum(),
            pl.col("residual_net_pnl").sum(),
            pl.col("residual_gross_pnl").sum(),
            pl.col("position").sum(),
            pl.len().alias("trades"),
            pl.col("QuoteCode").n_unique().alias("stocks"),
        )
    )
    calendar_frame = pl.DataFrame({"Date": calendar})
    daily = calendar_frame.join(daily, on="Date", how="left").with_columns(
        pl.col("net_pnl").fill_null(0.0),
        pl.col("gross_pnl").fill_null(0.0),
        pl.col("residual_net_pnl").fill_null(0.0),
        pl.col("residual_gross_pnl").fill_null(0.0),
        pl.col("position").fill_null(0.0),
        pl.col("trades").fill_null(0),
        pl.col("stocks").fill_null(0),
    )
    pnl = daily["net_pnl"].to_numpy()
    residual_pnl_values = daily["residual_net_pnl"].to_numpy()
    sharpe = (
        float(np.mean(pnl) / np.std(pnl, ddof=1) * np.sqrt(252.0))
        if len(pnl) > 1 and np.std(pnl, ddof=1) > 0.0
        else math.nan
    )
    cumulative = np.cumsum(pnl)
    drawdown = cumulative - np.maximum.accumulate(cumulative) if len(pnl) else np.array([])
    residual_sharpe = (
        float(
            np.mean(residual_pnl_values)
            / np.std(residual_pnl_values, ddof=1)
            * np.sqrt(252.0)
        )
        if len(residual_pnl_values) > 1
        and np.std(residual_pnl_values, ddof=1) > 0.0
        else math.nan
    )
    net_pnl = float(daily["net_pnl"].sum())
    gross_pnl = float(daily["gross_pnl"].sum())
    residual_net_pnl = float(daily["residual_net_pnl"].sum())
    residual_gross_pnl = float(daily["residual_gross_pnl"].sum())
    position = float(daily["position"].sum())
    return {
        "signal_variant": variant,
        "period": period,
        "fee_bp": fee_bp,
        "start": calendar[0] if calendar else None,
        "end": calendar[-1] if calendar else None,
        "calendar_days": len(calendar),
        "active_days": int((daily["trades"] > 0).sum()),
        "trades": int(daily["trades"].sum()),
        "stock_days": int(
            subset.select(pl.struct(["Date", "QuoteCode"]).n_unique()).item()
        )
        if len(subset)
        else 0,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "cost_pnl": gross_pnl - net_pnl,
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
        "daily_sharpe": sharpe,
        "residual_daily_sharpe": residual_sharpe,
        "daily_win_rate": float(np.mean(pnl > 0.0)) if len(pnl) else math.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else math.nan,
        "mean_daily_position": float(daily["position"].mean())
        if len(daily)
        else math.nan,
    }


def _monthly_rows(
    signals: pl.DataFrame, *, variant: str, fee_bp: float
) -> list[dict[str, object]]:
    return (
        signals.with_columns(
            pl.col("Date").str.slice(0, 6).alias("month"),
            (pl.col("BidPrice1") * (pl.col("label_raw_bp") - fee_bp) / 10.0)
            .alias("net_pnl"),
            (
                pl.col("BidPrice1")
                * (pl.col("label_residual_bp") - fee_bp)
                / 10.0
            ).alias("residual_net_pnl"),
            (pl.col("BidPrice1") * 1000.0).alias("position"),
        )
        .group_by("month")
        .agg(
            pl.col("net_pnl").sum(),
            pl.col("residual_net_pnl").sum(),
            pl.col("position").sum(),
            pl.len().alias("trades"),
            pl.col("Date").n_unique().alias("active_days"),
        )
        .with_columns(
            pl.lit(variant).alias("signal_variant"),
            pl.lit(fee_bp).alias("fee_bp"),
            (pl.col("net_pnl") / pl.col("position") * 10_000.0).alias(
                "cap_weighted_net_bp"
            ),
            (
                pl.col("residual_net_pnl") / pl.col("position") * 10_000.0
            ).alias("cap_weighted_residual_net_bp"),
        )
        .sort("month")
        .to_dicts()
    )


def _select_variant(
    summary: pl.DataFrame, *, model_start: str
) -> dict[str, object]:
    development = summary.filter(
        (pl.col("period") == "development")
        & pl.col("signal_variant").is_in(SELECTABLE_VARIANTS)
        & (pl.col("active_days") >= 60)
        & (pl.col("trades") >= 1000)
        & pl.col("daily_sharpe").is_finite()
    )
    if development.is_empty():
        raise RuntimeError(
            "no model-gated variant meets the frozen development support rule"
        )
    order = {name: index for index, name in enumerate(SELECTABLE_VARIANTS)}
    rows = development.to_dicts()
    rows.sort(
        key=lambda row: (
            -float(row["daily_sharpe"]),
            order[str(row["signal_variant"])],
        )
    )
    selected = str(rows[0]["signal_variant"])
    return {
        "selected_variant": selected,
        "selection_period": f"{model_start}-{DEVELOPMENT_END}",
        "selection_metric": "calendar-filled net daily Sharpe at 19.3bp",
        "minimum_active_days": 60,
        "minimum_trades": 1000,
        "eligible_development_rows": rows,
        "holdout_not_used_for_selection": True,
    }


def _run_backtest(signals: pl.DataFrame, *, fee_bp: float):
    if signals.is_empty():
        raise ValueError("BackTest requires at least one selected signal")
    pdf = signals.to_pandas()
    backtest = BackTest(pdf, fee=fee_bp)
    stats, _ = backtest.run_analysis(return_daily=True, plot=False)
    expected = (
        pdf["BidPrice1"] * (pdf["TakerSell_CloseBP"] - fee_bp) / 10.0
    ).sum()
    actual = float(backtest.df["PnL"].sum())
    if not np.isclose(expected, actual, rtol=1e-12, atol=1e-8):
        raise RuntimeError(
            f"BackTest PnL parity failed: expected={expected} actual={actual}"
        )
    return backtest, _json_value(stats), backtest.daily_result.copy()


def _save_exact_plot(
    signals: pl.DataFrame,
    *,
    path: Path,
    fee_bp: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    backtest, stats, daily = _run_backtest(signals, fee_bp=fee_bp)
    backtest._plot(daily)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(path, dpi=160, bbox_inches="tight")
    plt.close(plt.gcf())
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"BackTest plot was not saved: {path}")
    return stats, daily


def evaluate(
    thresholded: pl.DataFrame,
) -> tuple[dict[str, pl.DataFrame], dict[str, object]]:
    signals = build_variant_signals(thresholded)
    sessions = _calendar_sessions()
    model_dates = (
        thresholded.filter(pl.col("model_era"))["Date"].unique().sort().to_list()
    )
    if not model_dates:
        raise RuntimeError("rolling models produced no common model-era date")
    model_start = model_dates[0]
    periods = ("development", "holdout_2026_ytd", "full_model_era")
    summary_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for name in ALL_VARIANTS:
        for period in periods:
            summary_rows.append(
                _summary_row(
                    signals[name],
                    variant=name,
                    period=period,
                    calendar=_period_calendar(sessions, period, model_start),
                    fee_bp=TAKER_FEE_BP,
                )
            )
        monthly_rows.extend(
            _monthly_rows(signals[name], variant=name, fee_bp=TAKER_FEE_BP)
        )
    summary = pl.DataFrame(summary_rows)
    monthly = pl.DataFrame(monthly_rows)
    _atomic_parquet(summary, VARIANT_SUMMARY_PATH)
    _atomic_parquet(monthly, MONTHLY_SUMMARY_PATH)

    decision = _select_variant(summary, model_start=model_start)
    selected_name = str(decision["selected_variant"])
    selected = signals[selected_name]
    _atomic_parquet(selected, SELECTED_SIGNALS_PATH)

    fee_rows: list[dict[str, object]] = []
    for fee in FEE_GRID_BP:
        for period in periods:
            fee_rows.append(
                _summary_row(
                    selected,
                    variant=selected_name,
                    period=period,
                    calendar=_period_calendar(sessions, period, model_start),
                    fee_bp=fee,
                )
            )
    _atomic_parquet(pl.DataFrame(fee_rows), FEE_SENSITIVITY_PATH)

    full_stats, full_daily = _save_exact_plot(
        selected,
        path=BACKTEST_DIR / "backtest_full.png",
        fee_bp=TAKER_FEE_BP,
    )
    holdout = selected.filter(pl.col("Date") >= HOLDOUT_START)
    holdout_stats, _ = _save_exact_plot(
        holdout,
        path=BACKTEST_DIR / "backtest_2026_holdout.png",
        fee_bp=TAKER_FEE_BP,
    )
    daily_frame = pl.from_pandas(full_daily.reset_index())
    _atomic_parquet(daily_frame, SELECTED_DAILY_PATH)
    decision.update(
        {
            "full_backtest_stats": full_stats,
            "holdout_backtest_stats": holdout_stats,
            "selected_signal_rows": len(selected),
            "selected_signal_days": selected["Date"].n_unique(),
        }
    )
    _atomic_json(_json_value(decision), SELECTION_DECISION_PATH)
    return signals, decision


def write_run_manifest(decision: dict[str, object], args: argparse.Namespace) -> None:
    event_manifest = EVENT_INDEX_PATH.with_suffix(".manifest.json")
    dataset_manifest = EVENT_DATASET_DIR / "_dataset_manifest.json"
    artifact_paths = (
        STUDY_DIR / "study_spec.json",
        STUDY_DIR / "study.py",
        STUDY_DIR / "build_dataset.py",
        STUDY_DIR / "run_model_backtest.py",
        STUDY_DIR / "audit_sampling.py",
        STUDY_DIR / "audit_negfill_overlap.py",
        STUDY_DIR / "test_study.py",
        STUDY_DIR / "verify_run.py",
        event_manifest,
        EVENT_INDEX_PATH,
        dataset_manifest,
        *sorted(EVENT_DATASET_DIR.glob("*.parquet")),
        MODEL_INPUT_PATH,
        PREDICTION_PATH,
        MODEL_DIR / "timing_model_audit.json",
        MODEL_DIR / "selection_model_audit.json",
        TIMING_THRESHOLD_PATH,
        SELECTION_THRESHOLD_PATH,
        THRESHOLDED_PATH,
        VARIANT_SUMMARY_PATH,
        MONTHLY_SUMMARY_PATH,
        FEE_SENSITIVITY_PATH,
        SELECTED_SIGNALS_PATH,
        SELECTED_DAILY_PATH,
        SELECTION_DECISION_PATH,
        BACKTEST_DIR / "backtest_full.png",
        BACKTEST_DIR / "backtest_2026_holdout.png",
        ARTIFACT_DIR / "sampling_audit_v1" / "audit.json",
        ARTIFACT_DIR / "sampling_audit_v1" / "daily_audit.parquet",
        ARTIFACT_DIR / "negfill_overlap_v1" / "overlap.json",
        ARTIFACT_DIR / "negfill_overlap_v1" / "period_summary.parquet",
    )
    source_paths = (
        PROJECT_ROOT / "src/research/timeSeries/framework/events.py",
        PROJECT_ROOT / "src/research/timeSeries/framework/dataset.py",
        PROJECT_ROOT / "src/research/timeSeries/framework/features.py",
        PROJECT_ROOT / "src/research/timeSeries/framework/labels.py",
        PROJECT_ROOT / "src/research/timeSeries/framework/model.py",
        PROJECT_ROOT / "src/research/timeSeries/framework/prior_thresholds.py",
        PROJECT_ROOT
        / "src/research/timeSeries/framework/profiles/filllots_at_high.py",
        PROJECT_ROOT / "src/research/timeSeries/framework/profiles/negfill.py",
        PROJECT_ROOT / "src/research/rolling_model_multiLabel.py",
        PROJECT_ROOT / "src/research/backTest.py",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
    )
    package_versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "polars", "scikit-learn", "matplotlib"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    payload = {
        "contract": "filllots_at_high_short_signal_run/v1",
        "study_start": STUDY_START,
        "study_end": STUDY_END,
        "development_end": DEVELOPMENT_END,
        "holdout_start": HOLDOUT_START,
        "direction": "short",
        "candidate_id": "fh_neg_rejection",
        "label_profile_id": SHORT_TAKER_STOP108_V1.profile_id,
        "label_profile_hash": SHORT_TAKER_STOP108_V1.fingerprint,
        "feature_manifest": FEATURE_MANIFEST.canonical_dict(),
        "feature_manifest_hash": FEATURE_MANIFEST.fingerprint,
        "model": {
            "preprocessing_mode": "causal_v2",
            "feature_screening": False,
            "timing_lookback_days": TIMING_LOOKBACK_DAYS,
            "minimum_prior_event_dates_before_evaluation": TIMING_LOOKBACK_DAYS,
            "selection_lookback_days": SELECTION_LOOKBACK_DAYS,
            "ridge_alpha": RIDGE_ALPHA,
            "training_position_cap_wan": TRAINING_POSITION_CAP_WAN,
            "n_jobs": args.n_jobs,
        },
        "threshold_window_basis": WindowBasis.CALENDAR_SESSIONS.value,
        "taker_fee_bp": TAKER_FEE_BP,
        "raw_source_identity_contract": "path_size_mtime_ns",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions,
        },
        "selection_decision": decision,
        "artifacts": {
            str(path.relative_to(STUDY_DIR)): {
                "size": path.stat().st_size if path.exists() else None,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        },
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): {
                "size": path.stat().st_size if path.exists() else None,
                "sha256": _sha256(path),
            }
            for path in source_paths
        },
    }
    _atomic_json(_json_value(payload), RUN_MANIFEST_PATH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--force-input", action="store_true")
    parser.add_argument("--force-model", action="store_true")
    parser.add_argument("--force-thresholds", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="refresh hashes after post-backtest audits without recomputing outputs",
    )
    args = parser.parse_args()
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be zero")
    if args.manifest_only:
        if args.force_input or args.force_model or args.force_thresholds:
            parser.error("--manifest-only cannot be combined with force flags")
        if not SELECTION_DECISION_PATH.exists():
            raise FileNotFoundError(SELECTION_DECISION_PATH)
        decision = json.loads(SELECTION_DECISION_PATH.read_text(encoding="utf-8"))
        write_run_manifest(decision, args)
        print(f"[manifest] refreshed={RUN_MANIFEST_PATH}", flush=True)
        return 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    model_input = build_model_input(force=args.force_input)
    predictions = train_models(
        model_input, n_jobs=args.n_jobs, force=args.force_model
    )
    thresholded = apply_thresholds(
        predictions, force=args.force_thresholds or args.force_model
    )
    _, decision = evaluate(thresholded)
    write_run_manifest(decision, args)
    print(json.dumps(_json_value(decision), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
