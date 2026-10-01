"""Train causal models on fill-gated fh_pos_breakout maker-long events, threshold, BackTest.

Pipeline (all prior-only):

1. model input = event dataset + maker fill seconds + study features + defensive
   features + ``fhl.long_maker_tp109.v1`` labels; rows whose Bid1 order would not
   fill within ``FILL_HORIZON_SECONDS`` are dropped (no trade, no training row);
2. rolling Ridge timing (60d, within-stock-day z) and selection (20d, within-day z)
   models via ``train_and_predict_causal_v2``;
3. 16-cell prior-session quantile thresholds (frozen q from the short line);
4. eight execution variants = {static_only, timing_only, selection_only, dual}
   x universe {all, top_list}; each goes through the shared static gate, per-stock
   capacity, then ``BackTest(side='long', entry_price_col='BidPrice1',
   bp_col='label_raw_bp')`` at 19.3bp with no hedge;
5. variant selection on development only, holdout reported; fill-horizon
   sensitivity (60/300/600s) re-gates execution rows without retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.research.backTest import BackTest  # noqa: E402
from src.research.intradaySignal.framework.labels import (  # noqa: E402
    add_centered_labels,
    add_directional_labels,
)
from src.research.intradaySignal.framework.model import train_and_predict_causal_v2  # noqa: E402
from src.research.intradaySignal.framework.prior_thresholds import (  # noqa: E402
    PriorQuantileSpec,
    WindowBasis,
    apply_prior_threshold,
    build_prior_session_thresholds,
)

from study import (  # noqa: E402
    ARTIFACT_DIR,
    BACKTEST_DIR,
    DEVELOPMENT_END,
    ENTRY_PRICE_COL,
    EVENT_DATASET_DIR,
    EVENT_INDEX_PATH,
    FEATURE_MANIFEST,
    FEE_BP,
    FILL_DIR,
    FILL_HORIZON_SECONDS,
    FILL_HORIZON_SENSITIVITY,
    HOLDOUT_START,
    LABEL_SPEC,
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
    THRESHOLD_LOOKBACK_SESSIONS,
    THRESHOLD_MIN_OBSERVATIONS,
    THRESHOLD_MIN_SESSIONS,
    TIMING_LOOKBACK_DAYS,
    TIMING_QUANTILE,
    TIMING_TARGET,
    TRAINING_POSITION_CAP_WAN,
    add_defensive_features,
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
FILL_SENSITIVITY_PATH = BACKTEST_DIR / "fill_horizon_sensitivity.parquet"
SELECTED_SIGNALS_PATH = BACKTEST_DIR / "selected_signals.parquet"
SELECTED_DAILY_PATH = BACKTEST_DIR / "selected_daily.parquet"
SELECTION_DECISION_PATH = BACKTEST_DIR / "selection_decision.json"
RUN_MANIFEST_PATH = BACKTEST_DIR / "run_manifest.json"

GATES = ("static_only", "timing_only", "selection_only", "dual")
UNIVERSES = ("all", "top_list")
ALL_VARIANTS = tuple(f"{g}__{u}" for g in GATES for u in UNIVERSES)
SELECTABLE_VARIANTS = tuple(v for v in ALL_VARIANTS if not v.startswith("static_only"))
FEE_GRID_BP = (0.0, 10.0, 19.3, 25.0)
MIN_ACTIVE_DAYS = 60
MIN_TRADES = 200
PERIODS = ("development", "holdout_2026_ytd", "full_model_era")

BACKTEST_COLUMNS = (
    "Date",
    "QuoteCode",
    "ChannelSeq",
    "TransTime",
    "BidPrice1",
    "AskPrice1",
    "Close",
    "FutureHigh",
    "RefPrice",
    "TickSize",
    "label_raw_bp",
    "label_residual_bp",
    "label_z",
    "label_z_within_stock_day",
    "label_z_within_day",
    "maker_buy_fill_seconds",
    "maker_fill_branch",
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
    "in_top_list",
    "defensive_weight",
)


# --------------------------------------------------------------------------- #
def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False, encoding="utf-8")
    with handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    Path(handle.name).replace(path)


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    handle.close()
    frame.write_parquet(handle.name)
    Path(handle.name).replace(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return _json_value(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
def _read_source_data() -> pl.DataFrame:
    paths = sorted(p for p in EVENT_DATASET_DIR.glob("*.parquet") if STUDY_START <= p.stem <= STUDY_END)
    if not paths:
        raise FileNotFoundError(f"no event dataset partitions: {EVENT_DATASET_DIR}")
    parts: list[pl.DataFrame] = []
    for index, path in enumerate(paths, start=1):
        frame = pl.read_parquet(path, columns=list(REQUIRED_MODEL_INPUT_COLUMNS))
        fill_path = FILL_DIR / path.name
        if not fill_path.exists():
            raise FileNotFoundError(f"missing maker fill partition: {fill_path}")
        fills = pl.read_parquet(fill_path, columns=["Date", "QuoteCode", "ChannelSeq", "maker_buy_fill_seconds", "maker_fill_branch"])
        joined = frame.join(fills, on=["Date", "QuoteCode", "ChannelSeq"], how="left")
        if joined.height != frame.height:
            raise ValueError(f"{path.stem}: fill join changed row count")
        parts.append(joined)
        if index == 1 or index % 50 == 0:
            print(f"[model-input] read={index}/{len(paths)} date={path.stem}", flush=True)
    return pl.concat(parts, how="vertical_relaxed")


def build_model_input(*, force: bool) -> pl.DataFrame:
    if MODEL_INPUT_PATH.exists() and not force:
        return pl.read_parquet(MODEL_INPUT_PATH)

    raw = _read_source_data()
    total_rows = raw.height
    featured = add_defensive_features(add_study_features(raw))
    labeled = add_directional_labels(featured, spec=LABEL_SPEC)
    labeled = labeled.filter(
        pl.col("label_valid") & pl.col("label_raw_bp").is_finite() & (pl.col("label_raw_bp").abs() < 10_000.0)
    )
    valid_rows = labeled.height
    # maker fill gate: the training and execution population are the same
    labeled = labeled.filter(
        pl.col("maker_buy_fill_seconds").is_not_null() & (pl.col("maker_buy_fill_seconds") < FILL_HORIZON_SECONDS)
    )
    fill_rows = labeled.height
    labeled = add_centered_labels(labeled)
    FEATURE_MANIFEST.validate_columns(labeled.columns)
    keep = tuple(dict.fromkeys((*BACKTEST_COLUMNS, *FEATURE_MANIFEST.ordered_names)))
    output = labeled.select(keep).sort(["Date", "QuoteCode", "TransTime", "ChannelSeq"])
    if output.select(pl.struct(["Date", "QuoteCode", "ChannelSeq"]).is_duplicated().any()).item():
        raise ValueError("model input contains duplicate event keys")
    _atomic_parquet(output, MODEL_INPUT_PATH)
    _atomic_json(
        {
            "event_rows": total_rows,
            "label_valid_rows": valid_rows,
            "fill_gated_rows": fill_rows,
            "fill_horizon_seconds": FILL_HORIZON_SECONDS,
            "fill_pass_rate": fill_rows / valid_rows if valid_rows else None,
            "top_list_share": float(output["in_top_list"].mean()) if output.height else None,
            "fill_branch_share_after_gate": output.group_by("maker_fill_branch").len().with_columns((pl.col("len") / output.height).alias("share")).to_dicts(),
        },
        MODEL_DIR / "model_input_audit.json",
    )
    print(
        f"[model-input] rows={total_rows} label_valid={valid_rows} fill<{FILL_HORIZON_SECONDS:.0f}s={fill_rows} saved={MODEL_INPUT_PATH}",
        flush=True,
    )
    return output


def _save_model_audit(name: str, result: dict[str, object]) -> None:
    audit = {k: result.get(k) for k in ("feature_manifest", "feature_names", "screening_log", "coeff_log", "pls_log")}
    _atomic_json(_json_value(audit), MODEL_DIR / f"{name}_model_audit.json")


def train_models(model_input: pl.DataFrame, *, n_jobs: int, force: bool) -> pl.DataFrame:
    if PREDICTION_PATH.exists() and not force:
        return pl.read_parquet(PREDICTION_PATH)

    columns = tuple(dict.fromkeys((*BACKTEST_COLUMNS, *FEATURE_MANIFEST.ordered_names)))
    signal_pdf = model_input.select(columns).to_pandas()
    signal_pdf["Date"] = pd.to_datetime(signal_pdf["Date"], format="%Y%m%d")
    common = {
        "n_jobs": n_jobs,
        "feature_screening": False,
        "max_position_per_stock": TRAINING_POSITION_CAP_WAN,
        "position_sample_mode": "hard",
    }
    timing = train_and_predict_causal_v2(
        signal_pdf, FEATURE_MANIFEST, TIMING_TARGET, n_training_days=TIMING_LOOKBACK_DAYS, alpha=RIDGE_ALPHA, **common
    )
    _save_model_audit("timing", timing)
    selection = train_and_predict_causal_v2(
        timing["df"], FEATURE_MANIFEST, SELECTION_TARGET, n_training_days=SELECTION_LOOKBACK_DAYS, alpha=RIDGE_ALPHA, **common
    )
    _save_model_audit("selection", selection)

    timing_pred = f"{TIMING_TARGET}_pred"
    selection_pred = f"{SELECTION_TARGET}_pred"
    output_columns = tuple(dict.fromkeys((*BACKTEST_COLUMNS, timing_pred, selection_pred)))
    output = pl.from_pandas(selection["df"][list(output_columns)]).with_columns(pl.col("Date").dt.strftime("%Y%m%d"))
    output = output.sort(["Date", "QuoteCode", "TransTime", "ChannelSeq"])
    _atomic_parquet(output, PREDICTION_PATH)
    print(f"[model] predictions saved rows={len(output)}", flush=True)
    return output


def _calendar_sessions() -> list[str]:
    roots = data_roots()
    return sorted(p.name[:8] for p in roots.tick_data_dir.glob("*_StockTick.parquet") if STUDY_START <= p.name[:8] <= STUDY_END)


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
    model_warmup_start = event_dates[TIMING_LOOKBACK_DAYS]
    threshold_scores = bucketed.filter(pl.col("Date") >= model_warmup_start)
    bucket_cols = ("time_bucket", "to_ref_bucket")
    specs = {
        "timing": PriorQuantileSpec(
            score_col=timing_pred, threshold_col="timing_threshold", q=TIMING_QUANTILE, bucket_cols=bucket_cols,
            lookback_sessions=THRESHOLD_LOOKBACK_SESSIONS, min_sessions=THRESHOLD_MIN_SESSIONS,
            min_observations=THRESHOLD_MIN_OBSERVATIONS, window_basis=WindowBasis.CALENDAR_SESSIONS,
            score_unit="residual_z_centered_model_score",
        ),
        "selection": PriorQuantileSpec(
            score_col=selection_pred, threshold_col="selection_threshold", q=SELECTION_QUANTILE, bucket_cols=bucket_cols,
            lookback_sessions=THRESHOLD_LOOKBACK_SESSIONS, min_sessions=THRESHOLD_MIN_SESSIONS,
            min_observations=THRESHOLD_MIN_OBSERVATIONS, window_basis=WindowBasis.CALENDAR_SESSIONS,
            score_unit="residual_z_centered_model_score",
        ),
    }
    timing_thresholds = build_prior_session_thresholds(threshold_scores, sessions=sessions, spec=specs["timing"])
    selection_thresholds = build_prior_session_thresholds(threshold_scores, sessions=sessions, spec=specs["selection"])
    _atomic_parquet(timing_thresholds, TIMING_THRESHOLD_PATH)
    _atomic_parquet(selection_thresholds, SELECTION_THRESHOLD_PATH)

    output = apply_prior_threshold(
        bucketed, timing_thresholds.select("Date", *bucket_cols, "timing_threshold"),
        score_col=timing_pred, threshold_col="timing_threshold", bucket_cols=bucket_cols, output_col="passes_timing",
    )
    output = apply_prior_threshold(
        output, selection_thresholds.select("Date", *bucket_cols, "selection_threshold"),
        score_col=selection_pred, threshold_col="selection_threshold", bucket_cols=bucket_cols, output_col="passes_selection",
    )
    output = output.with_columns(
        (pl.col(timing_pred).is_finite() & pl.col(selection_pred).is_finite() & (pl.col("Date") >= model_warmup_start)).alias("model_era"),
        (pl.col("passes_timing") & pl.col("passes_selection")).alias("passes_dual"),
    )
    _atomic_parquet(output, THRESHOLDED_PATH)
    return output


# --------------------------------------------------------------------------- #
def _gate_expr(gate: str) -> pl.Expr:
    if gate == "static_only":
        return pl.col("model_era")
    if gate == "timing_only":
        return pl.col("model_era") & pl.col("passes_timing")
    if gate == "selection_only":
        return pl.col("model_era") & pl.col("passes_selection")
    if gate == "dual":
        return pl.col("model_era") & pl.col("passes_dual")
    raise ValueError(gate)


def _universe_expr(universe: str) -> pl.Expr:
    if universe == "all":
        return pl.lit(True)
    if universe == "top_list":
        return pl.col("in_top_list") > 0.5
    raise ValueError(universe)


def build_variant_signals(frame: pl.DataFrame, *, fill_horizon: float = FILL_HORIZON_SECONDS) -> dict[str, pl.DataFrame]:
    output: dict[str, pl.DataFrame] = {}
    fill_gate = pl.col("maker_buy_fill_seconds") < fill_horizon
    for name in ALL_VARIANTS:
        gate, universe = name.split("__")
        mask = (_gate_expr(gate) & _universe_expr(universe) & fill_gate).fill_null(False)
        gated = static_execution_filter(frame.filter(mask))
        signal = apply_capacity(gated).with_columns(
            pl.lit(name).alias("signal_variant"),
            pl.col("label_raw_bp").alias("MakerBuy_CloseBP"),
        )
        output[name] = signal
        print(f"[signal] variant={name} fill<{fill_horizon:.0f}s pre_cap={len(gated)} post_cap={len(signal)}", flush=True)
    return output


def _period_calendar(sessions: list[str], period: str, model_start: str) -> list[str]:
    if period == "development":
        return [d for d in sessions if model_start <= d <= DEVELOPMENT_END]
    if period == "holdout_2026_ytd":
        return [d for d in sessions if HOLDOUT_START <= d <= STUDY_END]
    if period == "full_model_era":
        return [d for d in sessions if model_start <= d <= STUDY_END]
    raise ValueError(period)


def _daily_frame(signals: pl.DataFrame, *, calendar: list[str], fee_bp: float) -> pl.DataFrame:
    price = pl.col(ENTRY_PRICE_COL)
    daily = (
        signals.filter(pl.col("Date").is_in(calendar))
        .with_columns(
            (price * (pl.col("label_raw_bp") - fee_bp) / 10.0).alias("net_pnl"),
            (price * pl.col("label_raw_bp") / 10.0).alias("gross_pnl"),
            (price * (pl.col("label_residual_bp") - fee_bp) / 10.0).alias("residual_net_pnl"),
            (price * 1000.0).alias("position"),
        )
        .group_by("Date")
        .agg(
            pl.col("net_pnl").sum(), pl.col("gross_pnl").sum(), pl.col("residual_net_pnl").sum(),
            pl.col("position").sum(), pl.len().alias("trades"), pl.col("QuoteCode").n_unique().alias("stocks"),
        )
    )
    return (
        pl.DataFrame({"Date": calendar}).join(daily, on="Date", how="left")
        .with_columns(
            pl.col("net_pnl").fill_null(0.0), pl.col("gross_pnl").fill_null(0.0), pl.col("residual_net_pnl").fill_null(0.0),
            pl.col("position").fill_null(0.0), pl.col("trades").fill_null(0), pl.col("stocks").fill_null(0),
        )
        .sort("Date")
    )


def _sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(252.0)) if len(x) > 1 and np.std(x, ddof=1) > 0 else math.nan


def _summary_row(signals: pl.DataFrame, *, variant: str, period: str, calendar: list[str], fee_bp: float, extra: dict | None = None) -> dict[str, object]:
    daily = _daily_frame(signals, calendar=calendar, fee_bp=fee_bp)
    subset = signals.filter(pl.col("Date").is_in(calendar))
    pnl = daily["net_pnl"].to_numpy()
    cumulative = np.cumsum(pnl)
    drawdown = cumulative - np.maximum.accumulate(cumulative) if len(pnl) else np.array([])
    net_pnl = float(pnl.sum())
    gross_pnl = float(daily["gross_pnl"].sum())
    residual_net = float(daily["residual_net_pnl"].sum())
    position = float(daily["position"].sum())
    months = daily.with_columns(pl.col("Date").str.slice(0, 6).alias("m")).group_by("m").agg(pl.col("net_pnl").sum())
    row = {
        "signal_variant": variant,
        "period": period,
        "fee_bp": fee_bp,
        "start": calendar[0] if calendar else None,
        "end": calendar[-1] if calendar else None,
        "calendar_days": len(calendar),
        "active_days": int((daily["trades"] > 0).sum()),
        "trades": int(daily["trades"].sum()),
        "stock_days": int(subset.select(pl.struct(["Date", "QuoteCode"]).n_unique()).item()) if len(subset) else 0,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "residual_net_pnl": residual_net,
        "cap_weighted_gross_bp": gross_pnl / position * 1e4 if position > 0 else math.nan,
        "cap_weighted_net_bp": net_pnl / position * 1e4 if position > 0 else math.nan,
        "cap_weighted_residual_net_bp": residual_net / position * 1e4 if position > 0 else math.nan,
        "mean_z": float(subset["label_z"].drop_nulls().drop_nans().mean()) if len(subset) else math.nan,
        "daily_sharpe": _sharpe(pnl),
        "residual_daily_sharpe": _sharpe(daily["residual_net_pnl"].to_numpy()),
        "daily_win_rate": float(np.mean(pnl > 0.0)) if len(pnl) else math.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else math.nan,
        "positive_months": int((months["net_pnl"] > 0).sum()),
        "months": int(months.height),
        "mean_daily_position": float(daily["position"].mean()) if len(daily) else math.nan,
        "mean_fill_seconds": float(subset["maker_buy_fill_seconds"].mean()) if len(subset) else math.nan,
    }
    if extra:
        row.update(extra)
    return row


def _monthly_rows(signals: pl.DataFrame, *, variant: str, fee_bp: float) -> list[dict[str, object]]:
    price = pl.col(ENTRY_PRICE_COL)
    return (
        signals.with_columns(
            pl.col("Date").str.slice(0, 6).alias("month"),
            (price * (pl.col("label_raw_bp") - fee_bp) / 10.0).alias("net_pnl"),
            (price * (pl.col("label_residual_bp") - fee_bp) / 10.0).alias("residual_net_pnl"),
            (price * 1000.0).alias("position"),
        )
        .group_by("month")
        .agg(pl.col("net_pnl").sum(), pl.col("residual_net_pnl").sum(), pl.col("position").sum(), pl.len().alias("trades"), pl.col("Date").n_unique().alias("active_days"))
        .with_columns(
            pl.lit(variant).alias("signal_variant"), pl.lit(fee_bp).alias("fee_bp"),
            (pl.col("net_pnl") / pl.col("position") * 1e4).alias("cap_weighted_net_bp"),
            (pl.col("residual_net_pnl") / pl.col("position") * 1e4).alias("cap_weighted_residual_net_bp"),
        )
        .sort("month")
        .to_dicts()
    )


def _select_variant(summary: pl.DataFrame, *, model_start: str) -> dict[str, object]:
    development = summary.filter(
        (pl.col("period") == "development")
        & pl.col("signal_variant").is_in(SELECTABLE_VARIANTS)
        & (pl.col("active_days") >= MIN_ACTIVE_DAYS)
        & (pl.col("trades") >= MIN_TRADES)
        & pl.col("daily_sharpe").is_finite()
    )
    order = {name: i for i, name in enumerate(SELECTABLE_VARIANTS)}
    rows = development.to_dicts()
    rows.sort(key=lambda r: (-float(r["daily_sharpe"]), order[str(r["signal_variant"])]))
    if not rows:
        raise RuntimeError("no model-gated variant meets the frozen development support rule")
    return {
        "selected_variant": str(rows[0]["signal_variant"]),
        "selection_period": f"{model_start}-{DEVELOPMENT_END}",
        "selection_metric": "calendar-filled net daily Sharpe at 19.3bp",
        "minimum_active_days": MIN_ACTIVE_DAYS,
        "minimum_trades": MIN_TRADES,
        "eligible_development_rows": rows,
        "holdout_not_used_for_selection": True,
    }


def _run_backtest(signals: pl.DataFrame, *, fee_bp: float):
    if signals.is_empty():
        raise ValueError("BackTest requires at least one selected signal")
    pdf = signals.to_pandas()
    backtest = BackTest(pdf, fee=fee_bp, side="long", entry_price_col=ENTRY_PRICE_COL, bp_col="label_raw_bp")
    stats, _ = backtest.run_analysis(return_daily=True, plot=False)
    expected = (pdf[ENTRY_PRICE_COL] * (pdf["label_raw_bp"] - fee_bp) / 10.0).sum()
    actual = float(backtest.df["PnL"].sum())
    if not np.isclose(expected, actual, rtol=1e-12, atol=1e-8):
        raise RuntimeError(f"BackTest PnL parity failed: expected={expected} actual={actual}")
    return backtest, _json_value(stats), backtest.daily_result.copy()


def _save_exact_plot(signals: pl.DataFrame, *, path: Path, fee_bp: float) -> tuple[dict[str, object], pd.DataFrame]:
    backtest, stats, daily = _run_backtest(signals, fee_bp=fee_bp)
    backtest._plot(daily)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(path, dpi=160, bbox_inches="tight")
    plt.close(plt.gcf())
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"BackTest plot was not saved: {path}")
    return stats, daily


def evaluate(thresholded: pl.DataFrame) -> dict[str, object]:
    sessions = _calendar_sessions()
    model_dates = thresholded.filter(pl.col("model_era"))["Date"].unique().sort().to_list()
    if not model_dates:
        raise RuntimeError("rolling models produced no common model-era date")
    model_start = model_dates[0]
    signals = build_variant_signals(thresholded)

    summary_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for name in ALL_VARIANTS:
        for period in PERIODS:
            summary_rows.append(_summary_row(signals[name], variant=name, period=period, calendar=_period_calendar(sessions, period, model_start), fee_bp=FEE_BP))
        monthly_rows.extend(_monthly_rows(signals[name], variant=name, fee_bp=FEE_BP))
    summary = pl.DataFrame(summary_rows)
    _atomic_parquet(summary, VARIANT_SUMMARY_PATH)
    _atomic_parquet(pl.DataFrame(monthly_rows), MONTHLY_SUMMARY_PATH)

    decision = _select_variant(summary, model_start=model_start)
    selected_name = str(decision["selected_variant"])
    selected = signals[selected_name]
    _atomic_parquet(selected, SELECTED_SIGNALS_PATH)

    fee_rows = [
        _summary_row(selected, variant=selected_name, period=period, calendar=_period_calendar(sessions, period, model_start), fee_bp=fee)
        for fee in FEE_GRID_BP
        for period in PERIODS
    ]
    _atomic_parquet(pl.DataFrame(fee_rows), FEE_SENSITIVITY_PATH)

    # fill-horizon sensitivity: re-gate execution rows only (no retraining)
    fill_rows: list[dict[str, object]] = []
    for horizon in FILL_HORIZON_SENSITIVITY:
        for exclude_undercut in (False, True):
            source = thresholded.filter(pl.col("maker_fill_branch") != "undercut_immediate") if exclude_undercut else thresholded
            variants = build_variant_signals(source, fill_horizon=horizon)
            for name in (selected_name, "static_only__all", "static_only__top_list"):
                for period in PERIODS:
                    fill_rows.append(
                        _summary_row(
                            variants[name], variant=name, period=period, calendar=_period_calendar(sessions, period, model_start),
                            fee_bp=FEE_BP, extra={"fill_horizon_seconds": horizon, "exclude_undercut_immediate": exclude_undercut},
                        )
                    )
    _atomic_parquet(pl.DataFrame(fill_rows), FILL_SENSITIVITY_PATH)

    full_stats, full_daily = _save_exact_plot(selected, path=BACKTEST_DIR / "backtest_full.png", fee_bp=FEE_BP)
    holdout = selected.filter(pl.col("Date") >= HOLDOUT_START)
    holdout_stats, _ = _save_exact_plot(holdout, path=BACKTEST_DIR / "backtest_2026_holdout.png", fee_bp=FEE_BP)
    # companion plot: same gate on the other universe
    gate, universe = selected_name.split("__")
    other = f"{gate}__{'top_list' if universe == 'all' else 'all'}"
    other_stats = None
    if not signals[other].is_empty():
        other_stats, _ = _save_exact_plot(signals[other], path=BACKTEST_DIR / f"backtest_full_{other}.png", fee_bp=FEE_BP)
    _atomic_parquet(pl.from_pandas(full_daily.reset_index()), SELECTED_DAILY_PATH)
    decision.update(
        {
            "model_start": model_start,
            "full_backtest_stats": full_stats,
            "holdout_backtest_stats": holdout_stats,
            "companion_variant": other,
            "companion_full_backtest_stats": other_stats,
            "selected_signal_rows": len(selected),
            "selected_signal_days": selected["Date"].n_unique(),
        }
    )
    _atomic_json(_json_value(decision), SELECTION_DECISION_PATH)
    return decision


def write_run_manifest(decision: dict[str, object], args: argparse.Namespace) -> None:
    artifact_paths = (
        STUDY_DIR / "study.py",
        STUDY_DIR / "build_dataset.py",
        STUDY_DIR / "run_model_backtest.py",
        EVENT_INDEX_PATH,
        EVENT_INDEX_PATH.with_suffix(".manifest.json"),
        EVENT_DATASET_DIR / "_dataset_manifest.json",
        MODEL_INPUT_PATH,
        MODEL_DIR / "model_input_audit.json",
        PREDICTION_PATH,
        MODEL_DIR / "timing_model_audit.json",
        MODEL_DIR / "selection_model_audit.json",
        TIMING_THRESHOLD_PATH,
        SELECTION_THRESHOLD_PATH,
        THRESHOLDED_PATH,
        VARIANT_SUMMARY_PATH,
        MONTHLY_SUMMARY_PATH,
        FEE_SENSITIVITY_PATH,
        FILL_SENSITIVITY_PATH,
        SELECTED_SIGNALS_PATH,
        SELECTED_DAILY_PATH,
        SELECTION_DECISION_PATH,
    )
    source_paths = (
        PROJECT_ROOT / "src/research/intradaySignal/framework/labels.py",
        PROJECT_ROOT / "src/research/intradaySignal/framework/model.py",
        PROJECT_ROOT / "src/research/intradaySignal/framework/prior_thresholds.py",
        PROJECT_ROOT / "src/research/intradaySignal/framework/profiles/filllots_at_high.py",
        PROJECT_ROOT / "src/research/intradaySignal/framework/profiles/fhl.py",
        PROJECT_ROOT / "src/research/rolling_model_multiLabel.py",
        PROJECT_ROOT / "src/research/backTest.py",
        PROJECT_ROOT / "src/research/intradaySignal/experiments/20260916_defensive_long_universe/defensive_list.py",
        PROJECT_ROOT / "uv.lock",
    )
    versions = {}
    for package in ("numpy", "pandas", "polars", "scikit-learn", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        "contract": "fh_pos_breakout_long_maker_signal_run/v1",
        "study_start": STUDY_START,
        "study_end": STUDY_END,
        "development_end": DEVELOPMENT_END,
        "holdout_start": HOLDOUT_START,
        "direction": "long",
        "entry": {"price_col": ENTRY_PRICE_COL, "liquidity": "maker", "fill_delay_ms": 100, "fill_horizon_seconds": FILL_HORIZON_SECONDS},
        "candidate_id": "fh_pos_breakout",
        "label_profile_id": LABEL_SPEC.profile_id,
        "label_profile_hash": LABEL_SPEC.fingerprint,
        "feature_manifest": FEATURE_MANIFEST.canonical_dict(),
        "feature_manifest_hash": FEATURE_MANIFEST.fingerprint,
        "model": {
            "preprocessing_mode": "causal_v2", "feature_screening": False,
            "timing_lookback_days": TIMING_LOOKBACK_DAYS, "selection_lookback_days": SELECTION_LOOKBACK_DAYS,
            "ridge_alpha": RIDGE_ALPHA, "training_position_cap_wan": TRAINING_POSITION_CAP_WAN, "n_jobs": args.n_jobs,
            "timing_quantile": TIMING_QUANTILE, "selection_quantile": SELECTION_QUANTILE,
        },
        "fee_bp": FEE_BP,
        "hedge": "none (label residual/z used only as model targets)",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions},
        "selection_decision": decision,
        "artifacts": {str(p.relative_to(STUDY_DIR)): {"size": p.stat().st_size if p.exists() else None, "sha256": _sha256(p)} for p in artifact_paths},
        "source_files": {str(p.relative_to(PROJECT_ROOT)): {"size": p.stat().st_size if p.exists() else None, "sha256": _sha256(p)} for p in source_paths},
    }
    _atomic_json(_json_value(payload), RUN_MANIFEST_PATH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--force-input", action="store_true")
    parser.add_argument("--force-model", action="store_true")
    parser.add_argument("--force-thresholds", action="store_true")
    args = parser.parse_args()
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be zero")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    model_input = build_model_input(force=args.force_input)
    predictions = train_models(model_input, n_jobs=args.n_jobs, force=args.force_model or args.force_input)
    thresholded = apply_thresholds(predictions, force=args.force_thresholds or args.force_model or args.force_input)
    decision = evaluate(thresholded)
    write_run_manifest(decision, args)
    print(json.dumps(_json_value({k: v for k, v in decision.items() if k != "eligible_development_rows"}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
